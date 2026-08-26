#!/usr/bin/env python3
"""
merge_pharmit_output.py — merge SDF archives, deduplicate by InChIKey,
and write a merged SDF (2D layout + vendor ID properties, docking affinity
kept as an SDF property) and a CSV lookup table (SMILES + affinity + per-vendor
IDs). Deduplication happens before the 2D layout is generated, against the
original sanitized docking-pose molecules, so it is unaffected by the 2D
flattening below.

Pharmit's per-record _Name field is a cross-reference list of that compound's
IDs across every database it's known in — not just the database a given input
file was searched against (the same compound's full ID list shows up
verbatim, just reordered, across all the files it happens to hit in). So
vendor IDs are classified per-token by matching each token's own ID format
(VENDOR_PATTERNS below) against every input file, regardless of which file a
token came from; an input file's name has no bearing on ID classification.

Each output molecule is assigned a sequential Compound_ID (ID_000001,
ID_000002, …) that is guaranteed unique regardless of vendor ID availability.
Structure_ID is set to the same value for compatibility with downstream tools
such as gnina.py (--id-column Structure_ID). The molecule _Name retains the
best available vendor ID for display in molecule viewers (priority: CHEMBL >
Enamine > ZINC > PubChem > MCULE > MolPort > CSC > ChemDiv > ChemSpace >
LabNetwork > NSC > MCULE-Ultimate > other > InChIKey). MCULE-Ultimate is a
virtual make-on-demand space with no persistent catalog numbers — its
compounds are addressed by InChIKey, so InChIKey-shaped tokens now get their
own column instead of being silently discarded as noise.

Usage:
    python merge_pharmit_output.py [FILE1.sdf.gz FILE2.sdf.gz ...] \
        [-o OUTPUT.sdf.gz] [--csv OUTPUT.csv]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchi
from rdkit.Chem import InchiToInchiKey
from rdkit.Chem.MolStandardize import rdMolStandardize


# --- SMILES standardization -----------------------------------------------

_LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize(mol: Chem.Mol) -> Chem.Mol | None:
    """Strip salts and neutralize charges, returning a flat, canonical mol.

    Used to derive the canonical SMILES and dedup key so that different salt
    forms / protonation states of the same compound collapse to one entry.
    The original 3D pose mol is left untouched for SDF output.
    """
    flat = Chem.RemoveHs(mol)
    try:
        flat = _LARGEST_FRAGMENT_CHOOSER.choose(flat)
        flat = _UNCHARGER.uncharge(flat)
        Chem.SanitizeMol(flat)
    except (Chem.KekulizeException, Chem.AtomValenceException, ValueError, RuntimeError):
        return None
    return flat


# --- vendor ID extraction -------------------------------------------------

VENDOR_PATTERNS = {
    "Enamine_ID":     re.compile(r'^Z\d{6,}$'),
    # Pharmit often lists a compound's legacy unprefixed ZINC ID (e.g. "69514405")
    # right alongside its prefixed one (e.g. "ZINC000069514405"); fold both forms
    # into ZINC_ID rather than leaving the bare digits to fall through to other_IDs.
    "ZINC_ID":        re.compile(r'^(?:ZINC\d+|\d+)$'),
    "PubChem_ID":     re.compile(r'^PubChem-\d+$'),
    "CHEMBL_ID":      re.compile(r'^CHEMBL\d+$'),
    "MCULE_ID":       re.compile(r'^MCULE-\d+$'),
    "MolPort_ID":     re.compile(r'^(?:MolPort|Molport)-\d{3}-\d{3}-\d{3}$'),
    "CSC_ID":         re.compile(r'^CSC\d+$'),
    "ChemDiv_ID":     re.compile(r'^ChemDiv-[A-Z0-9]{4}-\d{4}[A-Z]?$'),
    "ChemSpace_ID":   re.compile(r'^CSSS\d+$'),
    "LabNetwork_ID":  re.compile(r'^LN\d+$'),
    "NSC_ID":         re.compile(r'^NSC\d+$'),
    # MCULE-Ultimate is a virtual make-on-demand space too large for
    # persistent catalog numbers; its compounds are addressed by InChIKey.
    "MCULE-Ultimate_ID": re.compile(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$'),
}

NAME_PRIORITY = ["CHEMBL_ID", "Enamine_ID", "ZINC_ID", "PubChem_ID",
                 "MCULE_ID", "MolPort_ID", "CSC_ID", "ChemDiv_ID",
                 "ChemSpace_ID", "LabNetwork_ID", "NSC_ID", "MCULE-Ultimate_ID"]


def _normalize_zinc(tok: str) -> str:
    """Collapse ZINC's various ID spellings (bare digits, 8-digit legacy
    padding, 12-digit modern padding) to one canonical "ZINC" + 12-digit
    form, so the same accession doesn't show up multiple times per compound."""
    digits = tok[4:] if tok.startswith("ZINC") else tok
    return f"ZINC{int(digits):012d}"


def parse_vendor_ids(name_field: str) -> dict[str, list[str]]:
    """Split a space-separated name field into per-vendor ID lists, by
    matching each token's own ID format — not by which input file it's in
    (pharmit's _Name field cross-references every database a compound is
    known in, regardless of which database was actually searched)."""
    ids: dict[str, list[str]] = {k: [] for k in VENDOR_PATTERNS}
    ids["other_IDs"] = []
    for tok in name_field.split():
        matched = False
        for vendor, pat in VENDOR_PATTERNS.items():
            if pat.match(tok):
                if vendor == "ZINC_ID":
                    tok = _normalize_zinc(tok)
                if tok not in ids[vendor]:
                    ids[vendor].append(tok)
                matched = True
                break
        if not matched:
            ids["other_IDs"].append(tok)
    return ids


def merge_vendor_ids(a: dict, b: dict) -> dict:
    merged = {}
    for key in set(a) | set(b):
        combined = list(dict.fromkeys(a.get(key, []) + b.get(key, [])))
        merged[key] = combined
    return merged


def best_display_name(ids: dict, inchikey: str = "") -> str:
    for vendor in NAME_PRIORITY:
        if ids.get(vendor):
            return ids[vendor][0]
    others = ids.get("other_IDs", [])
    if others:
        return others[0]
    return inchikey if inchikey else "unknown"


# --- SDF I/O -------------------------------------------------------------

def open_sdf(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rb")
    return open(p, "rb")


def read_mols(path: str):
    """Yield every record's mol, including None for records that failed to
    parse/sanitize, so callers can count them alongside other skips."""
    with open_sdf(path) as fh:
        suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
        yield from suppl


def write_sdf_gz(mols_data: list, outpath: str):
    with gzip.open(outpath, "wt") as fh:
        writer = Chem.SDWriter(fh)   # type: ignore[arg-type]
        for i, (mol, vendor_ids, inchikey) in enumerate(mols_data, 1):
            for vendor, id_list in vendor_ids.items():
                if id_list:
                    mol.SetProp(vendor, ",".join(id_list))
                elif mol.HasProp(vendor):
                    mol.ClearProp(vendor)
            compound_id = f"ID_{i:06d}"
            mol.SetProp("_Name", best_display_name(vendor_ids, inchikey))
            mol.SetProp("Compound_ID", compound_id)
            mol.SetProp("Structure_ID", compound_id)
            # Docking-pose Z coords are non-zero absolute pocket coordinates
            # (~477-480 A); DataWarrior renders those as invisible structures,
            # so replace them with a proper 2D layout. Affinity/vendor IDs are
            # kept as SDF properties; only the geometry is flattened, and only
            # after dedup (see module docstring), so this doesn't affect
            # duplicate detection.
            AllChem.Compute2DCoords(mol)
            writer.write(mol)
        writer.close()


def write_csv(entries: list, outpath: str):
    """Write SMILES, affinity, and per-vendor ID columns for each unique compound."""
    vendor_cols = list(VENDOR_PATTERNS.keys()) + ["other_IDs"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Compound_ID", "SMILES", "affinity"] + vendor_cols)
        for i, (mol, vendor_ids, affinity, inchikey, smi) in enumerate(entries, 1):
            compound_id = f"ID_{i:06d}"
            row = [compound_id, smi, affinity] + [",".join(vendor_ids.get(v, [])) for v in vendor_cols]
            writer.writerow(row)


# --- main logic ----------------------------------------------------------

def aggregate(input_files: list[str], output_sdf: str | None, output_csv: str | None):
    # inchikey -> (mol, vendor_ids, affinity, canonical_smiles)
    registry: dict[str, tuple] = {}
    skipped = 0

    for path in input_files:
        print(f"Reading {path} …", file=sys.stderr)
        for mol in read_mols(path):
            if mol is None:
                skipped += 1
                continue
            flat = standardize(mol)
            if flat is None:
                skipped += 1
                continue
            smi = Chem.MolToSmiles(flat)
            inchi = MolToInchi(flat)
            if inchi is None:
                skipped += 1
                continue
            key = InchiToInchiKey(inchi)
            if key is None:
                skipped += 1
                continue

            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            vendor_ids = parse_vendor_ids(name)
            affinity = float(mol.GetProp("minimizedAffinity")) if mol.HasProp("minimizedAffinity") else 0.0

            if key in registry:
                existing_mol, existing_ids, existing_aff, existing_smi = registry[key]
                merged_ids = merge_vendor_ids(existing_ids, vendor_ids)
                # keep the pose with better (more negative) affinity
                if affinity < existing_aff:
                    registry[key] = (mol, merged_ids, affinity, smi)
                else:
                    registry[key] = (existing_mol, merged_ids, existing_aff, existing_smi)
            else:
                registry[key] = (mol, vendor_ids, affinity, smi)


    print(f"  Unique structures: {len(registry)}", file=sys.stderr)
    if skipped:
        print(f"  Skipped (parse/sanitize or InChI failure): {skipped}", file=sys.stderr)

    # sort by affinity (best first); carry InChIKey as fallback ID
    sorted_entries = [
        (mol, ids, aff, key, smi)
        for key, (mol, ids, aff, smi) in sorted(registry.items(), key=lambda x: x[1][2])
    ]

    if output_csv:
        write_csv(sorted_entries, output_csv)
        print(f"Wrote {len(sorted_entries)} rows → {output_csv}", file=sys.stderr)

    if output_sdf:
        write_sdf_gz([(mol, ids, key) for mol, ids, _, key, _ in sorted_entries], output_sdf)
        print(f"Wrote {len(sorted_entries)} molecules → {output_sdf}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", metavar="FILE.sdf[.gz]", help="Input SDF archives")
    parser.add_argument("-o", "--output", default="aggregated.sdf.gz",
                        help="Output SDF (default: aggregated.sdf.gz; empty string to skip)")
    parser.add_argument("--csv", default="pharmit_merged.csv",
                        help="Output CSV (default: pharmit_merged.csv; empty string to skip)")
    args = parser.parse_args()

    aggregate(args.inputs, args.output or None, args.csv or None)


if __name__ == "__main__":
    main()
