#!/usr/bin/env python3
"""
merge_pharmit_output.py — merge SDF archives, deduplicate by InChIKey,
and write a merged SDF (with 3D poses + vendor ID properties) and a CSV
lookup table (SMILES + affinity + per-vendor IDs).

Each output molecule carries a Structure_ID SDF property set to the best
available vendor ID (priority: CHEMBL > Enamine > ZINC > PubChem > MCULE >
MolPort > CSC > ChemDiv > ChemSpace > LabNetwork > other). This property is
used as the compound identifier by downstream tools such as gnina.py
(--id-column Structure_ID).

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


# --- vendor ID extraction -------------------------------------------------

VENDOR_PATTERNS = {
    "Enamine_ID":     re.compile(r'^Z\d{6,}$'),
    "ZINC_ID":        re.compile(r'^ZINC\d+$'),
    "PubChem_ID":     re.compile(r'^PubChem-\d+$'),
    "CHEMBL_ID":      re.compile(r'^CHEMBL\d+$'),
    "MCULE_ID":       re.compile(r'^MCULE-\d+$'),
    "MolPort_ID":     re.compile(r'^(?:MolPort|Molport)-\d{3}-\d{3}-\d{3}$'),
    "CSC_ID":         re.compile(r'^CSC\d+$'),
    "ChemDiv_ID":     re.compile(r'^[A-Z]\d{3,4}-\d{4,}[A-Z0-9]*$'),
    "ChemSpace_ID":   re.compile(r'^CSSS\d+$'),
    "LabNetwork_ID":  re.compile(r'^LN\d{6,}$'),
}

_INCHIKEY_RE = re.compile(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$')

NAME_PRIORITY = ["CHEMBL_ID", "Enamine_ID", "ZINC_ID", "PubChem_ID",
                 "MCULE_ID", "MolPort_ID", "CSC_ID", "ChemDiv_ID",
                 "ChemSpace_ID", "LabNetwork_ID"]


def parse_vendor_ids(name_field: str) -> dict[str, list[str]]:
    """Split a space-separated name field into per-vendor ID lists."""
    ids: dict[str, list[str]] = {k: [] for k in VENDOR_PATTERNS}
    ids["other_IDs"] = []
    for tok in name_field.split():
        if _INCHIKEY_RE.match(tok):
            continue  # InChIKey used as name (e.g. Mcule-Ultimate) — skip
        matched = False
        for vendor, pat in VENDOR_PATTERNS.items():
            if pat.match(tok):
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
    with open_sdf(path) as fh:
        suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False)
        for mol in suppl:
            if mol is not None:
                yield mol


def write_sdf_gz(mols_data: list, outpath: str):
    with gzip.open(outpath, "wt") as fh:
        writer = Chem.SDWriter(fh)   # type: ignore[arg-type]
        for i, (mol, vendor_ids, inchikey) in enumerate(mols_data, 1):
            for vendor, id_list in vendor_ids.items():
                if id_list:
                    mol.SetProp(vendor, " ".join(id_list))
                elif mol.HasProp(vendor):
                    mol.ClearProp(vendor)
            compound_id = f"ID_{i:06d}"
            mol.SetProp("_Name", best_display_name(vendor_ids, inchikey))
            mol.SetProp("Compound_ID", compound_id)
            mol.SetProp("Structure_ID", compound_id)
            AllChem.Compute2DCoords(mol)
            writer.write(mol)
        writer.close()


def write_csv(entries: list, outpath: str):
    """Write SMILES, affinity, and per-vendor ID columns for each unique compound."""
    vendor_cols = list(VENDOR_PATTERNS.keys()) + ["other_IDs"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Compound_ID", "SMILES", "affinity"] + vendor_cols)
        for i, (mol, vendor_ids, affinity, inchikey) in enumerate(entries, 1):
            smi = Chem.MolToSmiles(mol)
            compound_id = f"ID_{i:06d}"
            row = [compound_id, smi, affinity] + [" ".join(vendor_ids.get(v, [])) for v in vendor_cols]
            writer.writerow(row)


# --- main logic ----------------------------------------------------------

def aggregate(input_files: list[str], output_sdf: str | None, output_csv: str | None):
    # inchikey -> (mol, vendor_ids, affinity)
    registry: dict[str, tuple] = {}
    skipped = 0

    for path in input_files:
        print(f"Reading {path} …", file=sys.stderr)
        for mol in read_mols(path):
            inchi = MolToInchi(mol)
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
                existing_mol, existing_ids, existing_aff = registry[key]
                merged_ids = merge_vendor_ids(existing_ids, vendor_ids)
                # keep the pose with better (more negative) affinity
                if affinity < existing_aff:
                    registry[key] = (mol, merged_ids, affinity)
                else:
                    registry[key] = (existing_mol, merged_ids, existing_aff)
            else:
                registry[key] = (mol, vendor_ids, affinity)


    print(f"  Unique structures: {len(registry)}", file=sys.stderr)
    if skipped:
        print(f"  Skipped (no InChI): {skipped}", file=sys.stderr)

    # sort by affinity (best first); carry InChIKey as fallback ID
    sorted_entries = [
        (mol, ids, aff, key)
        for key, (mol, ids, aff) in sorted(registry.items(), key=lambda x: x[1][2])
    ]

    if output_csv:
        write_csv(sorted_entries, output_csv)
        print(f"Wrote {len(sorted_entries)} rows → {output_csv}", file=sys.stderr)

    if output_sdf:
        write_sdf_gz([(mol, ids, key) for mol, ids, _, key in sorted_entries], output_sdf)
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
