#!/usr/bin/env python3
"""
aggregate_structures.py — merge SDF archives, deduplicate by InChIKey,
and promote vendor IDs from the name field into separate SDF properties.

Usage:
    python aggregate_structures.py [FILE1.sdf.gz FILE2.sdf.gz ...] -o OUTPUT.sdf.gz
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

from rdkit import Chem
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

# InChIKey pattern — names matching this are structure identifiers, not catalog IDs
_INCHIKEY_RE = re.compile(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$')

# priority order used when choosing the display name for a merged entry
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


def best_display_name(ids: dict) -> str:
    for vendor in NAME_PRIORITY:
        if ids.get(vendor):
            return ids[vendor][0]
    others = ids.get("other_IDs", [])
    return others[0] if others else "unknown"


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
        for mol, vendor_ids in mols_data:
            for vendor, id_list in vendor_ids.items():
                if id_list:
                    mol.SetProp(vendor, " ".join(id_list))
                elif mol.HasProp(vendor):
                    mol.ClearProp(vendor)
            mol.SetProp("_Name", best_display_name(vendor_ids))
            writer.write(mol)
        writer.close()


# --- main logic ----------------------------------------------------------

def aggregate(input_files: list[str], output: str):
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

    # sort by affinity (best first)
    sorted_entries = sorted(registry.values(), key=lambda x: x[2])
    write_sdf_gz([(mol, ids) for mol, ids, _ in sorted_entries], output)
    print(f"Wrote {len(sorted_entries)} molecules → {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", metavar="FILE.sdf[.gz]", help="Input SDF archives")
    parser.add_argument("-o", "--output", default="aggregated.sdf.gz", help="Output SDF (default: aggregated.sdf.gz)")
    args = parser.parse_args()
    aggregate(args.inputs, args.output)


if __name__ == "__main__":
    main()
