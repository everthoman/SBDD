#!/usr/bin/env python3
"""
merge_pharmit_output.py — merge SDF archives, deduplicate by InChIKey,
and write a merged SDF (2D layout + vendor ID properties, docking affinity
kept as an SDF property) and a CSV lookup table (SMILES + affinity + per-vendor
IDs). Deduplication happens before the 2D layout is generated, against the
original sanitized docking-pose molecules, so it is unaffected by the 2D
flattening below.

Each input file is one vendor/database's search results. The vendor name is
taken from the filename's trailing _VENDOR segment (e.g.
search1_hits_Enamine.sdf.gz → "Enamine"), or can be set explicitly with
VENDOR=FILE.sdf[.gz]; either way it's used verbatim as that file's ID column
name — not inferred from the compound name text.

Each output molecule is assigned a sequential Compound_ID (ID_000001,
ID_000002, …) that is guaranteed unique regardless of vendor ID availability.
Structure_ID is set to the same value for compatibility with downstream tools
such as gnina.py (--id-column Structure_ID). The molecule _Name retains the
best available vendor ID for display in molecule viewers (priority: order the
vendors were given on the command line, then InChIKey).

Usage:
    python merge_pharmit_output.py results_hits_Enamine.sdf.gz results_hits_ZINC.sdf.gz ... \
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

_INCHIKEY_RE = re.compile(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$')


def extract_ids(name_field: str) -> list[str]:
    """Split a space-separated name field into ID tokens, dropping any
    token that's just an InChIKey used as a name (e.g. Mcule-Ultimate)."""
    return [tok for tok in name_field.split() if not _INCHIKEY_RE.match(tok)]


def merge_vendor_ids(a: dict, b: dict) -> dict:
    merged = {}
    for key in set(a) | set(b):
        combined = list(dict.fromkeys(a.get(key, []) + b.get(key, [])))
        merged[key] = combined
    return merged


def best_display_name(ids: dict, vendor_order: list[str], inchikey: str = "") -> str:
    for vendor in vendor_order:
        if ids.get(vendor):
            return ids[vendor][0]
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


def write_sdf_gz(mols_data: list, outpath: str, vendor_order: list[str]):
    with gzip.open(outpath, "wt") as fh:
        writer = Chem.SDWriter(fh)   # type: ignore[arg-type]
        for i, (mol, vendor_ids, inchikey) in enumerate(mols_data, 1):
            for vendor in vendor_order:
                prop = f"{vendor}_ID"
                id_list = vendor_ids.get(vendor, [])
                if id_list:
                    mol.SetProp(prop, ",".join(id_list))
                elif mol.HasProp(prop):
                    mol.ClearProp(prop)
            compound_id = f"ID_{i:06d}"
            mol.SetProp("_Name", best_display_name(vendor_ids, vendor_order, inchikey))
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


def write_csv(entries: list, outpath: str, vendor_order: list[str]):
    """Write SMILES, affinity, and per-vendor ID columns for each unique compound."""
    vendor_cols = [f"{v}_ID" for v in vendor_order]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Compound_ID", "SMILES", "affinity"] + vendor_cols)
        for i, (mol, vendor_ids, affinity, inchikey) in enumerate(entries, 1):
            # Docking poses carry explicit Hs (removeHs=False on read); strip
            # them for the lookup-table SMILES so it's a normal canonical
            # SMILES instead of e.g. "[H]OC([H])([H])C([H])([H])[H]".
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            compound_id = f"ID_{i:06d}"
            row = [compound_id, smi, affinity] + [",".join(vendor_ids.get(v, [])) for v in vendor_order]
            writer.writerow(row)


# --- main logic ----------------------------------------------------------

def aggregate(inputs: list[tuple[str, str]], output_sdf: str | None, output_csv: str | None):
    # inchikey -> (mol, vendor_ids, affinity)
    registry: dict[str, tuple] = {}
    skipped = 0
    vendor_order: list[str] = []

    for vendor, path in inputs:
        if vendor not in vendor_order:
            vendor_order.append(vendor)
        print(f"Reading {path} (vendor={vendor}) …", file=sys.stderr)
        for mol in read_mols(path):
            if mol is None:
                skipped += 1
                continue
            inchi = MolToInchi(mol)
            if inchi is None:
                skipped += 1
                continue
            key = InchiToInchiKey(inchi)
            if key is None:
                skipped += 1
                continue

            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            ids = extract_ids(name)
            vendor_ids = {vendor: ids} if ids else {}
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
        print(f"  Skipped (parse/sanitize or InChI failure): {skipped}", file=sys.stderr)

    # sort by affinity (best first); carry InChIKey as fallback ID
    sorted_entries = [
        (mol, ids, aff, key)
        for key, (mol, ids, aff) in sorted(registry.items(), key=lambda x: x[1][2])
    ]

    if output_csv:
        write_csv(sorted_entries, output_csv, vendor_order)
        print(f"Wrote {len(sorted_entries)} rows → {output_csv}", file=sys.stderr)

    if output_sdf:
        write_sdf_gz([(mol, ids, key) for mol, ids, _, key in sorted_entries], output_sdf, vendor_order)
        print(f"Wrote {len(sorted_entries)} molecules → {output_sdf}", file=sys.stderr)


def vendor_from_filename(path: str) -> str:
    """Derive the vendor name from a file named like blabla_blablabla_VENDOR.sdf[.gz]
    — the last underscore-separated segment of the filename stem."""
    name = Path(path).name
    for suffix in (".sdf.gz", ".sdf"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    vendor = name.rsplit("_", 1)[-1]
    if not vendor:
        raise argparse.ArgumentTypeError(
            f"could not derive a vendor name from filename {path!r} "
            f"(expected ..._VENDOR.sdf[.gz]); use VENDOR={path} to set it explicitly"
        )
    return vendor


def parse_input_arg(arg: str) -> tuple[str, str]:
    """VENDOR=FILE.sdf[.gz] for an explicit vendor tag, or just FILE.sdf[.gz]
    to derive the vendor from the filename's trailing _VENDOR segment."""
    if "=" in arg:
        vendor, _, path = arg.partition("=")
        vendor = vendor.strip()
        if not vendor:
            raise argparse.ArgumentTypeError(f"empty vendor name in {arg!r}")
        return vendor, path
    return vendor_from_filename(arg), arg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=parse_input_arg,
                        metavar="FILE.sdf[.gz]",
                        help="Input SDF archives, named ..._VENDOR.sdf[.gz] "
                             "(vendor is the filename's trailing _VENDOR segment), "
                             "or VENDOR=FILE.sdf[.gz] to set the vendor explicitly")
    parser.add_argument("-o", "--output", default="aggregated.sdf.gz",
                        help="Output SDF (default: aggregated.sdf.gz; empty string to skip)")
    parser.add_argument("--csv", default="pharmit_merged.csv",
                        help="Output CSV (default: pharmit_merged.csv; empty string to skip)")
    args = parser.parse_args()

    aggregate(args.inputs, args.output or None, args.csv or None)


if __name__ == "__main__":
    main()
