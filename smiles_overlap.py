#!/usr/bin/env python3
"""
smiles_overlap.py

Calculate the exact structural overlap between two SMILES files using RDKit.
Molecules are sanitized and reduced to a canonical form before comparison, so
overlap reflects identical chemical structures rather than identical SMILES
strings (e.g. "CCO" and "OCC" are recognised as the same molecule).

Usage:
    python smiles_overlap.py file1.smi file2.smi
    python smiles_overlap.py file1.smi file2.smi --no-stereo
    python smiles_overlap.py file1.smi file2.smi --use-inchikey
    python smiles_overlap.py file1.smi file2.smi -o shared.smi

Dependencies:
    pip install rdkit
"""

import argparse
import sys

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_smiles(filepath):
    """Parse a whitespace- or comma-delimited SMILES file.

    Accepted line formats (auto-detected):
        SMILES [name]
        SMILES,name
        name,SMILES
    Lines starting with '#' are treated as comments.

    Returns a list of (mol, name, original_smiles) tuples. Molecules that
    fail sanitization are skipped with a warning.
    """
    entries = []
    with open(filepath) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if not parts:
                continue
            smiles, name = parts[0], f"mol_{i}"
            if len(parts) >= 2:
                if _looks_like_smiles(parts[0]):
                    name = parts[1]
                else:
                    smiles, name = parts[1], parts[0]

            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                print(f"  WARNING [{filepath}]: invalid/unsanitizable SMILES "
                      f"on line {i + 1}: '{smiles}' — skipping", file=sys.stderr)
                continue
            entries.append((mol, name, smiles))
    return entries


def _looks_like_smiles(s):
    smiles_chars = set("CNOPSFClBrIcnops()[]=#@+-/\\0123456789.")
    return sum(c in smiles_chars for c in s) / max(len(s), 1) > 0.6


# ---------------------------------------------------------------------------
# Canonicalization / keying
# ---------------------------------------------------------------------------

def structure_key(mol, isomeric=True, use_inchikey=False):
    """Return a hashable key representing the exact structure of mol."""
    if use_inchikey:
        return Chem.MolToInchiKey(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def index_by_structure(entries, isomeric=True, use_inchikey=False):
    """Group (mol, name, original_smiles) entries by structural key.

    Returns dict: key -> list of (name, original_smiles)
    """
    index = {}
    for mol, name, orig in entries:
        key = structure_key(mol, isomeric=isomeric, use_inchikey=use_inchikey)
        index.setdefault(key, []).append((name, orig))
    return index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the exact structural overlap between two SMILES files "
            "using RDKit. Molecules are sanitized and canonicalized before "
            "comparison."
        )
    )
    parser.add_argument("file1", help="First SMILES file.")
    parser.add_argument("file2", help="Second SMILES file.")
    parser.add_argument(
        "-o", "--output",
        help="Optional path to write the overlapping (shared) structures as SMILES."
    )
    parser.add_argument(
        "--no-stereo", action="store_true",
        help="Ignore stereochemistry when comparing structures (default: stereo-aware)."
    )
    parser.add_argument(
        "--use-inchikey", action="store_true",
        help=(
            "Compare via InChIKey instead of canonical SMILES. More tolerant of "
            "equivalent representations (e.g. charge/tautomer normalisation "
            "differences), at the cost of being a hashed rather than direct match."
        )
    )
    return parser.parse_args()


def main():
    args = parse_args()
    isomeric = not args.no_stereo

    print(f"File 1 : {args.file1}")
    print(f"File 2 : {args.file2}")
    print(f"Key    : {'InChIKey' if args.use_inchikey else 'canonical SMILES'}"
          f"{'' if isomeric or args.use_inchikey else ' (stereo ignored)'}")
    print()

    entries1 = load_smiles(args.file1)
    entries2 = load_smiles(args.file2)
    print(f"  {len(entries1)} valid molecules loaded from '{args.file1}'")
    print(f"  {len(entries2)} valid molecules loaded from '{args.file2}'\n")

    if not entries1 or not entries2:
        print("ERROR: one or both files contained no valid molecules.", file=sys.stderr)
        sys.exit(1)

    index1 = index_by_structure(entries1, isomeric=isomeric, use_inchikey=args.use_inchikey)
    index2 = index_by_structure(entries2, isomeric=isomeric, use_inchikey=args.use_inchikey)

    keys1, keys2 = set(index1), set(index2)
    shared = keys1 & keys2
    union = keys1 | keys2

    n1_unique_structs, n2_unique_structs = len(keys1), len(keys2)
    n_shared = len(shared)
    jaccard = n_shared / len(union) if union else 0.0

    print("Results (unique structures):")
    print(f"  Unique in file 1      : {n1_unique_structs}")
    print(f"  Unique in file 2      : {n2_unique_structs}")
    print(f"  Shared (overlap)      : {n_shared}")
    print(f"  Union                 : {len(union)}")
    print(f"  Overlap / file 1      : {n_shared / n1_unique_structs:.2%}")
    print(f"  Overlap / file 2      : {n_shared / n2_unique_structs:.2%}")
    print(f"  Jaccard index         : {jaccard:.4f}")

    if shared:
        print("\nOverlapping compounds:")
        for key in sorted(shared):
            names1 = ", ".join(n for n, _ in index1[key])
            names2 = ", ".join(n for n, _ in index2[key])
            print(f"  {key}")
            print(f"    file1: {names1}")
            print(f"    file2: {names2}")

    if args.output:
        with open(args.output, "w") as fh:
            fh.write("smiles\tfile1_ids\tfile2_ids\n")
            for key in sorted(shared):
                names1 = ",".join(n for n, _ in index1[key])
                names2 = ",".join(n for n, _ in index2[key])
                fh.write(f"{key}\t{names1}\t{names2}\n")
        print(f"\nWrote {n_shared} shared structures to '{args.output}'")


if __name__ == "__main__":
    main()
