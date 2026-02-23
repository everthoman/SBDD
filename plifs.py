#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
This script calculates protein-ligand interaction fingerprint similarities using ODDT.

Purpose:
- Provides a protein structure (PDB), a reference ligand (SDF),
  and a set of docking poses (SDF).
- Uses ODDT's InteractionFingerprint method to generate binary interaction
  fingerprints (per-residue, 8-bit representation of possible interaction types).
- Compares the docking poses to the reference ligand using the Tanimoto
  similarity coefficient.
- Prints similarity scores sorted by similarity (1.000 = identical, 0.000 = none).

Requirements:
- ODDT installed (and its dependencies, e.g. OpenBabel or RDKit).
- Input files correctly formatted: PDB for protein, SDF for ligands.

Written by Perplexity, 2025-08-13
Updated by Claude, 2026-02-23: Converted to argparse with tab completion
Updated by Claude, 2026-02-23: Added -o/--output, molecule name labels, sorted output
"""

import argparse
from rich_argparse import RawDescriptionRichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter

import oddt
from oddt.fingerprints import InteractionFingerprint, tanimoto


def get_mol_name(mol, idx):
    """Try to get molecule name from an oddt molecule object."""
    try:
        name = mol.Mol.GetProp('_Name')  # RDKit backend
        if name.strip():
            return name.strip()
    except Exception:
        pass
    try:
        name = mol.OBMol.GetTitle()      # OpenBabel backend
        if name.strip():
            return name.strip()
    except Exception:
        pass
    return f"pose_{idx + 1}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate protein-ligand interaction fingerprint (PLIF) similarities using ODDT.",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  %(prog)s -p protein.pdb -r reference.sdf --poses docked.sdf
  %(prog)s -p protein.pdb -r reference.sdf --poses docked.sdf -o similarities.tsv
        """
    )
    parser.add_argument(
        "-p", "--protein",
        required=True,
        help="Path to the protein PDB file"
    ).completer = FilesCompleter(allowednames=(".pdb",))
    parser.add_argument(
        "-r", "--reference",
        required=True,
        help="Path to the reference ligand SDF file"
    ).completer = FilesCompleter(allowednames=(".sdf",))
    parser.add_argument(
        "--poses",
        required=True,
        help="Path to the docking poses SDF file"
    ).completer = FilesCompleter(allowednames=(".sdf",))
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output TSV file path (default: print to stdout only)"
    ).completer = FilesCompleter(allowednames=(".tsv",))
    argcomplete.autocomplete(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    protein = next(oddt.toolkit.readfile('pdb', args.protein))
    protein.protein = True

    reference_ligand = next(oddt.toolkit.readfile('sdf', args.reference))
    docking_poses = list(oddt.toolkit.readfile('sdf', args.poses))

    ref_fp = InteractionFingerprint(reference_ligand, protein, strict=True)
    results = []

    for i, mol in enumerate(docking_poses):
        fp = InteractionFingerprint(mol, protein, strict=True)
        if fp is None or len(fp) == 0:
            print(f"Warning: pose {i + 1} fingerprint could not be generated, skipped.")
            continue
        name = get_mol_name(mol, i)
        similarity = tanimoto(ref_fp, fp)
        results.append((name, similarity))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Pose':<30} {'PLIF Similarity':>16}")
    print("-" * 48)
    for name, sim in results:
        print(f"{name:<30} {sim:>16.3f}")

    if args.output:
        with open(args.output, 'w') as f:
            f.write("Pose\tPLIF_Similarity\n")
            for name, sim in results:
                f.write(f"{name}\t{sim:.4f}\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
