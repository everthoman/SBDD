"""
diverse_ligands.py

Select N diverse ligands from an SDF or SMILES file using RDKit MaxMinPicker.
Optionally bias selection away from a second reference file of molecules.

How the bias works:
    The reference molecules are added to the combined fingerprint pool and
    pre-seeded as 'firstPicks'. MaxMinPicker then maximises minimum Tanimoto
    distance to *all* already-picked molecules (including the reference set),
    so primary picks will be pushed away from the reference chemical space.

Usage:
    python diverse_ligands.py -i compounds.sdf -n 10 -o selected.sdf
    python diverse_ligands.py -i compounds.sdf -n 10 -o selected.sdf \
        --reference existing_actives.sdf
    python diverse_ligands.py -i compounds.smi -n 10 -o selected.sdf \
        --reference known.smi --seed 99

Dependencies:
    pip install rdkit
"""

import argparse
import os
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


# ---------------------------------------------------------------------------
# Molecule loading
# ---------------------------------------------------------------------------

def load_molecules(filepath):
    """Load molecules from an SDF or SMILES file.

    Returns a list of (mol, name) tuples. Invalid entries are skipped.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".sdf", ".mol"):
        return _load_sdf(filepath)
    elif ext in (".smi", ".smiles", ".csv", ".txt"):
        return _load_smiles(filepath)
    else:
        raise ValueError(
            f"Unsupported extension '{ext}'. "
            "Expected .sdf, .mol, .smi, .smiles, .csv, or .txt"
        )


def _load_sdf(filepath):
    supplier = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=True)
    mols = []
    for i, mol in enumerate(supplier):
        if mol is None:
            print(f"  WARNING: could not parse molecule at index {i} — skipping",
                  file=sys.stderr)
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else f"mol_{i}"
        if not name:
            name = f"mol_{i}"
        mols.append((mol, name))
    return mols


def _load_smiles(filepath):
    """Parse a whitespace- or comma-delimited SMILES file.

    Accepted line formats (auto-detected):
        SMILES [name]
        SMILES,name
        name,SMILES
    Lines starting with '#' are treated as comments.
    """
    mols = []
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
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  WARNING: invalid SMILES on line {i + 1}: '{smiles}' — skipping",
                      file=sys.stderr)
                continue
            mols.append((mol, name))
    return mols


def _looks_like_smiles(s):
    smiles_chars = set("CNOPSFClBrIcnops()[]=#@+-/\\0123456789.")
    return sum(c in smiles_chars for c in s) / max(len(s), 1) > 0.6


# ---------------------------------------------------------------------------
# Fingerprints & diversity picking
# ---------------------------------------------------------------------------

def compute_fingerprints(mols, radius=2, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [gen.GetFingerprint(mol) for mol, _ in mols]


def pick_diverse(primary_mols, primary_fps, num_picks, seed=42,
                 reference_fps=None):
    """Use MaxMinPicker to select num_picks diverse molecules from primary_mols.

    Parameters
    ----------
    primary_mols  : list of (mol, name)
    primary_fps   : Morgan fingerprints for primary_mols (same order)
    num_picks     : how many diverse molecules to return
    seed          : random seed for MaxMinPicker
    reference_fps : optional list of fps used to bias selection away from a
                    reference chemical space. These are pre-seeded as
                    firstPicks; the picker then maximises distance to them.

    Returns
    -------
    selected : list of (mol, name) chosen from primary_mols
    indices  : their original indices in primary_mols
    """
    n_primary = len(primary_mols)

    if num_picks >= n_primary:
        print(f"  NOTE: requested {num_picks} picks but only {n_primary} "
              "valid primary molecules — returning all.", file=sys.stderr)
        return primary_mols, list(range(n_primary))

    if reference_fps:
        # Pool = [reference ... | primary ...]
        # Reference indices: 0 .. n_ref-1  (pre-seeded as firstPicks)
        # Primary indices  : n_ref .. n_ref+n_primary-1
        n_ref = len(reference_fps)
        all_fps = reference_fps + primary_fps
        n_total = len(all_fps)
        first_picks = list(range(n_ref))          # lock in the reference set
        n_to_pick = n_ref + num_picks             # total picks incl. reference
    else:
        all_fps = primary_fps
        n_total = n_primary
        first_picks = []
        n_to_pick = num_picks
        n_ref = 0

    def tanimoto_dist(i, j):
        return 1.0 - DataStructs.TanimotoSimilarity(all_fps[i], all_fps[j])

    picker = MaxMinPicker()
    all_picks = list(
        picker.LazyPick(tanimoto_dist, n_total, n_to_pick,
                        firstPicks=first_picks, seed=seed)
    )

    # Keep only indices that fall in the primary partition
    primary_pick_indices = [idx - n_ref for idx in all_picks if idx >= n_ref]

    selected = [(primary_mols[i][0], primary_mols[i][1]) for i in primary_pick_indices]
    return selected, primary_pick_indices


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_sdf(selected_mols, output_path):
    writer = Chem.SDWriter(output_path)
    for mol, name in selected_mols:
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    print(f"Wrote {len(selected_mols)} molecules to '{output_path}'")


def write_smiles(selected_mols, output_path):
    with open(output_path, "w") as fh:
        for mol, name in selected_mols:
            smi = Chem.MolToSmiles(mol)
            fh.write(f"{smi} {name}\n")
    print(f"Wrote {len(selected_mols)} molecules to '{output_path}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select diverse ligands from an SDF or SMILES file using RDKit "
            "MaxMinPicker. Optionally bias selection away from a reference set."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Primary input file (SDF or SMILES) to pick from."
    )
    parser.add_argument(
        "-n", "--num-picks", type=int, default=10,
        help="Number of diverse molecules to select (default: 10)."
    )
    parser.add_argument(
        "-o", "--output", default="selected.sdf",
        help="Output file. Format determined by extension: .sdf or .smi (default: selected.sdf)."
    )
    parser.add_argument(
        "-r", "--reference",
        help=(
            "Optional reference file (SDF or SMILES). Selected molecules will "
            "be biased away from the chemical space of these molecules."
        )
    )
    parser.add_argument(
        "--radius", type=int, default=2,
        help="Morgan fingerprint radius (default: 2)."
    )
    parser.add_argument(
        "--nbits", type=int, default=2048,
        help="Morgan fingerprint bit length (default: 2048)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for MaxMinPicker (default: 42)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Input      : {args.input}")
    if args.reference:
        print(f"Reference  : {args.reference}  (bias-away mode)")
    print(f"Picks      : {args.num_picks}")
    print(f"FP         : radius={args.radius}, bits={args.nbits}")
    print(f"Seed       : {args.seed}")
    print()

    # Load primary set
    print("Loading primary molecules...")
    primary_mols = load_molecules(args.input)
    print(f"  {len(primary_mols)} valid molecules loaded.\n")

    if not primary_mols:
        print("ERROR: No valid molecules found in primary file.", file=sys.stderr)
        sys.exit(1)

    # Load reference set (optional)
    reference_fps = None
    if args.reference:
        print("Loading reference molecules...")
        ref_mols = load_molecules(args.reference)
        print(f"  {len(ref_mols)} valid reference molecules loaded.\n")
        if ref_mols:
            reference_fps = compute_fingerprints(
                ref_mols, radius=args.radius, n_bits=args.nbits
            )
        else:
            print("  WARNING: Reference file contained no valid molecules — ignoring.",
                  file=sys.stderr)

    # Fingerprint primary set
    print("Computing Morgan fingerprints for primary set...")
    primary_fps = compute_fingerprints(
        primary_mols, radius=args.radius, n_bits=args.nbits
    )

    # Diverse pick
    bias_note = " (biased away from reference)" if reference_fps else ""
    print(f"Running MaxMinPicker (n={args.num_picks}){bias_note}...")
    selected, indices = pick_diverse(
        primary_mols, primary_fps, args.num_picks,
        seed=args.seed, reference_fps=reference_fps
    )

    print(f"\nSelected molecules:")
    for rank, (i, (_, name)) in enumerate(zip(indices, selected), 1):
        print(f"  {rank:>3}. [idx {i:>5}]  {name}")

    ext = os.path.splitext(args.output)[1].lower()
    print()
    if ext in (".smi", ".smiles"):
        write_smiles(selected, args.output)
    elif ext in (".sdf", ".mol"):
        write_sdf(selected, args.output)
    else:
        print(f"ERROR: unrecognised output extension '{ext}'. Use .sdf or .smi",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
