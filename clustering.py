# PYTHON_ARGCOMPLETE_OK
"""
This script performs clustering of molecules based on Morgan fingerprint similarity
using RDKit. It reads an input SDF file with molecular structures and properties,
clusters molecules by Tanimoto similarity seeded by a score column, and outputs
a tab-separated file with cluster assignments.

Written by Perplexity 4.0.0, 2025-08-20
Updated by Claude, 2026-02-23: Converted to argparse with tab completion
Updated by Claude, 2026-02-23: Added --radius, --nbits, --output-sdf, progress bar, summary stats
"""

import argparse
from rich_argparse import RawDescriptionRichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster molecules by Morgan fingerprint similarity (Tanimoto).",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i molecules.sdf --id-col Compound_ID --score-col docking_score -o clusters.tsv
  %(prog)s -i molecules.sdf --id-col Name --score-col CNNaffinity -t 0.4 -o results.tsv
  %(prog)s -i molecules.sdf --id-col Name --score-col Score -o clusters.tsv --output-sdf reps.sdf
        """
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input SDF file path"
    ).completer = FilesCompleter(allowednames=(".sdf",))
    parser.add_argument(
        "--id-col",
        required=True,
        metavar="COLUMN",
        help="Unique compound identifier property name in SDF"
    )
    parser.add_argument(
        "--score-col",
        required=True,
        metavar="COLUMN",
        help="Score property name used for cluster seeding (best score = cluster representative)"
    )
    parser.add_argument(
        "-t", "--tanimoto-cutoff",
        type=float,
        default=0.6,
        metavar="CUTOFF",
        help="Tanimoto similarity cutoff for clustering (default: 0.6)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output TSV file path"
    ).completer = FilesCompleter(allowednames=(".tsv",))
    parser.add_argument(
        "--output-sdf",
        default=None,
        help="Output SDF file containing only cluster representatives"
    ).completer = FilesCompleter(allowednames=(".sdf",))
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default: 2, i.e. ECFP4)"
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit length (default: 2048)"
    )
    argcomplete.autocomplete(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    suppl = Chem.SDMolSupplier(args.input)
    molecules = [mol for mol in suppl if mol is not None]
    print(f"Loaded {len(molecules)} molecules from {args.input}")

    records = []
    for mol in molecules:
        props = mol.GetPropsAsDict()
        record = {args.id_col: props.get(args.id_col)}
        record.update(props)
        record["_mol"] = mol
        records.append(record)

    df = pd.DataFrame(records)
    df = df.reset_index(drop=True)

    if args.score_col not in df.columns:
        raise ValueError(f"Score column '{args.score_col}' not found in SDF properties. Available columns: {list(df.columns)}")

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=args.radius, nBits=args.nbits)
        if mol is not None else None
        for mol in df["_mol"]
    ]

    assigned = [False] * len(df)
    cluster_labels = [None] * len(df)
    cluster_id = 0
    cluster_info = []

    with tqdm(total=len(df), desc="Clustering", unit="mol") as pbar:
        prev_assigned = 0
        while not all(assigned[i] or fps[i] is None for i in range(len(df))):
            unassigned_idxs = [i for i in range(len(df)) if not assigned[i] and fps[i] is not None]
            if not unassigned_idxs:
                break
            best_idx = max(unassigned_idxs, key=lambda i: df.loc[i, args.score_col])
            cluster_members = [best_idx]
            assigned[best_idx] = True
            for i in unassigned_idxs:
                if i == best_idx:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[best_idx], fps[i])
                if sim >= args.tanimoto_cutoff:
                    cluster_members.append(i)
                    assigned[i] = True
            for i in cluster_members:
                cluster_labels[i] = cluster_id
            cluster_info.append({
                "Cluster_ID": cluster_id,
                "Num_Members": len(cluster_members),
                "Member_Indices": cluster_members,
                "Best_Score": df.loc[best_idx, args.score_col],
                "Best_Ligand_ID": df.loc[best_idx, args.id_col]
            })
            cluster_id += 1
            newly_assigned = sum(assigned) - prev_assigned
            pbar.update(newly_assigned)
            prev_assigned = sum(assigned)

    df["Cluster_ID"] = cluster_labels
    df = df.sort_values(by=["Cluster_ID", args.score_col], ascending=[True, False]).reset_index(drop=True)
    cluster_sizes = {info["Cluster_ID"]: info["Num_Members"] for info in cluster_info}
    df["Cluster_Size"] = df["Cluster_ID"].map(cluster_sizes)
    best_structures = {info["Cluster_ID"]: info["Best_Ligand_ID"] for info in cluster_info}
    df["Cluster_Representative"] = df["Cluster_ID"].map(best_structures)

    columns_to_output = [args.id_col, args.score_col, "Cluster_ID", "Cluster_Size", "Cluster_Representative"]
    df_out = df[columns_to_output]
    df_out.to_csv(args.output, sep="\t", index=False)
    print(f"Results saved to {args.output}")

    # Summary stats
    sizes = [info["Num_Members"] for info in cluster_info]
    n_singletons = sum(1 for s in sizes if s == 1)
    print(f"\nClustering summary (cutoff={args.tanimoto_cutoff}):")
    print(f"  Clusters:         {len(cluster_info)}")
    print(f"  Singletons:       {n_singletons} ({100 * n_singletons / len(cluster_info):.1f}%)")
    print(f"  Largest cluster:  {max(sizes)} members")
    print(f"  Avg cluster size: {len(molecules) / len(cluster_info):.1f}")

    # Write cluster representatives SDF
    if args.output_sdf:
        rep_ids = {info["Best_Ligand_ID"] for info in cluster_info}
        writer = Chem.SDWriter(args.output_sdf)
        for mol in molecules:
            if mol.HasProp(args.id_col) and mol.GetProp(args.id_col) in rep_ids:
                writer.write(mol)
        writer.close()
        print(f"Cluster representatives written to {args.output_sdf}")


if __name__ == "__main__":
    main()
