#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import os
import subprocess
import shutil
import argparse
from rich_argparse import RawDescriptionRichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter
import pandas as pd


def check_fpocket():
    if shutil.which('fpocket') is None:
        raise RuntimeError(
            "fpocket executable not found in PATH. "
            "Please install fpocket: https://github.com/Discngine/fpocket"
        )


def run_fpocket(pdb_path, out_dir, keep_existing=False):
    if os.path.exists(out_dir):
        if keep_existing:
            print(f"Reusing existing fpocket output in {out_dir}")
            return
        shutil.rmtree(out_dir)
    print(f"Running fpocket on {pdb_path} ...")
    result = subprocess.run(['fpocket', '-f', pdb_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"fpocket failed:\n{result.stderr}")
    print("fpocket run completed.")


def parse_pocket_info(base_name, out_dir):
    possible_paths = [
        os.path.join(out_dir, "pockets_info.txt"),
        os.path.join(out_dir, "pockets", "pockets_info.txt"),
        os.path.join(out_dir, f"{base_name}_info.txt"),
    ]
    pockets_list_file = next((p for p in possible_paths if os.path.isfile(p)), None)
    if pockets_list_file is None:
        raise FileNotFoundError(f"Could not find pocket info file in {possible_paths}")
    pocket_data = []
    with open(pockets_list_file, 'r') as f:
        pocket = {}
        for line in f:
            line = line.strip()
            if line.startswith("Pocket"):
                if pocket:
                    pocket_data.append(pocket)
                pocket = {"Name": line.strip(':')}
            elif ':' in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    value = float(value)
                except ValueError:
                    pass
                pocket[key] = value
        if pocket:
            pocket_data.append(pocket)
    return pd.DataFrame(pocket_data)


def sort_pockets(df):
    for col in ['Drug Score', 'Druggability Score', 'Score']:
        if col in df.columns:
            return df.sort_values(col, ascending=False).reset_index(drop=True)
    return df


def write_viz_pml(out_dir, pockets_df):
    lines = [
        "# Protein: cartoon + rainbow carbons + element colours for heteroatoms",
        "show cartoon, polymer",
        "util.cbaw polymer",
        "spectrum resi, rainbow, (polymer and elem C)",
        "",
        "# Residues within 5 Å of any pocket sphere (resn STP = fpocket alpha spheres)",
        "select pocket_residues, byres (polymer within 5.0 of resn STP)",
        "show lines, pocket_residues",
        'label (pocket_residues and name CA), "%s%s" % (resn, resi)',
        "",
    ]
    pocket_names = ' '.join(
        f"pocket{int(''.join(filter(str.isdigit, str(row['Name']))))}"
        for _, row in pockets_df.iterrows()
    )
    lines += [f"order {pocket_names}, sort=0", "", "deselect"]

    path = os.path.join(out_dir, "viz.pml")
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return "viz.pml"


def run_pymol_script(out_dir, base_name, pockets_df):
    pml_file = f"{base_name}.pml"
    pml_path = os.path.join(out_dir, pml_file)
    if os.path.isfile(pml_path):
        abs_out_dir = os.path.abspath(out_dir)
        viz_file = write_viz_pml(out_dir, pockets_df)
        print(f"Launching PyMOL: {pml_path} ...")
        env = os.environ.copy()
        env.update({
            'LIBGL_ALWAYS_INDIRECT': '0',
            'GALLIUM_DRIVER': 'd3d12',
            'MESA_D3D12_DEFAULT_ADAPTER_NAME': 'NVIDIA',
        })
        subprocess.run(
            ['pymol', pml_file, viz_file],
            check=True, cwd=abs_out_dir, env=env,
        )
    else:
        print(f"PyMOL script not found: {pml_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fpocket on a PDB file and visualize pockets in PyMOL.",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i protein.pdb
  %(prog)s -i protein.pdb --no-pymol
  %(prog)s -i protein.pdb -o results/ --top-n 5
  %(prog)s -i protein.pdb --keep-existing --top-n 3
        """
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the PDB file"
    ).completer = FilesCompleter(allowednames=(".pdb",))
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        dest="output_dir",
        help="Output directory (default: <pdb_basename>_out)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Show only the top N pockets by druggability score"
    )
    parser.add_argument(
        "--no-pymol",
        action="store_true",
        help="Skip PyMOL visualization"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Reuse existing fpocket output directory instead of re-running fpocket"
    )
    argcomplete.autocomplete(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    pdb_path = args.input

    if not os.path.isfile(pdb_path):
        print(f"File not found: {pdb_path}")
        return

    check_fpocket()

    base_name = os.path.splitext(os.path.basename(pdb_path))[0]
    out_dir = args.output_dir if args.output_dir else f"{base_name}_out"

    run_fpocket(pdb_path, out_dir, keep_existing=args.keep_existing)
    pockets_df = parse_pocket_info(base_name, out_dir)

    print(f"Found {len(pockets_df)} pockets.")

    pockets_df = sort_pockets(pockets_df)

    if args.top_n is not None:
        pockets_df = pockets_df.head(args.top_n)
        print(f"Showing top {args.top_n} pockets by druggability score.")

    csv_out = os.path.join(out_dir, "pockets.csv")
    pockets_df.to_csv(csv_out, index=False)
    print(f"Pocket data saved to {csv_out}")

    print("Pocket summary:")
    print(pockets_df)

    if not args.no_pymol:
        run_pymol_script(out_dir, base_name, pockets_df)


if __name__ == "__main__":
    main()
