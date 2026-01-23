#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
================================================================================
GNINA Single-Point Docking Script
================================================================================

Description:
    This script performs molecular docking using GNINA, a deep learning-based
    molecular docking program. It processes a library of ligands against a
    protein receptor, utilizing GPU acceleration and multi-threading for
    efficient batch processing.

Features:
    - GPU-accelerated docking with support for multiple GPUs
    - Round-robin batch splitting for load balancing
    - Automatic binding site detection using a reference ligand
    - RDKit-based SDF merging and validation
    - Sorted output with score extraction to TSV format

Input Files:
    - Receptor:   Protein structure file in PDB format (.pdb)
    - Reference:  Reference ligand for autobox definition (.sdf)
    - Ligands:    Library of ligands to dock (.sdf)

Output Files:
    - <output>.sdf:         Docked poses sorted by Structure_ID
    - <output>_scores.tsv:  Tabulated scores and properties

Dependencies:
    - GNINA (https://github.com/gnina/gnina)
    - RDKit (conda install -c conda-forge rdkit)
    - tqdm (conda install -c conda-forge tqdm)

Usage:
    python gnina_sp.py -r receptor.pdb -ref reference.sdf -l ligands.sdf -o output
    python gnina_sp.py --help

Author:     [Your Name]
Version:    1.0
Date:       [Date]
================================================================================
"""

import os
import re
import glob
import gzip
import shutil
import queue
import threading
import subprocess
import io
import time
import argparse
from rich_argparse import RawDescriptionRichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # Disable RDKit warning/info messages globally


# ===== CONFIGURATION =====

GNINA_PATH = "/opt/gnina/gnina.1.3.2"  # Or just "gnina" if in PATH
OUTPUT_DIR = "gnina_outputs"
NUM_CPUS = 10
NUM_GPUS = 1
THREADS_PER_JOB = NUM_CPUS // NUM_GPUS
NUM_BATCHES_PER_GPU = 4


# ===== ARGUMENT PARSER =====

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GNINA Single-Point Docking Script - GPU-accelerated molecular docking",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
    Basic usage:
        python gnina_sp.py -r protein.pdb -ref ref_ligand.sdf -l ligands.sdf -o docked

    With custom GPU/CPU settings:
        python gnina_sp.py -r protein.pdb -ref ref_ligand.sdf -l ligands.sdf -o docked --num-gpus 2 --num-cpus 16

    Custom GNINA path:
        python gnina_sp.py -r protein.pdb -ref ref_ligand.sdf -l ligands.sdf -o docked --gnina /path/to/gnina
        """
    )

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument(
        '-r', '--receptor',
        type=str,
        required=True,
        help='Protein receptor file (.pdb)'
    ).completer = FilesCompleter(allowednames=(".pdb",))
    required.add_argument(
        '-ref', '--reference',
        type=str,
        required=True,
        help='Reference ligand for autobox definition (.sdf)'
    ).completer = FilesCompleter(allowednames=(".sdf",))
    required.add_argument(
        '-l', '--ligands',
        type=str,
        required=True,
        help='Ligand library file to dock (.sdf)'
    ).completer = FilesCompleter(allowednames=(".sdf",))
    required.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Base name for output files (no extension)'
    )

    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument(
        '--gnina',
        type=str,
        default=GNINA_PATH,
        help=f'Path to GNINA executable (default: {GNINA_PATH})'
    )
    optional.add_argument(
        '--num-gpus',
        type=int,
        default=NUM_GPUS,
        help=f'Number of GPUs to use (default: {NUM_GPUS})'
    )
    optional.add_argument(
        '--num-cpus',
        type=int,
        default=NUM_CPUS,
        help=f'Number of CPUs to use (default: {NUM_CPUS})'
    )
    optional.add_argument(
        '--batches-per-gpu',
        type=int,
        default=NUM_BATCHES_PER_GPU,
        help=f'Number of batches per GPU (default: {NUM_BATCHES_PER_GPU})'
    )
    optional.add_argument(
        '--exhaustiveness',
        type=int,
        default=8,
        help='GNINA exhaustiveness parameter (default: 8)'
    )
    optional.add_argument(
        '--num-modes',
        type=int,
        default=1,
        help='Number of binding modes to generate (default: 1)'
    )
    optional.add_argument(
        '--autobox-add',
        type=float,
        default=4.0,
        help='Padding around autobox in Angstroms (default: 4.0)'
    )
    optional.add_argument(
        '--seed',
        type=int,
        default=666,
        help='Random seed for reproducibility (default: 666)'
    )
    optional.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary batch files after completion'
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Validate input files
    if not os.path.isfile(args.receptor):
        parser.error(f"Receptor file not found: {args.receptor}")
    if not args.receptor.lower().endswith('.pdb'):
        parser.error(f"Receptor file must be a .pdb file: {args.receptor}")

    if not os.path.isfile(args.reference):
        parser.error(f"Reference ligand file not found: {args.reference}")
    if not args.reference.lower().endswith('.sdf'):
        parser.error(f"Reference ligand file must be a .sdf file: {args.reference}")

    if not os.path.isfile(args.ligands):
        parser.error(f"Ligands file not found: {args.ligands}")
    if not args.ligands.lower().endswith('.sdf'):
        parser.error(f"Ligands file must be a .sdf file: {args.ligands}")

    # Validate GNINA path
    if not (os.path.isfile(args.gnina) or shutil.which(args.gnina)):
        parser.error(f"GNINA executable not found: {args.gnina}")

    # Calculate threads per job
    args.threads_per_job = args.num_cpus // args.num_gpus

    return args


# ===== ROUND-ROBIN SPLIT =====

def split_ligands(input_file, total_batches):
    """Split SDF into total_batches in round-robin fashion."""
    os.makedirs("batches", exist_ok=True)
    batch_files = [f"batches/ligands_batch_{i}.sdf" for i in range(total_batches)]
    writers = [open(fname, "w") for fname in batch_files]
    with open(input_file, "r") as f:
        idx = 0
        buf = []
        for line in f:
            buf.append(line)
            if line.strip() == "$$$$":
                writers[idx].writelines(buf)
                buf = []
                idx = (idx + 1) % total_batches
    for w in writers:
        w.close()
    return batch_files


# ===== RUN GNINA =====

def run_gnina(ligand_batch, gpu_id, args):
    """Run GNINA docking for a single batch."""
    batch_name = os.path.splitext(os.path.basename(ligand_batch))[0]
    out_subdir = os.path.join(OUTPUT_DIR, batch_name)
    os.makedirs(out_subdir, exist_ok=True)
    out_sdf_gz = os.path.join(out_subdir, "docked.sdf.gz")
    log_file = os.path.join(out_subdir, "gnina.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = str(args.threads_per_job)

    cmd = [
        args.gnina,
        "-r", args.receptor,
        "-l", ligand_batch,
        "--autobox_ligand", args.reference,
        "--autobox_add", str(args.autobox_add),
        "--exhaustiveness", str(args.exhaustiveness),
        "--num_modes", str(args.num_modes),
        "--seed", str(args.seed),
        "-o", out_sdf_gz,
        "--log", log_file
    ]

    with subprocess.Popen(cmd, env=env,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL) as proc:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


# ===== GPU WORKER =====

def gpu_worker(gpu_id, job_queue, args, progress_bar):
    """Worker thread for processing docking jobs on a specific GPU."""
    while True:
        try:
            lig_batch = job_queue.get_nowait()
        except queue.Empty:
            break
        try:
            run_gnina(lig_batch, gpu_id, args)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR][GPU {gpu_id}] Job failed: {e}")
        finally:
            job_queue.task_done()
            progress_bar.update(1)


# ===== MERGE USING RDKIT =====

def merge_sdf_rdkit_safe(temp_sdf):
    """Parse and merge all docked.sdf.gz files safely using RDKit."""
    sdf_files = sorted(glob.glob(f"{OUTPUT_DIR}/*/docked.sdf.gz"))
    writer = Chem.SDWriter(temp_sdf)
    bad_count = 0
    total_count = 0
    for sdf_gz in sdf_files:
        with gzip.open(sdf_gz, 'rb') as f:
            bio = io.BytesIO(f.read())
            suppl = Chem.ForwardSDMolSupplier(bio, sanitize=False, removeHs=False)
            for mol in suppl:
                if mol is not None:
                    writer.write(mol)
                    total_count += 1
                else:
                    bad_count += 1
    writer.close()
    print(f"[MERGE] Unsorted merged SDF saved to {temp_sdf}")
    print(f"[MERGE] Molecules merged: {total_count}, skipped: {bad_count}")


# ===== SORT & EXPORT TSV =====

def sort_sdf_and_export_scores(input_sdf, output_sdf, tsv_file):
    """Sort SDF by Structure_ID and export scores to TSV."""
    suppl = Chem.SDMolSupplier(input_sdf, sanitize=False, removeHs=False)
    mols = [m for m in suppl if m is not None]

    all_props = set()
    for m in mols:
        all_props.update(list(m.GetPropNames()))
    all_props.discard("Structure_ID")
    prop_list = ["Structure_ID"] + sorted(all_props)

    def sort_key(m):
        if m.HasProp("Structure_ID"):
            match = re.search(r'TH(\d+)', m.GetProp("Structure_ID"))
            if match:
                return int(match.group(1))
        return float('inf')

    mols.sort(key=sort_key)

    writer = Chem.SDWriter(output_sdf)
    for m in mols:
        writer.write(m)
    writer.close()
    print(f"[SORT] Final sorted SDF saved to {output_sdf}")

    with open(tsv_file, "w") as out:
        out.write("\t".join(prop_list) + "\n")
        for m in mols:
            row = [(m.GetProp(p) if m.HasProp(p) else "") for p in prop_list]
            out.write("\t".join(row) + "\n")
    print(f"[SCORES] TSV score table saved to {tsv_file}")


# ===== COUNT MOLECULES IN SDF =====

def count_molecules_in_sdf(sdf_file):
    """Count the number of valid molecules in an SDF file."""
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
    return sum(1 for m in suppl if m is not None)


# ===== FORMATTED ELAPSED TIME =====

def format_elapsed_time(seconds):
    """Format elapsed time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


# ===== CLEANUP =====

def cleanup(keep_temp=False):
    """Remove temporary files and directories."""
    if keep_temp:
        print("[CLEANUP] Keeping temporary files as requested.")
        return

    if os.path.isdir("batches"):
        shutil.rmtree("batches")
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    print("[CLEANUP] Temporary files removed.")


# ===== MAIN =====

def main():
    """Main entry point for the docking script."""
    args = parse_arguments()

    # Print configuration summary
    print("=" * 60)
    print("GNINA Single-Point Docking")
    print("=" * 60)
    print(f"Receptor:        {args.receptor}")
    print(f"Reference:       {args.reference}")
    print(f"Ligands:         {args.ligands}")
    print(f"Output:          {args.output}.sdf")
    print(f"GPUs:            {args.num_gpus}")
    print(f"CPUs:            {args.num_cpus}")
    print(f"Threads/job:     {args.threads_per_job}")
    print(f"Batches/GPU:     {args.batches_per_gpu}")
    print(f"Exhaustiveness:  {args.exhaustiveness}")
    print(f"Num modes:       {args.num_modes}")
    print(f"Seed:            {args.seed}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_batches = args.num_gpus * args.batches_per_gpu
    batches = split_ligands(args.ligands, total_batches)

    job_queue = queue.Queue()
    for bf in batches:
        job_queue.put(bf)

    start_time = time.time()

    with tqdm(total=total_batches, desc="Docking progress", unit="batch") as pbar:
        threads = []
        for gpu_id in range(args.num_gpus):
            t = threading.Thread(
                target=gpu_worker,
                args=(gpu_id, job_queue, args, pbar)
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    temp_unsorted = f"{args.output}_unsorted.sdf"
    merge_sdf_rdkit_safe(temp_unsorted)

    sort_sdf_and_export_scores(
        temp_unsorted,
        f"{args.output}.sdf",
        f"{args.output}_scores.tsv"
    )
    os.remove(temp_unsorted)

    cleanup(keep_temp=args.keep_temp)

    end_time = time.time()
    elapsed = end_time - start_time

    total_ligands = count_molecules_in_sdf(f"{args.output}.sdf")
    secs_per_ligand = elapsed / total_ligands if total_ligands > 0 else float('inf')

    print("=" * 60)
    print("[STATS] Docking Statistics")
    print("=" * 60)
    print(f"Total ligands docked:    {total_ligands}")
    print(f"Total docking time:      {format_elapsed_time(elapsed)}")
    print(f"Average time per ligand: {secs_per_ligand:.2f} seconds")
    print("=" * 60)
    print("[ALL DONE] Docking completed successfully.")


if __name__ == "__main__":
    main()