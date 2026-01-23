#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
GNINA High-Throughput Virtual Screening (HTVS) Script

This script performs CPU-based molecular docking using GNINA, a deep learning
framework for molecular docking. It is optimized for screening large ligand
libraries against a protein receptor.

Workflow:
  1. Splits the input ligand library into batches for parallel processing
  2. Runs GNINA docking on each batch using multiple CPU threads
  3. Merges all docked poses into a single SDF file
  4. Sorts results by Structure_ID and exports a TSV score table
  5. Cleans up temporary files

Requirements:
  - GNINA executable (https://github.com/gnina/gnina)
  - Python packages: rdkit, tqdm

Usage:
  python gnina_htvs.py -r receptor.pdb -a reference.sdf -l ligands.sdf -o output_name

Author: Evert J. Homan, PhD
Date: January 23, 2026
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
from tqdm import tqdm  # conda install -c conda-forge tqdm
from rdkit import Chem  # conda install -c conda-forge rdkit
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # suppress RDKit warnings

# ===== DEFAULT CONFIGURATION =====

DEFAULT_GNINA_PATH = "/opt/gnina/gnina.1.3.2"
DEFAULT_OUTPUT_DIR = "gnina_outputs"
DEFAULT_NUM_CPUS = 10

# ===== ROUND-ROBIN SPLIT =====

def split_ligands(input_file, total_batches):
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

# ===== RUN GNINA CPU ONLY =====

def run_gnina_cpu(ligand_batch, receptor, autobox_ligand, gnina_path, output_dir):
    batch_name = os.path.splitext(os.path.basename(ligand_batch))[0]
    out_subdir = os.path.join(output_dir, batch_name)
    os.makedirs(out_subdir, exist_ok=True)
    out_sdf_gz = os.path.join(out_subdir, "docked.sdf.gz")
    log_file = os.path.join(out_subdir, "gnina.log")

    cmd = [
        gnina_path,
        "--no_gpu",
        "--cnn_scoring", "none",
        "-r", receptor,
        "-l", ligand_batch,
        "--autobox_ligand", autobox_ligand,
        "--autobox_add", "4",
        "--exhaustiveness", "8",
        "--num_modes", "1",
        "--seed", "666",
        "-o", out_sdf_gz,
        "--log", log_file
    ]

    with subprocess.Popen(cmd,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL) as proc:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

# ===== CPU WORKER THREAD =====

def cpu_worker(job_queue, receptor, autobox_ligand, gnina_path, output_dir, progress_bar):
    while True:
        try:
            lig_batch = job_queue.get_nowait()
        except queue.Empty:
            break
        try:
            run_gnina_cpu(lig_batch, receptor, autobox_ligand, gnina_path, output_dir)
        except Exception as e:
            print(f"[ERROR][CPU Worker] Error running batch {lig_batch}: {e}")
        finally:
            job_queue.task_done()
            progress_bar.update(1)

# ===== MERGE SDF BATCHES SAFELY WITH RDKit =====

def merge_sdf_rdkit_safe(temp_sdf, output_dir):
    sdf_files = sorted(glob.glob(f"{output_dir}/*/docked.sdf.gz"))
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

# ===== SORT SDF AND EXPORT SCORES TSV =====

def sort_sdf_and_export_scores(input_sdf, output_sdf, tsv_file):
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
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
    return sum(1 for m in suppl if m is not None)

# ===== FORMATTED ELAPSED TIME =====

def format_elapsed_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

# ===== CLEANUP =====

def cleanup(output_dir):
    if os.path.isdir("batches"):
        shutil.rmtree("batches")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    print("[CLEANUP] Temporary files removed.")

# ===== ARGUMENT PARSING =====

def parse_args():
    parser = argparse.ArgumentParser(
        description="GNINA High-Throughput Virtual Screening (CPU-only)",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r receptor.pdb -a reference.sdf -l ligands.sdf -o results
  %(prog)s -r receptor.pdb -a reference.sdf -l ligands.sdf -o results --cpus 16
  %(prog)s -r receptor.pdb -a reference.sdf -l ligands.sdf -o results --gnina /path/to/gnina
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-r", "--receptor",
        required=True,
        help="Protein receptor file (.pdb)"
    ).completer = FilesCompleter(allowednames=(".pdb",))
    parser.add_argument(
        "-a", "--autobox-ligand",
        required=True,
        dest="autobox_ligand",
        help="Reference ligand file for autobox (.sdf)"
    ).completer = FilesCompleter(allowednames=(".sdf",))
    parser.add_argument(
        "-l", "--ligands",
        required=True,
        help="Ligand library file to dock (.sdf)"
    ).completer = FilesCompleter(allowednames=(".sdf",))
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Base name for output files (no extension)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--cpus",
        type=int,
        default=DEFAULT_NUM_CPUS,
        help=f"Number of CPU cores/batches to use (default: {DEFAULT_NUM_CPUS})"
    )
    parser.add_argument(
        "--gnina",
        default=DEFAULT_GNINA_PATH,
        help=f"Path to gnina executable (default: {DEFAULT_GNINA_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for intermediate outputs (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    argcomplete.autocomplete(parser)
    return parser.parse_args()

# ===== VALIDATE INPUTS =====

def validate_inputs(args):
    errors = []
    
    # Check receptor file
    if not os.path.isfile(args.receptor):
        errors.append(f"Receptor file not found: {args.receptor}")
    elif not args.receptor.lower().endswith(".pdb"):
        errors.append(f"Receptor file must be a .pdb file: {args.receptor}")
    
    # Check autobox ligand file
    if not os.path.isfile(args.autobox_ligand):
        errors.append(f"Autobox ligand file not found: {args.autobox_ligand}")
    elif not args.autobox_ligand.lower().endswith(".sdf"):
        errors.append(f"Autobox ligand file must be a .sdf file: {args.autobox_ligand}")
    
    # Check ligands file
    if not os.path.isfile(args.ligands):
        errors.append(f"Ligands file not found: {args.ligands}")
    elif not args.ligands.lower().endswith(".sdf"):
        errors.append(f"Ligands file must be a .sdf file: {args.ligands}")
    
    # Check gnina executable
    if not (os.path.isfile(args.gnina) or shutil.which(args.gnina)):
        errors.append(f"Gnina executable not found: {args.gnina}")
    
    # Check output name
    if not args.output.strip():
        errors.append("Output base name cannot be empty")
    
    # Check CPU count
    if args.cpus < 1:
        errors.append("Number of CPUs must be at least 1")
    
    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        exit(1)

# ===== MAIN FUNCTION =====

def main():
    args = parse_args()
    validate_inputs(args)
    
    gnina_path = args.gnina
    output_dir = args.output_dir
    num_batches = args.cpus
    receptor = args.receptor
    autobox_ligand = args.autobox_ligand
    ligands_file = args.ligands
    output_basename = args.output
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"[CONFIG] Receptor: {receptor}")
    print(f"[CONFIG] Autobox ligand: {autobox_ligand}")
    print(f"[CONFIG] Ligands file: {ligands_file}")
    print(f"[CONFIG] Output basename: {output_basename}")
    print(f"[CONFIG] CPUs/batches: {num_batches}")
    print(f"[CONFIG] Gnina path: {gnina_path}")

    batches = split_ligands(ligands_file, num_batches)

    job_queue = queue.Queue()
    for bf in batches:
        job_queue.put(bf)

    start_time = time.time()

    with tqdm(total=num_batches, desc="Docking progress", unit="batch") as pbar:
        threads = []
        for _ in range(num_batches):
            t = threading.Thread(
                target=cpu_worker,
                args=(job_queue, receptor, autobox_ligand, gnina_path, output_dir, pbar)
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    temp_unsorted_sdf = f"{output_basename}_unsorted.sdf"

    merge_sdf_rdkit_safe(temp_unsorted_sdf, output_dir)

    sort_sdf_and_export_scores(
        temp_unsorted_sdf,
        f"{output_basename}.sdf",
        f"{output_basename}_scores.tsv"
    )

    os.remove(temp_unsorted_sdf)

    cleanup(output_dir)

    end_time = time.time()
    elapsed = end_time - start_time

    total_ligands = count_molecules_in_sdf(f"{output_basename}.sdf")
    secs_per_ligand = elapsed / total_ligands if total_ligands > 0 else float('inf')

    print(f"[STATS] Total ligands docked: {total_ligands}")
    print(f"[STATS] Total docking time: {format_elapsed_time(elapsed)}")
    print(f"[STATS] Average time per ligand: {secs_per_ligand:.2f} seconds")

    print("[ALL DONE] Docking completed successfully.")

if __name__ == "__main__":
    main()