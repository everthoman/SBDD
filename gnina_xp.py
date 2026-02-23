#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

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
from tqdm import tqdm # conda install -c conda-forge tqdm
from rdkit import Chem # conda install -c conda-forge rdkit
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*') # Disable RDKit warning/info messages globally

# ===== CONFIGURATION =====

gnina_path = "/opt/gnina/gnina.1.3.2" # Or just "gnina" if in PATH

output_dir = "gnina_outputs"

num_cpus = 36

num_gpus = 2

threads_per_job = num_cpus // num_gpus

num_batches_per_gpu = 4

# ===== ARGUMENT PARSING =====

def parse_args():
    parser = argparse.ArgumentParser(
        description="GNINA Extra-Precision (XP) Docking Script - GPU-accelerated molecular docking",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r receptor.pdb -a reference.sdf -l ligands.sdf -o results
        """
    )
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
    argcomplete.autocomplete(parser)
    return parser.parse_args()

# ===== COUNT LIGANDS =====

def count_ligands_in_sdf(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    return sum(1 for mol in supplier if mol is not None)

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

def run_gnina(ligand_batch, gpu_id, receptor, autobox_ligand, refinement):
    batch_name = os.path.splitext(os.path.basename(ligand_batch))[0]
    out_subdir = os.path.join(output_dir, batch_name)
    os.makedirs(out_subdir, exist_ok=True)
    out_sdf_gz = os.path.join(out_subdir, "docked.sdf.gz")
    log_file = os.path.join(out_subdir, "gnina.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = str(threads_per_job)

    cmd = [
        gnina_path,
        "--cnn_scoring", refinement,
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

    with subprocess.Popen(cmd, env=env,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL) as proc:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

# ===== GPU WORKER =====

def gpu_worker(gpu_id, job_queue, receptor, autobox_ligand, progress_bar, refinement):
    while True:
        try:
            lig_batch = job_queue.get_nowait()
        except queue.Empty:
            break
        try:
            run_gnina(lig_batch, gpu_id, receptor, autobox_ligand, refinement)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR][GPU {gpu_id}] Job failed: {e}")
        finally:
            job_queue.task_done()
            progress_bar.update(1)

# ===== MERGE USING RDKIT =====

def merge_sdf_rdkit_safe(temp_sdf):
    """Parse and merge all docked.sdf.gz files safely using RDKit."""
    sdf_files = sorted(glob.glob(f"{output_dir}/*/docked.sdf.gz"))
    writer = Chem.SDWriter(temp_sdf)
    bad_count = 0
    total_count = 0
    for sdf_gz in sdf_files:
        with gzip.open(sdf_gz, 'rb') as f: # binary mode
            bio = io.BytesIO(f.read()) # in-memory buffer
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

def cleanup():
    if os.path.isdir("batches"):
        shutil.rmtree("batches")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    print("[CLEANUP] Temporary files removed.")

# ===== MAIN =====

def main():
    args = parse_args()
    os.makedirs(output_dir, exist_ok=True)

    receptor = args.receptor
    autobox_ligand = args.autobox_ligand
    ligands_file = args.ligands
    output_basename = args.output

    refinement = "refinement"  # Set your cnn_scoring mode here: none, rescore, refinement, or all

    ligand_count = count_ligands_in_sdf(ligands_file)
    max_batches = num_gpus * num_batches_per_gpu
    total_batches = min(max_batches, ligand_count) if ligand_count > 0 else 1

    if ligand_count < max_batches:
        print(f"[WARNING] Number of ligands ({ligand_count}) is less than max batches {max_batches}.")
        print(f"[INFO] Reducing total_batches to {total_batches} to avoid empty batches.")

    batches = split_ligands(ligands_file, total_batches) # round-robin split

    job_queue = queue.Queue()
    for bf in batches:
        job_queue.put(bf)

    start_time = time.time()

    with tqdm(total=total_batches, desc="Docking progress", unit="batch") as pbar:
        threads = []
        for gpu_id in range(num_gpus):
            t = threading.Thread(target=gpu_worker,
                                 args=(gpu_id, job_queue, receptor, autobox_ligand, pbar, refinement))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    temp_unsorted = f"{output_basename}_unsorted.sdf"
    merge_sdf_rdkit_safe(temp_unsorted)

    sort_sdf_and_export_scores(
        temp_unsorted,
        f"{output_basename}.sdf",
        f"{output_basename}_scores.tsv"
    )
    os.remove(temp_unsorted)

    cleanup()

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