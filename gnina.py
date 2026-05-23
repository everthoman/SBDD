#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
GNINA Docking Script — HTVS / SP / XP

Combined script for three docking precision modes:

  htvs   High-throughput virtual screening. CPU-only, no CNN scoring.
         Spawns one GNINA process per CPU core for maximum throughput.

  sp     Standard-precision GPU docking with CNN rescoring.
         CPUs are split evenly across GPUs for pose generation.

  xp     Extra-precision GPU docking with full CNN refinement.
         Higher exhaustiveness recommended (e.g. --exhaustiveness 16).

Resource allocation (automatic):
  HTVS: --cpus  parallel GNINA processes, each pinned to 1 CPU thread.
  SP/XP: --cpus CPU threads split evenly across --num-gpus GPUs.
         Each GPU processes --batches-per-gpu batches sequentially,
         so total batches = num_gpus × batches_per_gpu.

Requirements:
  GNINA  https://github.com/gnina/gnina
  Python rdkit  tqdm  rich-argparse

Usage:
  gnina.py htvs -r receptor.pdb -a ref.sdf -l library.sdf -o results
  gnina.py sp   -r receptor.pdb -a ref.sdf -l hits.sdf    -o results --num-gpus 2
  gnina.py xp   -r receptor.pdb -a ref.sdf -l hits.sdf    -o results --num-gpus 2 --exhaustiveness 16

Author: Evert J. Homan, PhD
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

RDLogger.DisableLog('rdApp.*')

_AUTO_CPUS = os.cpu_count() or 1
_DEFAULT_GNINA = "/opt/gnina/gnina"


# ─────────────────────────── SHARED UTILITIES ────────────────────────────────

def count_molecules_in_sdf(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
    return sum(1 for m in suppl if m is not None)


def split_ligands(input_file, total_batches, batch_dir):
    """Round-robin split of an SDF file into total_batches files."""
    os.makedirs(batch_dir, exist_ok=True)
    batch_files = [os.path.join(batch_dir, f"batch_{i}.sdf")
                   for i in range(total_batches)]
    writers = [open(f, "w") for f in batch_files]
    with open(input_file) as fh:
        idx = 0
        buf = []
        for line in fh:
            buf.append(line)
            if line.strip() == "$$$$":
                writers[idx].writelines(buf)
                buf = []
                idx = (idx + 1) % total_batches
    for w in writers:
        w.close()
    return batch_files


def merge_sdf(temp_sdf, dock_output_dir):
    """Merge all docked.sdf.gz outputs into one unsorted SDF."""
    sdf_files = sorted(glob.glob(os.path.join(dock_output_dir, "*/docked.sdf.gz")))
    writer = Chem.SDWriter(temp_sdf)
    total, bad = 0, 0
    for sdf_gz in sdf_files:
        with gzip.open(sdf_gz, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(
                io.BytesIO(f.read()), sanitize=False, removeHs=False
            )
            for mol in suppl:
                if mol is not None:
                    writer.write(mol)
                    total += 1
                else:
                    bad += 1
    writer.close()
    print(f"[MERGE] {total} poses merged, {bad} skipped → {temp_sdf}")


def sort_and_export(input_sdf, output_sdf, tsv_file, id_column):
    """
    Sort molecules by id_column and write a sorted SDF + TSV score table.

    Sort priority:
      1. Pure numeric IDs  → ascending numeric
      2. IDs with trailing digits (e.g. TH1234)  → ascending by trailing number,
         then alphabetically within ties
      3. Pure string IDs  → alphabetical
      4. Missing id_column  → last
    """
    suppl = Chem.SDMolSupplier(input_sdf, sanitize=False, removeHs=False)
    mols = [m for m in suppl if m is not None]

    def sort_key(m):
        if not m.HasProp(id_column):
            return (3, 0, "")
        val = m.GetProp(id_column).strip()
        try:
            return (0, float(val), val)
        except ValueError:
            match = re.search(r'(\d+)$', val)
            if match:
                return (1, int(match.group(1)), val)
            return (2, 0, val)

    mols.sort(key=sort_key)

    writer = Chem.SDWriter(output_sdf)
    for m in mols:
        writer.write(m)
    writer.close()
    print(f"[SORT]   Sorted SDF → {output_sdf}")

    all_props = set()
    for m in mols:
        all_props.update(m.GetPropNames())
    all_props.discard(id_column)
    prop_list = [id_column] + sorted(all_props)

    with open(tsv_file, "w") as out:
        out.write("\t".join(prop_list) + "\n")
        for m in mols:
            row = [m.GetProp(p) if m.HasProp(p) else "" for p in prop_list]
            out.write("\t".join(row) + "\n")
    print(f"[SCORES] TSV score table → {tsv_file}")


def cleanup(batch_dir, dock_output_dir, keep_temp):
    if keep_temp:
        print("[CLEANUP] Keeping temporary files.")
        return
    for d in (batch_dir, dock_output_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
    print("[CLEANUP] Temporary files removed.")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    return f"{seconds / 3600:.2f} h"


def validate_inputs(args, parser):
    for path, label in [
        (args.receptor,       "Receptor"),
        (args.autobox_ligand, "Autobox ligand"),
        (args.ligands,        "Ligands"),
    ]:
        if not os.path.isfile(path):
            parser.error(f"{label} file not found: {path}")
    if not (os.path.isfile(args.gnina) or shutil.which(args.gnina)):
        parser.error(f"GNINA executable not found: {args.gnina}")


def print_config(args, n_batches, threads_per_job):
    mode = args.mode.upper()
    print(f"\n{'═' * 56}")
    print(f"  GNINA {mode} Docking")
    print(f"{'═' * 56}")
    print(f"  Receptor:         {args.receptor}")
    print(f"  Autobox ligand:   {args.autobox_ligand}")
    print(f"  Ligands:          {args.ligands}")
    print(f"  Output:           {args.output}")
    print(f"  CNN scoring:      {args.cnn_scoring}")
    print(f"  Exhaustiveness:   {args.exhaustiveness}")
    print(f"  Binding modes:    {args.num_modes}")
    print(f"  Autobox add:      {args.autobox_add} Å")
    print(f"  Seed:             {args.seed}")
    print(f"  Batches:          {n_batches}")
    print(f"  Threads/job:      {threads_per_job}")
    if hasattr(args, 'num_gpus'):
        print(f"  GPUs:             {args.num_gpus}")
    print(f"  GNINA:            {args.gnina}")
    print(f"{'═' * 56}\n")


def finalize(args, dock_output_dir, start_time, n_input):
    temp_sdf = f"{args.output}_unsorted.sdf"
    merge_sdf(temp_sdf, dock_output_dir)
    sort_and_export(
        temp_sdf,
        f"{args.output}.sdf",
        f"{args.output}_scores.tsv",
        args.id_column,
    )
    os.remove(temp_sdf)

    elapsed = time.time() - start_time
    n_docked = count_molecules_in_sdf(f"{args.output}.sdf")
    secs_per = elapsed / n_docked if n_docked else float('inf')

    print(f"\n{'═' * 56}")
    print(f"  Mode:             {args.mode.upper()}")
    print(f"  Input ligands:    {n_input}")
    print(f"  Docked poses:     {n_docked}")
    print(f"  Total time:       {format_time(elapsed)}")
    print(f"  Avg / ligand:     {secs_per:.2f} s")
    print(f"  Output SDF:       {args.output}.sdf")
    print(f"  Scores TSV:       {args.output}_scores.tsv")
    print(f"{'═' * 56}")


# ─────────────────────────── HTVS (CPU-ONLY) ─────────────────────────────────

def run_gnina_cpu(ligand_batch, args, dock_output_dir):
    batch_name = os.path.splitext(os.path.basename(ligand_batch))[0]
    out_subdir = os.path.join(dock_output_dir, batch_name)
    os.makedirs(out_subdir, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"   # one thread per process → true CPU parallelism

    cmd = [
        args.gnina,
        "--no_gpu",
        "--cnn_scoring", args.cnn_scoring,
        "-r",                args.receptor,
        "-l",                ligand_batch,
        "--autobox_ligand",  args.autobox_ligand,
        "--autobox_add",     str(args.autobox_add),
        "--exhaustiveness",  str(args.exhaustiveness),
        "--num_modes",       str(args.num_modes),
        "--seed",            str(args.seed),
        "-o",                os.path.join(out_subdir, "docked.sdf.gz"),
        "--log",             os.path.join(out_subdir, "gnina.log"),
    ]

    with subprocess.Popen(cmd, env=env,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL) as proc:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def cpu_worker(job_queue, args, dock_output_dir, pbar):
    while True:
        try:
            batch = job_queue.get_nowait()
        except queue.Empty:
            break
        try:
            run_gnina_cpu(batch, args, dock_output_dir)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(batch)}: {e}")
        finally:
            job_queue.task_done()
            pbar.update(1)


def run_htvs(args, parser):
    validate_inputs(args, parser)

    batch_dir      = os.path.join(args.output_dir, "_batches")
    dock_output_dir = os.path.join(args.output_dir, "_docked")

    print(f"[INFO] Counting ligands…")
    n_ligands = count_molecules_in_sdf(args.ligands)

    # Never create more batches than ligands
    n_batches = min(args.cpus, n_ligands)
    if n_ligands < args.cpus:
        print(f"[INFO] Fewer ligands ({n_ligands}) than CPUs ({args.cpus}); "
              f"using {n_batches} batches.")

    print_config(args, n_batches=n_batches, threads_per_job=1)

    batches = split_ligands(args.ligands, n_batches, batch_dir)

    jq = queue.Queue()
    for b in batches:
        jq.put(b)

    start = time.time()
    with tqdm(total=n_batches, desc="Docking (HTVS)", unit="batch") as pbar:
        threads = [
            threading.Thread(target=cpu_worker,
                             args=(jq, args, dock_output_dir, pbar))
            for _ in range(n_batches)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    finalize(args, dock_output_dir, start, n_ligands)
    cleanup(batch_dir, dock_output_dir, args.keep_temp)


# ─────────────────────────── SP / XP (GPU) ───────────────────────────────────

def run_gnina_gpu(ligand_batch, gpu_id, threads_per_job, args, dock_output_dir):
    batch_name = os.path.splitext(os.path.basename(ligand_batch))[0]
    out_subdir = os.path.join(dock_output_dir, batch_name)
    os.makedirs(out_subdir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"]      = str(threads_per_job)

    cmd = [
        args.gnina,
        "--cnn_scoring", args.cnn_scoring,
        "-r",                args.receptor,
        "-l",                ligand_batch,
        "--autobox_ligand",  args.autobox_ligand,
        "--autobox_add",     str(args.autobox_add),
        "--exhaustiveness",  str(args.exhaustiveness),
        "--num_modes",       str(args.num_modes),
        "--seed",            str(args.seed),
        "-o",                os.path.join(out_subdir, "docked.sdf.gz"),
        "--log",             os.path.join(out_subdir, "gnina.log"),
    ]

    with subprocess.Popen(cmd, env=env,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL) as proc:
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def gpu_worker(gpu_id, threads_per_job, job_queue, args, dock_output_dir, pbar):
    while True:
        try:
            batch = job_queue.get_nowait()
        except queue.Empty:
            break
        try:
            run_gnina_gpu(batch, gpu_id, threads_per_job, args, dock_output_dir)
        except Exception as e:
            print(f"[ERROR][GPU {gpu_id}] {os.path.basename(batch)}: {e}")
        finally:
            job_queue.task_done()
            pbar.update(1)


def run_gpu_docking(args, parser):
    validate_inputs(args, parser)

    batch_dir       = os.path.join(args.output_dir, "_batches")
    dock_output_dir = os.path.join(args.output_dir, "_docked")

    # CPU threads split evenly among GPUs
    threads_per_job = max(1, args.cpus // args.num_gpus)

    print(f"[INFO] Counting ligands…")
    n_ligands   = count_molecules_in_sdf(args.ligands)
    max_batches = args.num_gpus * args.batches_per_gpu
    n_batches   = min(max_batches, n_ligands)

    if n_ligands < max_batches:
        print(f"[INFO] Fewer ligands ({n_ligands}) than max batches ({max_batches}); "
              f"using {n_batches} batches.")

    print_config(args, n_batches=n_batches, threads_per_job=threads_per_job)

    batches = split_ligands(args.ligands, n_batches, batch_dir)

    jq = queue.Queue()
    for b in batches:
        jq.put(b)

    start = time.time()
    with tqdm(total=n_batches, desc=f"Docking ({args.mode.upper()})", unit="batch") as pbar:
        # One worker thread per GPU; each drains the shared queue sequentially,
        # so no two jobs ever share the same GPU simultaneously.
        threads = [
            threading.Thread(target=gpu_worker,
                             args=(gpu_id, threads_per_job, jq, args, dock_output_dir, pbar))
            for gpu_id in range(args.num_gpus)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    finalize(args, dock_output_dir, start, n_ligands)
    cleanup(batch_dir, dock_output_dir, args.keep_temp)


# ─────────────────────────── ARGUMENT PARSING ────────────────────────────────

def _add_io_args(p):
    p.add_argument(
        "-r", "--receptor", required=True,
        help="Protein receptor file (.pdb)",
    ).completer = FilesCompleter(allowednames=(".pdb",))
    p.add_argument(
        "-a", "--autobox-ligand", required=True, dest="autobox_ligand",
        help="Reference ligand defining the binding-site autobox (.sdf)",
    ).completer = FilesCompleter(allowednames=(".sdf",))
    p.add_argument(
        "-l", "--ligands", required=True,
        help="Ligand library to dock (.sdf)",
    ).completer = FilesCompleter(allowednames=(".sdf",))
    p.add_argument(
        "-o", "--output", required=True,
        help="Base name for output files (no extension)",
    )


def _add_docking_args(p, cnn_default):
    p.add_argument(
        "--exhaustiveness", type=int, default=8,
        help="Search exhaustiveness (default: 8; increase for XP)",
    )
    p.add_argument(
        "--num-modes", type=int, default=1, dest="num_modes",
        help="Binding modes to generate per ligand (default: 1)",
    )
    p.add_argument(
        "--autobox-add", type=float, default=4.0, dest="autobox_add",
        help="Padding around the autobox in Å (default: 4.0)",
    )
    p.add_argument(
        "--seed", type=int, default=666,
        help="Random seed for reproducibility (default: 666)",
    )
    p.add_argument(
        "--cnn-scoring", default=cnn_default, dest="cnn_scoring",
        choices=["none", "rescore", "refinement", "all"],
        help=f"CNN scoring mode (default: {cnn_default})",
    )


def _add_common_args(p):
    p.add_argument(
        "--gnina", default=_DEFAULT_GNINA,
        help=f"Path to GNINA executable (default: {_DEFAULT_GNINA})",
    )
    p.add_argument(
        "--output-dir", default="gnina_tmp", dest="output_dir",
        help="Directory for intermediate batch files (default: gnina_tmp)",
    )
    p.add_argument(
        "--id-column", default="Structure_ID", dest="id_column",
        help="SDF property used to sort output (default: Structure_ID)",
    )
    p.add_argument(
        "--keep-temp", action="store_true", dest="keep_temp",
        help="Keep intermediate batch files after completion",
    )


def make_parser():
    parser = argparse.ArgumentParser(
        description="GNINA docking: htvs (CPU-only) · sp (GPU) · xp (GPU + CNN refinement)",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Resource allocation:
  htvs  One GNINA process per CPU core (OMP_NUM_THREADS=1 each).
        Set --cpus to override; defaults to all available cores.
  sp/xp One GNINA process per GPU. CPU threads are split evenly across GPUs
        (threads/job = --cpus ÷ --num-gpus). Each GPU sequentially works
        through --batches-per-gpu batches.

Examples:
  %(prog)s htvs -r receptor.pdb -a ref.sdf -l library.sdf -o htvs_out
  %(prog)s htvs -r receptor.pdb -a ref.sdf -l library.sdf -o htvs_out --cpus 32

  %(prog)s sp   -r receptor.pdb -a ref.sdf -l hits.sdf    -o sp_out
  %(prog)s sp   -r receptor.pdb -a ref.sdf -l hits.sdf    -o sp_out --num-gpus 2 --cpus 16

  %(prog)s xp   -r receptor.pdb -a ref.sdf -l hits.sdf    -o xp_out --exhaustiveness 16
  %(prog)s xp   -r receptor.pdb -a ref.sdf -l hits.sdf    -o xp_out --num-gpus 2 --cnn-scoring all
        """,
    )

    subs = parser.add_subparsers(dest="mode", metavar="MODE")
    subs.required = True

    # ── HTVS ─────────────────────────────────────────────────────────────────
    htvs = subs.add_parser(
        "htvs",
        help="High-throughput virtual screening (CPU-only, no CNN scoring)",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog=(
            "Spawns --cpus parallel GNINA processes (OMP_NUM_THREADS=1 each).\n"
            "Ideal for screening large libraries on CPU-only nodes."
        ),
    )
    _add_io_args(htvs)
    _add_docking_args(htvs, cnn_default="none")
    _add_common_args(htvs)
    htvs.add_argument(
        "--cpus", type=int, default=_AUTO_CPUS,
        help=f"Parallel CPU processes (default: {_AUTO_CPUS} — all available cores)",
    )

    # ── SP ────────────────────────────────────────────────────────────────────
    sp = subs.add_parser(
        "sp",
        help="Standard-precision GPU docking with CNN rescoring",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog=(
            "GPU-accelerated docking. CPU threads are divided evenly across GPUs.\n"
            "Each GPU works through its batches one at a time."
        ),
    )
    _add_io_args(sp)
    _add_docking_args(sp, cnn_default="rescore")
    _add_common_args(sp)
    sp.add_argument(
        "--num-gpus", type=int, default=1, dest="num_gpus",
        help="Number of GPUs to use (default: 1)",
    )
    sp.add_argument(
        "--cpus", type=int, default=_AUTO_CPUS,
        help=f"Total CPU threads, split across GPUs (default: {_AUTO_CPUS})",
    )
    sp.add_argument(
        "--batches-per-gpu", type=int, default=4, dest="batches_per_gpu",
        help="Batches per GPU processed sequentially (default: 4)",
    )

    # ── XP ────────────────────────────────────────────────────────────────────
    xp = subs.add_parser(
        "xp",
        help="Extra-precision GPU docking with CNN refinement",
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog=(
            "Full CNN refinement. Consider --exhaustiveness 16 and --num-modes 9\n"
            "for higher pose quality. CPU threads are divided evenly across GPUs."
        ),
    )
    _add_io_args(xp)
    _add_docking_args(xp, cnn_default="refinement")
    _add_common_args(xp)
    xp.add_argument(
        "--num-gpus", type=int, default=1, dest="num_gpus",
        help="Number of GPUs to use (default: 1)",
    )
    xp.add_argument(
        "--cpus", type=int, default=_AUTO_CPUS,
        help=f"Total CPU threads, split across GPUs (default: {_AUTO_CPUS})",
    )
    xp.add_argument(
        "--batches-per-gpu", type=int, default=4, dest="batches_per_gpu",
        help="Batches per GPU processed sequentially (default: 4)",
    )

    return parser


# ─────────────────────────── ENTRY POINT ─────────────────────────────────────

def main():
    parser = make_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == "htvs":
        run_htvs(args, parser)
    else:
        run_gpu_docking(args, parser)


if __name__ == "__main__":
    main()
