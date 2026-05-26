# gnina.py — GNINA Docking (HTVS / SP / XP)

Three-mode wrapper around [GNINA](https://github.com/gnina/gnina) for molecular docking with
automatic resource allocation, batched parallelism, pose merging, and sorted SDF + TSV output.
Modeled loosely on Schrödinger Glide's HTVS / SP / XP tiers.

| Mode | Hardware | CNN scoring | Typical use |
|---|---|---|---|
| `htvs` | CPU only | none | High-throughput screening of large libraries |
| `sp`   | GPU | `rescore` | Standard-precision docking of hit lists |
| `xp`   | GPU | `refinement` | Extra-precision docking with full CNN refinement |

---

## Dependencies

| Package | Role | Required |
|---|---|---|
| GNINA | docking engine (default path: `/opt/gnina/gnina`) | yes |
| RDKit | SDF I/O, molecule counting, sorted output | yes |
| tqdm | progress bars | yes |
| rich-argparse | coloured help | yes |
| argcomplete | tab completion | optional |
| CUDA-capable GPU(s) | required for `sp` and `xp` | for GPU modes |

Install in a conda environment:

```bash
conda install -c conda-forge rdkit tqdm
pip install rich-argparse argcomplete
# GNINA: see https://github.com/gnina/gnina for binary or build instructions
```

---

## Pipeline

```
Input ligand SDF
  → count molecules (RDKit)
  → round-robin split into N batches → _batches/batch_*.sdf
  → parallel GNINA processes
      htvs:  one process per CPU core, OMP_NUM_THREADS=1
      sp/xp: one worker per GPU, each draining a shared queue;
             CPU threads divided evenly across GPUs
  → per-batch docked.sdf.gz + gnina.log → _docked/<batch>/
  → merge all docked.sdf.gz → unsorted SDF (RDKit)
  → sort by --id-column (numeric → trailing-digit → alphabetical → missing)
  → output: <output>.sdf  +  <output>_scores.tsv
  → cleanup intermediate files (unless --keep-temp)
```

### Resource allocation

| Mode | Workers | Batches | CPU threads/job |
|---|---|---|---|
| `htvs` | `--cpus` processes | `min(--cpus, n_ligands)` | 1 |
| `sp` / `xp` | `--num-gpus` workers | `min(--num-gpus × --batches-per-gpu, n_ligands)` | `--cpus ÷ --num-gpus` |

Each GPU worker drains the shared queue sequentially, so no two jobs share a GPU at once.

---

## Usage

```
gnina.py {htvs,sp,xp} -r RECEPTOR -a AUTOBOX_LIGAND -l LIGANDS -o OUTPUT
                      [--exhaustiveness N] [--num-modes N] [--autobox-add Å]
                      [--seed N] [--cnn-scoring MODE]
                      [--gnina PATH] [--output-dir DIR] [--id-column NAME] [--keep-temp]
                      [htvs:    --cpus N]
                      [sp/xp:   --num-gpus N --cpus N --batches-per-gpu N]
```

### Options

#### Input / output (all modes)

| Flag | Default | Description |
|---|---|---|
| `-r / --receptor FILE` | required | Protein receptor (`.pdb`) |
| `-a / --autobox-ligand FILE` | required | Reference ligand defining the binding-site autobox (`.sdf`) |
| `-l / --ligands FILE` | required | Ligand library to dock (`.sdf`) |
| `-o / --output NAME` | required | Base name for output files (no extension) |

#### Docking (all modes)

| Flag | Default | Description |
|---|---|---|
| `--exhaustiveness INT` | `8` | Search exhaustiveness; increase for XP (e.g. `16`) |
| `--num-modes INT` | `1` | Binding modes per ligand |
| `--autobox-add FLOAT` | `4.0` | Padding around autobox (Å) |
| `--seed INT` | `666` | Random seed for reproducibility |
| `--cnn-scoring MODE` | per-mode | `none` / `rescore` / `refinement` / `all` (defaults: htvs=`none`, sp=`rescore`, xp=`refinement`) |

#### Common

| Flag | Default | Description |
|---|---|---|
| `--gnina PATH` | `/opt/gnina/gnina` | Path to GNINA executable |
| `--output-dir DIR` | `gnina_tmp` | Directory for intermediate batch files |
| `--id-column NAME` | `Structure_ID` | SDF property used to sort output |
| `--keep-temp` | off | Keep `_batches/` and `_docked/` after completion |

#### HTVS

| Flag | Default | Description |
|---|---|---|
| `--cpus INT` | all cores | Parallel CPU processes |

#### SP / XP

| Flag | Default | Description |
|---|---|---|
| `--num-gpus INT` | `1` | Number of GPUs to use |
| `--cpus INT` | all cores | Total CPU threads, split evenly across GPUs |
| `--batches-per-gpu INT` | `4` | Batches per GPU processed sequentially |

---

## Output

| File | Description |
|---|---|
| `<output>.sdf` | All docked poses, sorted by `--id-column` |
| `<output>_scores.tsv` | All SDF properties as a tab-separated table (id column first) |

Sort priority for `--id-column`:
1. Pure numeric IDs → ascending numeric
2. IDs with trailing digits (e.g. `TH1234`) → ascending by trailing number, then alphabetical
3. Pure string IDs → alphabetical
4. Missing column → last

---

## Examples

### HTVS — large library on a CPU node
```bash
gnina.py htvs -r receptor.pdb -a ref.sdf -l library.sdf -o htvs_out
gnina.py htvs -r receptor.pdb -a ref.sdf -l library.sdf -o htvs_out --cpus 32
```

### SP — hit list on a single GPU
```bash
gnina.py sp -r receptor.pdb -a ref.sdf -l hits.sdf -o sp_out
```

### SP — two GPUs, 16 CPU threads total
```bash
gnina.py sp -r receptor.pdb -a ref.sdf -l hits.sdf -o sp_out --num-gpus 2 --cpus 16
```

### XP — higher exhaustiveness, multiple binding modes
```bash
gnina.py xp -r receptor.pdb -a ref.sdf -l hits.sdf -o xp_out \
            --exhaustiveness 16 --num-modes 9
```

### XP — two GPUs, full CNN scoring
```bash
gnina.py xp -r receptor.pdb -a ref.sdf -l hits.sdf -o xp_out \
            --num-gpus 2 --cnn-scoring all
```

---

## Notes

- **Autobox vs. explicit box**: the binding site is defined by `--autobox-ligand` plus
  `--autobox-add` Å of padding. Pose the reference ligand in the desired pocket beforehand
  (e.g. from co-crystallised ligand or alignment).

- **HTVS uses `--no_gpu` and CNN disabled by default** for maximum throughput; this is the only
  combination that scales linearly to all CPU cores. Adding CNN scoring to HTVS will dramatically
  slow it down — use SP instead.

- **Round-robin batching** distributes molecules evenly across batches by `$$$$` delimiter, so
  fast and slow ligands are mixed in every batch. This gives good load balancing without
  pre-sorting by complexity.

- **GPU sharing is avoided**: SP/XP launch exactly one GNINA worker per GPU. Each worker drains
  the shared batch queue sequentially. `--batches-per-gpu` controls how many batches a GPU
  processes one after the other (default 4 → 8 batches total for 2 GPUs).

- **`OMP_NUM_THREADS` is set per process**: HTVS uses 1 (one thread per process, true parallelism);
  SP/XP use `--cpus ÷ --num-gpus` (each GPU job gets its share of CPU threads for the Vina pose
  generation step).

- **Intermediate files** live under `--output-dir` (default `gnina_tmp`): `_batches/` for split
  ligands and `_docked/<batch>/` for per-batch outputs. Removed at the end unless `--keep-temp`.

- **Tab completion**: run `eval "$(register-python-argcomplete gnina.py)"` once to enable
  `gnina.py <TAB>` completion for files and arguments.

---

## Reference

McNutt AT, Francoeur P, Aggarwal R, Masuda T, Meli R, Ragoza M, Sunseri J, Koes DR.
*GNINA 1.0: molecular docking with deep learning.*
J. Cheminform. 2021, **13**, 43.
DOI: [10.1186/s13321-021-00522-2](https://doi.org/10.1186/s13321-021-00522-2)
