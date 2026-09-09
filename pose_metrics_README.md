# pose_metrics.py — Score Docking Poses Against a Reference Ligand

Takes a protein structure (PDB), a reference ligand (SDF) and a stack of docking poses
(SDF, with whatever score fields the docking program wrote), and annotates every pose with
2D similarity, MCS RMSD, 3D shape similarity, interaction-fingerprint similarity and
PoseBusters flags. The same set of pose annotations the
[GNINA web app](https://github.com/gnina/gnina) produces, as a standalone CLI.

Results are written both as an **annotated SDF** (original fields preserved, metric fields
appended) and as a **CSV table** carrying every original pose field alongside the metrics.

---

## Metrics

| Field | What it measures | Needs |
|---|---|---|
| `Ref_Sim` | Morgan ECFP4 (r=2, 2048 bit) Tanimoto to the reference ligand. Both sides are neutralized and tautomer-canonicalized first, so a different protomer/tautomer of the same compound still scores 1.0. | reference |
| `MCS_RMSD` | Heavy-atom RMSD over the maximum common substructure with the reference, in Å. Symmetry-equivalent MCS mappings are all tried and the best is reported. | reference |
| `Shape_Sim` | RDKit shape Tanimoto, `1 - ShapeTanimotoDist`, against the reference. | reference |
| `PLIF_Sim` | Protein–ligand interaction fingerprint Tanimoto to the reference (ProLIF, or ODDT as fallback). | reference + receptor |
| `PB_Flags` | Number of failed PoseBusters checks (`dock` config). | receptor |
| `PB_Failed` | Names of the failing checks, `;`-separated. | receptor |

`MCS_RMSD` and `Shape_Sim` are computed **in place** by default: the poses already sit in the
binding site, so the comparison answers *"does this pose occupy the reference's position?"*.
Pass `--mcs-align` / `--shape-align` to superimpose first, which instead answers
*"how similar is this conformer/shape to the reference, wherever it sits?"*.

A metric that cannot be computed for a pose is written as `N/A` rather than dropping the pose.

---

## Dependencies

| Package | Role | Required |
|---|---|---|
| RDKit | SDF I/O, fingerprints, MCS, shape overlap | yes |
| tqdm | progress bars | yes |
| rich-argparse | coloured help | yes |
| argcomplete | tab completion | optional |
| PoseBusters (`bust`) | `pb` metric | for `pb` |
| ProLIF + MDAnalysis | `plif` metric (preferred backend) | for `plif` |
| ODDT + OpenBabel | `plif` metric (fallback backend) | for `plif` |

### Environment notes

The project env `sbdd` has ODDT but not ProLIF, so `PLIF_Sim` runs on the ODDT backend there —
fine, but ProLIF's interaction definitions are the ones the GNINA web app uses.

`sbdd` also ships PoseBusters 0.5.0 against RDKit 2022.09.5, a combination that raises
`Mol.GetProp(autoConvert=)` on every molecule. Point the tool at a working PoseBusters instead:

```bash
--bust-executable /home/evehom/Programs/miniconda3/envs/gnina_webapp/bin/bust
```

The `gnina_webapp` env (RDKit 2025.03, PoseBusters 0.6.5, ProLIF 2.1) has every backend, but
lacks `argcomplete` and `rich-argparse`; installing those two there runs the whole tool natively.

---

## Usage

```bash
# Every metric, writing results.sdf + results.csv
pose_metrics.py -r receptor.pdb -a ref.sdf -l poses.sdf -o results

# 2D/3D ligand comparison only — no receptor needed
pose_metrics.py -a ref.sdf -l poses.sdf -o results -m ref,mcs,shape

# PoseBusters only, keeping the full per-check report
pose_metrics.py -r receptor.pdb -l poses.sdf -o busted -m pb --pb-report checks.csv

# Ask how similar the shapes can be, rather than how they overlap as docked
pose_metrics.py -a ref.sdf -l poses.sdf -o results -m shape --shape-align
```

### Options

| Flag | Meaning |
|---|---|
| `-l, --ligands` | Docking poses, SDF or SDF.gz (required) |
| `-r, --receptor` | Protein PDB; required for `plif` and `pb` |
| `-a, --reference` | Reference ligand SDF; required for `ref`, `mcs`, `shape`, `plif` |
| `-o, --output` | Output stem: writes `NAME.sdf` and `NAME.csv` (default `pose_metrics`) |
| `-m, --metrics` | Comma-separated subset of `ref,mcs,shape,plif,pb` (default: `all`) |
| `--mcs-timeout` | Per-pose MCS search timeout in seconds (default 2) |
| `--mcs-align` | Superimpose on the MCS before measuring RMSD |
| `--shape-align` | Open3DAlign onto the reference before the shape comparison |
| `--plif-backend` | `auto` (default), `prolif` or `oddt` |
| `--bust-executable` | PoseBusters executable (default `bust` on PATH) |
| `--pb-chunk-size` | Poses per `bust` invocation (default 200) |
| `--pb-report` | Write the full per-check PoseBusters table to this CSV |
| `--id-field` | SDF field to use as the pose `Name` in the CSV |
| `-j, --workers` | Worker processes for `MCS_RMSD` (PoseBusters parallelizes internally) |
| `--no-sdf` / `--no-csv` | Skip either output |
| `-q, --quiet` | Suppress progress bars |

---

## Output

`NAME.sdf` — the input poses byte-for-byte, with the metric fields appended. Re-running over an
already-annotated SDF replaces those fields rather than duplicating them, so metrics can be added
in separate passes.

`NAME.csv` — one row per pose:

```
Name,Ligand_ID,minimizedAffinity,Ref_Sim,MCS_RMSD,Shape_Sim,PLIF_Sim,PB_Flags,PB_Failed
TH005006_148585,TH005006_148585,-7.08636,0.1455,6.1519,0.2240,0.2727,0,
```

`Name` is the molecule title, falling back to `Ligand_ID`/`ID`/`Name` (or `--id-field`) when
docking output leaves the title line blank, and finally to `pose_N`.

A run summary goes to the terminal:

```
Summary:
  Ref_Sim     n=20/20  min=0.095  mean=0.132  max=0.196
  MCS_RMSD    n=20/20  min=1.124  mean=4.868  max=7.864
  Shape_Sim   n=20/20  min=0.130  mean=0.244  max=0.399
  PLIF_Sim    n=20/20  min=0.083  mean=0.279  max=0.500  [oddt]
  PB_Flags    n=20/20  clean=19  flagged=1

PoseBusters failures by check:
  internal_steric_clash                    1
```

---

## Notes

- **Sanitization.** Docking output routinely fails a strict RDKit sanitize (odd valences, MDL
  bond type 4 outside a ring). Molecules are parsed with `sanitize=False` and repaired the same
  way the GNINA web app does — including resolving stray non-ring aromatic bonds, which would
  otherwise make an identical molecule fingerprint differently — so the two tools score alike.
- **Hydrogens.** `Ref_Sim`, `MCS_RMSD` and `Shape_Sim` strip explicit Hs so a pose with Hs and an
  H-free reference stay comparable. `PLIF_Sim` goes the other way and completes them, on the poses
  *and* the reference: H-bond donor perception needs them, and ProLIF's `VdWContact` sums van der
  Waals radii over every atom, hydrogens included — so H-completing one side only would give that
  side a richer contact profile and bias the similarity. The added positions come from idealised
  geometry, not from optimisation, so for rotatable donors (hydroxyls, protonated amines) the
  torsion is arbitrary; treat H-bond bits that depend on *added* Hs as softer evidence than ones
  placed by ligand prep.
- **Empty title lines.** SDF records are split without stripping leading blank lines — an empty
  molecule title is legal, and trimming it shifts the counts line and makes RDKit reject the
  whole molecule. Poses from GNINA typically have exactly that shape.
- **Failure isolation.** A pose PoseBusters cannot evaluate is isolated by bisection, so it costs
  only itself and not the rest of its chunk; those poses get `PB_Flags = N/A`.
