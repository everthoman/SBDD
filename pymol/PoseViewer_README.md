# PoseViewer v1.8

A PyMOL plugin for Maestro-inspired protein-ligand interaction visualization with support for multi-pose docking review and multi-ligand structure browsing.

**Authors:** Evert J. Homan, PhD; Claude (Anthropic)  
**License:** MIT

---

## Features

- Detects and visualizes all major non-covalent protein-ligand interactions
- Steps through docking poses (multi-state objects) or individual ligand objects
- **Auto-split**: load any PDB with multiple HETATM ligands and PoseViewer automatically separates them into individual objects for per-ligand browsing and per-pocket surface display
- **Compare mode**: select any two poses simultaneously — including pose #3 of ligand A vs pose #7 of ligand B — to overlay them in the binding site with distinct colors
- Per-ligand pocket surface: residue shell, CA labels, and transparent surface update to the current ligand's binding site in objects mode
- **Charge-colored surface**: pocket surface is colored by charged atom (blue for cationic N, red for anionic O) rather than flat grey
- **Water-mediated H-bonds**: bridging crystal waters between ligand and protein are detected and drawn as two-segment dashes
- **Pose bookmarking**: mark interesting poses with ★ from the GUI; bookmarks are tied to the pose itself, so they stay put when objects are added, deleted or renumbered, and are visible in the pose table
- **Table export**: copy the pose table to the clipboard or write it to CSV/TSV, from the GUI or via `ci_export`
- **Protein selector**: the Protein field is a dropdown listing all loaded protein objects, enabling quick switching between multiple structures in the same session
- **Docking poses mode**: explicit toggle that gates H-bond compare — avoids meaningless cross-pocket H-bond overlays when browsing extracted ligands from a multi-ligand crystal structure
- Reference ligand overlay: always-visible co-crystal/reference with its own interaction lines
- Pose data table: sortable, clickable table of docking scores and SD properties per pose
- **Computed pose metrics**: MCS RMSD, 3D shape similarity, 2D similarity, PLIF similarity and PoseBusters flags are calculated from what is loaded in the session, so they are available even when the docking program never wrote them into the SDF
- Qt GUI panel with collapsible groups and per-type interaction toggles
- Interaction summary printed to the PyMOL console on every step

### Interaction types

| Category | Type | Color |
|---|---|---|
| Non-covalent | H-bonds | Yellow |
| Non-covalent | Halogen bonds | Purple |
| Non-covalent | Salt bridges | Magenta |
| Non-covalent | Aromatic H-bonds | Green |
| Non-covalent | Water bridges | Light blue |
| Pi | Pi-pi stacking (face-to-face & edge-to-face) | Cyan |
| Pi | Pi-cation | Green |
| Contacts | Good contacts (≤ 1.30× VDW sum, heavy atoms) | Green |
| Clashes | Bad clashes (< 0.89× VDW sum) | Orange |
| Clashes | Ugly clashes (< 0.75× VDW sum) | Red |

H-bonds are found by PyMOL's `cmd.distance(mode=2)` polar contact detection, then vetted on geometry and redrawn, so the picture and the console listing are always the same set.

**The angle filter.** PyMOL's `h_bond_max_angle` (default 63°) is measured *at the donor heavy atom* — between the D–H bond and the D···A vector — not on the D–H···A angle that gets quoted in papers. A 63° cone at the donor admits D–H···A angles down to about 100°, i.e. a perfect-looking donor–acceptor distance with the hydrogen pointing somewhere else entirely. PoseViewer therefore measures the angle at the proton and drops anything below **`hbond_min_angle`, 130° by default** (`ci_hbond_angle`, or the spin box beside the Hydrogen bonds checkbox; 0 disables it). The summary reports each surviving H-bond's angle and says how many were filtered out.

A real example, a uracil fragment in UNG2: the ring N–H sits 2.91 Å from a backbone carbonyl O — textbook distance — but at a D–H···A angle of 102°, so the proton is not pointing at the acceptor at all. PyMOL draws it because the angle at the donor is 58°, inside its 63° default. The filter removes it; the two genuine H-bonds in that site (144° and 169°) are untouched.

**This needs explicit hydrogens.** A contact with no hydrogen on either endpoint has nothing to measure, and is passed through unfiltered rather than guessed at — so the filter does its work on protonated or MD-minimised structures and stays out of the way on a bare PDB from the RCSB. (Before v1.6 the summary used a separate proximity rule — any N/O/S/F pair within 3.5 Å — which counted acceptor–acceptor pairs such as two carbonyl oxygens as H-bonds and could also miss ones PyMOL drew.) All other interaction types are detected geometrically.

Salt bridges and the ligand side of pi-cation need to know which ligand atoms are charged. SDF and mol2 carry formal charges and are used as-is; PDB has no charge column, so a ligand read out of a complex arrives neutral. In that case PoseViewer infers the groups that are ionised at physiological pH — carboxylate/phosphate/sulfonate as anions, quaternary N, guanidinium/amidinium and non-aromatic aliphatic amines as cations — and deliberately stays silent on ring nitrogens, anilines and ureas, whose basicity depends on context. Non-polar hydrogens (C-H) are excluded from clash detection. Contacts/clashes are hidden by default.

Reference ligand interactions are drawn with the same color scheme but thinner dashes (65% radius) to distinguish them from pose interactions.

### Surface residue color scheme

| Color | Residue type | Residues |
|---|---|---|
| Blue | Positive charged | ARG, LYS, HIS |
| Red | Negative charged | ASP, GLU |
| Grey | Everything else | — |

---

## Installation

**Option A — Plugin Manager (recommended):**
1. In PyMOL: Plugin → Plugin Manager → Install New Plugin
2. Select `pymol/PoseViewer.py`

**Option B — Direct load:**
```
run /path/to/pymol/PoseViewer.py
ci_gui
```

---

## Quick start

### Docking poses (multi-state object)

1. Load your protein and docking output into PyMOL
2. Open the GUI: `ci_gui`
3. Fill in protein/ligand fields and click **Setup**
4. Step through poses with **Prev / Next** or the Go-to spinner
5. Optionally load a scores SDF to display docking properties per pose

### Multi-ligand PDB (e.g. crystal structure with cofactors)

1. Load the PDB: `load 5VDH.pdb`
2. Run `ci_gui` and click **Setup** — PoseViewer auto-splits the organic ligands into `obj01`, `obj02`, ... and steps through each with its own pocket surface
3. No manual extraction needed

### Two protein structures in the same session

1. Load both structures: `load 3FCI.pdb` and `load 5VDH.pdb`
2. Open `ci_gui` — the Protein dropdown lists both objects
3. Select the structure you want to inspect and click **Setup**

---

## CLI commands

| Command | Description |
|---|---|
| `ci_gui` | Open the GUI panel |
| `ci_setup [protein [, ligands [, mode]]]` | Setup from command line |
| `ci_next` | Step to next pose |
| `ci_prev` | Step to previous pose |
| `ci_goto <index>` | Jump to pose by 0-based index |
| `ci_update` | Re-detect interactions for current pose |
| `ci_refresh` | Sync panel to current PyMOL state |
| `ci_load_scores <path>` | Load per-pose SD properties from an SDF file |
| `ci_calc [metrics]` | Compute pose metrics (see [Pose metrics](#pose-metrics)) |
| `ci_bookmarks` | List all bookmarked poses to the console |
| `ci_export <path> [, bookmarked]` | Write the pose table (SD properties plus computed metrics) to CSV; a `.tsv`/`.txt` extension switches to tab-separated |
| `ci_hbond_angle [degrees]` | Show or set the minimum D–H···A angle for H-bonds (default 130°, 0 = off) |
| `ci_clear` | Remove all PoseViewer objects |

### `ci_setup` parameters

- **protein** — PyMOL selection for the receptor (default: `polymer.protein`)
- **ligands** — object name(s) or selection keyword (default: `organic`)
  - Comma-separated list for objects mode: `LIG1,LIG2,LIG3`
  - Single object name with multiple states for states mode: `poses`
- **mode** — `auto` (default), `objects`, or `states`

### Examples

```
ci_setup
ci_setup protein=chain A, ligands=LIG1,LIG2,LIG3
ci_setup protein=polymer.protein, ligands=poses, mode=states
ci_load_scores /path/to/gnina_output.sdf
ci_calc                         # MCS_RMSD, Shape_Sim, Ref_Sim
ci_calc all
ci_calc mcs_rmsd,plif_sim
ci_export /path/to/poses.csv
ci_export /path/to/marked.tsv, bookmarked
ci_hbond_angle 145              # stricter H-bond geometry
ci_hbond_angle 0                # off: show whatever PyMOL reports
```

---

## Modes

**objects mode** — each ligand is a separate PyMOL object. The plugin cycles through them, enabling one at a time. The pocket surface updates per ligand.

**states mode** — all docking poses are states of a single PyMOL object (e.g. GNINA output). The plugin steps through states. The pocket surface is built once at setup (all poses share the same binding site).

**auto mode** — inspects loaded objects. Uses states mode if exactly one object matching the ligand selection has more than one state; otherwise uses objects mode.

### Auto-split

If no separate ligand objects are detected (e.g. a PDB loaded as a single object), `ci_setup` splits the organic selection by `(chain, resn, resi)` into individual PyMOL objects named `obj01`, `obj02`, ... following PyMOL's own extract naming convention. The original atoms in the source object are hidden. Auto-split objects are cleaned up by `ci_clear` or the next `ci_setup`.

---

## GUI reference

### Setup group

| Field | Description |
|---|---|
| Protein | Editable dropdown listing all loaded objects that contain protein atoms, plus the default `polymer.protein`. Refreshes automatically when objects are added or removed. Custom selection strings can be typed directly. |
| Ligand(s) | Object name(s) or selection (comma-separated for objects mode) |
| Scores (SDF) | Optional path to an SDF file with per-pose SD data tags (e.g. GNINA output). Browse button available. Scores are read directly from the file since open-source PyMOL does not preserve SDF properties on load. |

### Navigate group

| Control | Description |
|---|---|
| Prev / Next | Step through poses in current table sort order (exits compare mode) |
| Refresh | Re-detect interactions for the current PyMOL state |
| Go to # | Jump to pose by 1-based number (exits compare mode) |
| Docking poses | Marks the session as a docking run. Auto-checked when multi-state objects are detected; must be ticked manually for single-pose-per-ligand docking sessions. Gates the H-bonds in compare mode checkbox. |
| H-bonds in compare mode | When checked, H-bond dashes are drawn during compare mode, colored to match each pose. Only available when **Docking poses** is checked (not meaningful when each ligand sits in a different pocket). |

### Reference ligand group

Selects a persistent reference ligand (e.g. co-crystal structure) that remains visible alongside every pose and always shows its own interaction lines. The reference is colored magenta (C atoms) to distinguish it from docking poses.

| Control | Description |
|---|---|
| Object dropdown | Lists all organic objects that are not the receptor. Auto-populated on Setup; can be overridden. Select `(none)` to disable and hide the reference ligand. |
| Show ref | Hides/shows the reference ligand object and all its interaction lines |
| Show pose | Hides/shows the current docking pose object and its interaction lines. Uncheck to isolate the reference ligand and its interactions in the view. |

Reference ligand interaction lines respect the same **Show distance labels** toggle as pose interactions.

### Pose Data group

Sortable table showing SD data tag properties for all poses (e.g. `minimizedAffinity`, `CNNscore` from GNINA). Column headers are movable. Rank columns are excluded.

**Where the columns come from.** Only Incentive PyMOL reads SD tags off a loaded SDF automatically; open-source PyMOL discards them at load. Everywhere else you must point the **Scores (SDF)** field at the poses file (or run `ci_load_scores`) *before* pressing Setup, otherwise the table shows only the `Ligand_ID` column and none of the docking scores. Setup prints a note to the console when it ends up in that state.

**Single-click** a row to navigate to that pose. **Ctrl-click** (or click a second row) to enter compare mode — the two most recently selected rows are shown simultaneously. A third selection automatically drops the oldest, maintaining a rolling window of two. Clicking Prev/Next or Go exits compare mode and resumes single-pose navigation.

#### Getting the data out

| Control | Description |
|---|---|
| Copy all | Copies the whole table to the clipboard as tab-separated text — paste straight into Excel, Numbers or a notebook |
| Ctrl+C | With the table focused, copies just the selected rows |
| Export… | Writes the table to a file; `.csv` gives commas, `.tsv`/`.txt` gives tabs |

All three follow what you see: the current sort order, any columns you have dragged around, and the ★ column (exported as a `Bookmarked` field). Numeric cells are written at full precision rather than the two decimals the table displays, so computed metrics survive the trip into a spreadsheet. The `ci_export` command does the same thing without the GUI.

### Calculate group

Computes the pose metrics described in [Pose metrics](#pose-metrics) for every pose in
the current setup.

| Control | Description |
|---|---|
| Metric checkboxes | MCS RMSD, shape similarity, 2D similarity are on by default; PLIF similarity and PoseBusters flags are opt-in (slower, and they need a second interpreter) |
| Calculate | Exports poses/reference/receptor, then runs the selected metrics on a background thread |
| Cancel | Stops the run; metrics already finished are kept |
| Progress bar / status | Per-metric progress, then a per-field count of how many poses got a value. Errors are shown here and printed to the console |

### Interaction groups (Non-covalent bonds / Pi interactions / Contacts/Clashes)

Each group has:
- An **enable checkbox** (bold title) — toggles all interactions in that group on/off independently
- A **collapse arrow** (▶/▼) — hides/shows the group body without affecting the enable state

Individual interaction types can be toggled within each group. Contacts/Clashes are disabled by default.

The **Non-covalent bonds** group carries a `min D–H···A angle` spin box under the Hydrogen bonds checkbox, which is the same setting as `ci_hbond_angle` (see [Interaction types](#interaction-types)). Changing it re-detects immediately.

### Display group

| Option | Default | Description |
|---|---|---|
| Show distance labels | On | Show/hide Å labels on interaction dashes |
| Show surface | On | Show/hide the transparent pocket surface |
| surface reach | 5.0 Å | How far the pocket surface extends around the ligand. The surface is carved from the protein's molecular surface at this radius; larger shows more of the pocket wall. Rebuilds the surface on change. |
| Show residue labels | On | Show/hide CA residue name+number labels on the shell |
| Auto-zoom to binding site | On | Frame the residue shell on each pose change, rather than the ligand alone. Zooming the ligand puts the camera on top of it and the pocket falls out of view; on this test case the shell spans 20 Å against 7 Å for a single pose. Set `_stepper.zoom_to_shell = False` for the old ligand-only framing, or `_stepper.zoom_buffer` to change the padding (2 Å default) |
| Show nonpolar H on ligands | Off | Show all hydrogens (including nonpolar C-H) on pose and reference ligands as sticks. Off by default (only polar H on N/O/S shown). |

The Display group enable checkbox hides all display elements at once (surface, labels). Unticking it remembers what was on; ticking it again restores exactly that, rather than switching everything on — which used to turn on nonpolar ligand H even though it defaults to off.

---

## Pose metrics

Docking output usually carries only the program's own score (`minimizedAffinity`,
`CNNscore`, ...). PoseViewer can compute five further per-pose metrics from what is
already loaded in the session — the poses, the reference ligand and the receptor —
so no re-docking or external post-processing pass is needed. Results are added to the
Pose Data table as ordinary sortable columns.

Field names match those written by the GNINA webapp, so a computed value and one read
from the SDF are interchangeable; computing a metric overrides the SDF value for that
column.

| Field | Metric | Needs | Notes |
|---|---|---|---|
| `MCS_RMSD` | Heavy-atom RMSD over the maximum common substructure with the reference | reference | Symmetry-aware (all MCS matchings tried, best kept). **No superposition** — poses and reference already share the receptor frame, so this measures pose deviation, not conformer difference |
| `Shape_Sim` | 3D shape Tanimoto vs the reference (1 = identical) | reference | Computed on the poses as placed, again without alignment, so it measures overlap with the reference in the site |
| `Ref_Sim` | 2D Morgan ECFP4 (r=2, 2048 bit) Tanimoto vs the reference | reference | Both molecules are neutralized and tautomer-canonicalized first, so a different protomer/tautomer of the same compound scores 1.0 |
| `PLIF_Sim` | ProLIF interaction-fingerprint Tanimoto vs the reference | reference + receptor + `prolif` | Reference and all poses run through one `Fingerprint` instance so the bitvectors stay comparable |
| `PB_Flags` | Number of failed PoseBusters checks (0 = all pass) | receptor + `posebusters` | `dock` config: ligand internal geometry plus intermolecular checks against the receptor |

### Running it

**GUI** — the **Calculate** group (bottom panel): tick the metrics, click **Calculate**.
The run happens on a background thread with a progress bar, so PyMOL stays responsive,
and **Cancel** stops it while keeping whatever finished. Reference-based metrics are
dropped automatically when you pick a different reference ligand.

**Command line** — `ci_calc`, which blocks and prints progress:

```
ci_calc                      # MCS_RMSD, Shape_Sim, Ref_Sim (the default set)
ci_calc all                  # adds PLIF_Sim and PB_Flags
ci_calc mcs_rmsd,plif_sim    # or: MCS_RMSD,PLIF_Sim — field names work too
```

An open GUI picks up `ci_calc` results on its next sync tick (within ~0.5 s).

### Dependencies

`MCS_RMSD`, `Shape_Sim` and `Ref_Sim` need only **RDKit** in the interpreter running
PyMOL, and run in-process.

`PLIF_Sim` and `PB_Flags` need `prolif` + `MDAnalysis` and `posebusters` respectively,
which PyMOL's own environment rarely has. PoseViewer therefore runs them in a
**subprocess under another interpreter**, found by scanning, in order:

1. `$POSEVIEWER_PYTHON`
2. the interpreter running PyMOL
3. conda environments under `$CONDA_PREFIX/..`, `~/miniconda3/envs`, `~/anaconda3/envs`,
   `~/Programs/miniconda3/envs`, `/opt/conda/envs`

The first environment whose `site-packages` contains the needed package wins (checked on
the filesystem, not by importing, so the scan is instant). If none is found, the
Calculate panel says so and points at `$POSEVIEWER_PYTHON`; the other metrics still run.
Set it explicitly to skip the search:

```bash
export POSEVIEWER_PYTHON=~/Programs/miniconda3/envs/gnina_webapp/bin/python
```

### What gets sent where

Poses, reference and receptor are exported from the live session on the main thread
(`cmd.get_str("mol", ...)` per pose, `cmd.save` for the receptor) before any work
starts — the background thread and the subprocess only ever see molblock strings and
temporary files, never the PyMOL API. Molecules are sanitized once in PyMOL's process,
including a fix for the non-ring aromatic bonds PyMOL writes for delocalized groups
(a carboxylate exported as `C(:O):[O-]`), which would otherwise change fingerprints and
make PoseBusters reject the pose. Explicit hydrogens are added before the external run
because ProLIF perceives H-bond donors from them. The temporary directory is removed
when the job ends.

### Cost

Per pose, roughly: `Ref_Sim` and `Shape_Sim` are milliseconds, `MCS_RMSD` tens of
milliseconds, `PLIF_Sim` is one batched run of a few seconds plus the receptor parse,
and `PB_Flags` is by far the most expensive (order of a second per pose, parallelized
inside the worker). Tick PoseBusters only for a shortlist, not a full docking run.

### Caveats

- Poses loaded from a PDB have no bond orders, so RDKit's perception of them is a guess —
  metrics are most reliable for poses loaded from SDF/MOL2
- `MCS_RMSD` and `Shape_Sim` need a reference with 3D coordinates; `MCS_RMSD` is `N/A`
  when the common substructure has fewer than 3 atoms
- `PLIF_Sim` uses the receptor as loaded. A receptor without hydrogens limits ProLIF's
  H-bond perception, so prepare (protonate) the structure for the most meaningful numbers
- The receptor passed to both external metrics is the current **Protein** selection

---

## Compare mode

Compare mode lets you overlay any two poses side-by-side in the binding site, regardless of which ligand object or state they come from — e.g. pose #3 of ligand A vs pose #7 of ligand B.

**How it works:**  
Each selected pose is extracted into a temporary single-state PyMOL object (`_cmp_0`, `_cmp_1`) via `cmd.create`. This bypasses PyMOL's global state slider, which would otherwise force both objects to the same state number. The temporary objects are automatically removed when you navigate away or click Clear.

**Colors:**  
Poses from different ligand objects are colored by their object's palette entry (assigned at Setup). Poses from the same object (e.g. two states of the same docking run) receive distinct slot colors instead. The color palette is: cyan, orange, forest green, hotpink, violet, salmon. The reference ligand always remains magenta.

**Interactions:**  
All interaction types are hidden in compare mode to keep the view uncluttered — the spatial overlay is the primary information. Enable **H-bonds in compare mode** in the Navigate group to overlay H-bond dashes colored to match each pose. This option is only available when **Docking poses** is checked, since H-bond comparison is only meaningful when poses share the same binding site.

**Limitations:**  
Maximum two poses at a time (excluding the reference ligand). The full interaction panel (all types) remains available in single-pose mode.

---

## Detection thresholds

| Interaction | Criterion |
|---|---|
| H-bonds (dashes and console listing) | PyMOL polar contacts `cmd.distance(mode=2)`, then D–H···A ≥ `hbond_min_angle` (130° default) wherever an explicit hydrogen exists |
| Halogen bonds | Cl/Br/I donor ··· O/N/S acceptor, ≤ 3.5 Å |
| Salt bridges | Cationic ligand N ··· Asp/Glu O, or Arg/Lys/His N ··· anionic ligand O, ≤ 4.0 Å. Ligand charges come from the file when it has them, otherwise inferred (see above) |
| Aromatic H-bonds | Aromatic C ··· O/N/S acceptor, ≤ 3.5 Å, C-H···A angle > 120° |
| Water bridges | HOH oxygen simultaneously within 3.5 Å of ligand N/O/S/F and protein N/O/S/F |
| Pi-pi face-to-face | Centroid distance ≤ 4.8 Å, normal angle ≤ 40° |
| Pi-pi edge-to-face | Centroid distance ≤ 5.5 Å, normal angle 45–90° |
| Pi-cation | Ligand ring centroid ··· Arg CZ / Lys NZ, or cationic ligand atom ··· Phe/Tyr/Trp/His ring centroid, ≤ 6.0 Å |
| Good contact | d ≤ 1.30× VDW sum, heavy atoms only |
| Bad clash | d < 0.89× VDW sum |
| Ugly clash | d < 0.75× VDW sum |

---

## Notes

- Requires PyMOL with Qt support (PyMOL 2.x+)
- `rdkit` is required only for the Calculate group; it is imported on first use, so the plugin loads normally without it
- `numpy` is optional. It is used only for the best-fit-plane SVD in the aromatic ring planarity test, which falls back to Newell's method without it. The per-pair geometry is deliberately scalar Python — numpy's per-call overhead dominates on 3-vectors and made pose stepping about twice as slow
- **Incentive PyMOL**: SDF data fields are preserved on load and read automatically via `get_property_list` / `get_property` — no scores file needed, leave the Scores field blank
- **Open-source PyMOL**: SDF data fields are stripped on load. Scores must be loaded from the original SDF file via the Scores field or `ci_load_scores`
- The shell shows residues within 5 Å of the current ligand as lines with CA labels. The pocket surface is a carved patch of the real protein molecular surface: the solvent-excluded surface is computed on a wider residue shell (8 Å) for correct geometry, then trimmed to the wall within the **surface reach** distance of the ligand (5 Å default, adjustable in the Display group) — so it hugs the binding site rather than closing over into a blob around whole side chains. In objects mode shell and surface update per ligand step; in states mode they are computed once at setup, around the first pose
- **Large pose sets** are fine: Setup on a 1600-pose SDF takes well under a second, and stepping is ~20 ms per pose. Every distance-based selection is evaluated against a single-state copy of the current pose and pinned to state 1, because PyMOL's default evaluates `within` once per state in the session — 0.001 s at one state, 0.13 s at 800 — which is what used to make PyMOL appear to hang on a full docking run. A corollary: the shell and surface follow the pose on screen rather than the union of every pose
- Duplicate interactions caused by alternate conformations (altloc atoms) in PDB structures are automatically removed by spatial deduplication
- Water bridges require HOH residues in the loaded structure. HOH is searched globally (not limited to `polymer.protein`), so crystallographic waters in the protein PDB are detected even when the protein selection excludes them
- Bookmarks are per-session only, and are keyed to the pose itself — `(object, state)` — so they follow their pose when the pose list is rebuilt (objects added, deleted or renumbered) and survive a re-`ci_setup` of the same poses. The GUI **Clear** button drops them; `ci_clear` only removes PoseViewer's objects
