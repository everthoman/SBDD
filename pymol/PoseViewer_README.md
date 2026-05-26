# PoseViewer v1.5

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
- **Residue-colored surface**: pocket surface is colored by residue type (hydrophobic/polar/charged) rather than flat grey
- **Water-mediated H-bonds**: bridging crystal waters between ligand and protein are detected and drawn as two-segment dashes
- **Pose bookmarking**: mark interesting poses with ★ from the GUI; bookmarks persist across navigation and are visible in the pose table
- **Residue identity in pose table**: extracted ligands (auto-split from a protein structure) show their original `resn`, `resi`, and `chain` columns in the pose data table
- **Protein selector**: the Protein field is a dropdown listing all loaded protein objects, enabling quick switching between multiple structures in the same session
- **Docking poses mode**: explicit toggle that gates H-bond compare — avoids meaningless cross-pocket H-bond overlays when browsing extracted ligands from a multi-ligand crystal structure
- Reference ligand overlay: always-visible co-crystal/reference with its own interaction lines
- Pose data table: sortable, clickable table of docking scores and SD properties per pose
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

H-bond dashes are drawn via PyMOL's `cmd.distance(mode=2)` polar contact detection. Individual H-bond pairs are also listed in the console summary via geometric D-A distance detection (N/O/S/F within 3.5 Å). All other interaction types are detected geometrically. Non-polar hydrogens (C-H) are excluded from clash detection. Contacts/clashes are hidden by default.

Reference ligand interactions are drawn with the same color scheme but thinner dashes (65% radius) to distinguish them from pose interactions.

### Surface residue color scheme

| Color | Residue type | Residues |
|---|---|---|
| Wheat | Hydrophobic | ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO |
| Pale green | Polar | SER, THR, TYR, CYS, ASN, GLN |
| Blue | Positive charged | ARG, LYS, HIS |
| Red | Negative charged | ASP, GLU |
| Grey | Other | GLY, etc. |

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
3. The pose data table shows each ligand's original residue name and number (`resn`, `resi`, `chain`)
4. No manual extraction needed

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
| `ci_bookmarks` | List all bookmarked poses to the console |
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

Sortable table showing SD data tag properties for all poses (e.g. `minimizedAffinity`, `CNNscore` from GNINA). Column headers are movable. Rank columns are excluded. Requires a scores SDF to be loaded, or Incentive PyMOL.

When browsing auto-split ligands from a protein structure (no SDF), the table shows `resn`, `resi`, and `chain` columns derived from the original PDB residue identity of each ligand.

**Single-click** a row to navigate to that pose. **Ctrl-click** (or click a second row) to enter compare mode — the two most recently selected rows are shown simultaneously. A third selection automatically drops the oldest, maintaining a rolling window of two. Clicking Prev/Next or Go exits compare mode and resumes single-pose navigation.

### Interaction groups (Non-covalent bonds / Pi interactions / Contacts/Clashes)

Each group has:
- An **enable checkbox** (bold title) — toggles all interactions in that group on/off independently
- A **collapse arrow** (▶/▼) — hides/shows the group body without affecting the enable state

Individual interaction types can be toggled within each group. Contacts/Clashes are disabled by default.

### Display group

| Option | Default | Description |
|---|---|---|
| Show distance labels | On | Show/hide Å labels on interaction dashes |
| Show surface | On | Show/hide the transparent pocket surface |
| Show residue labels | On | Show/hide CA residue name+number labels on the shell |
| Auto-zoom to pose | On | Zoom to binding site on each pose change |
| Show nonpolar H on ligands | Off | Show all hydrogens (including nonpolar C-H) on pose and reference ligands as sticks. Off by default (only polar H on N/O/S shown). |

The Display group enable checkbox hides all display elements at once (surface, labels) without changing individual settings.

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
| H-bonds (visual dashes) | PyMOL polar contacts `cmd.distance(mode=2)` |
| H-bonds (console listing) | N/O/S/F ··· N/O/S/F donor-acceptor distance ≤ 3.5 Å |
| Halogen bonds | Cl/Br/I donor ··· O/N/S acceptor, ≤ 3.5 Å |
| Salt bridges | Formal charged N ··· Asp/Glu O or Arg/Lys/His N ··· formal charged O, ≤ 4.0 Å |
| Aromatic H-bonds | Aromatic C ··· O/N/S acceptor, ≤ 3.5 Å, C-H···A angle > 120° |
| Water bridges | HOH oxygen simultaneously within 3.5 Å of ligand N/O/S/F and protein N/O/S/F |
| Pi-pi face-to-face | Centroid distance ≤ 4.8 Å, normal angle ≤ 40° |
| Pi-pi edge-to-face | Centroid distance ≤ 5.5 Å, normal angle 45–90° |
| Pi-cation | Ring centroid ··· Arg CZ / Lys NZ or formal+ atom, ≤ 6.0 Å |
| Good contact | d ≤ 1.30× VDW sum, heavy atoms only |
| Bad clash | d < 0.89× VDW sum |
| Ugly clash | d < 0.75× VDW sum |

---

## Notes

- Requires PyMOL with Qt support (PyMOL 2.x+)
- `numpy` is used if available; falls back to pure Python otherwise
- **Incentive PyMOL**: SDF data fields are preserved on load and read automatically via `get_property_list` / `get_property` — no scores file needed, leave the Scores field blank
- **Open-source PyMOL**: SDF data fields are stripped on load. Scores must be loaded from the original SDF file via the Scores field or `ci_load_scores`
- The shell shows residues within 5 Å of the current ligand as lines with CA labels; the surface covers atoms within 5 Å. In objects mode both update per ligand step; in states mode they are computed once at setup
- Duplicate interactions caused by alternate conformations (altloc atoms) in PDB structures are automatically removed by spatial deduplication
- Water bridges require HOH residues in the loaded structure. HOH is searched globally (not limited to `polymer.protein`), so crystallographic waters in the protein PDB are detected even when the protein selection excludes them
- Bookmarks are per-session only and are cleared by `ci_clear` or a new `ci_setup`
