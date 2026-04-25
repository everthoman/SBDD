# PoseViewer — beta

A PyMOL plugin for Maestro-inspired protein-ligand interaction visualization with support for multi-pose docking review.

**Authors:** Evert J. Homan, PhD; Claude (Anthropic)  
**License:** MIT  
**Status:** beta

---

## Features

- Detects and visualizes all major non-covalent protein-ligand interactions
- Steps through docking poses (multi-state objects) or individual ligand objects
- Residue shell with line representation, CA labels, and a transparent surface
- Reference ligand overlay: always-visible co-crystal/reference with its own interaction lines
- Pose data table: sortable, clickable table of docking scores and SD properties per pose
- Qt GUI panel with collapsible groups and per-type interaction toggles

### Interaction types

| Category | Type | Color |
|---|---|---|
| Non-covalent | H-bonds (PyMOL polar contacts) | Yellow |
| Non-covalent | Halogen bonds | Purple |
| Non-covalent | Salt bridges | Magenta |
| Non-covalent | Aromatic H-bonds | Green |
| Pi | Pi-pi stacking (face-to-face & edge-to-face) | Cyan |
| Pi | Pi-cation | Green |
| Contacts | Good contacts (≤ 1.30× VDW sum, heavy atoms) | Green |
| Clashes | Bad clashes (< 0.89× VDW sum) | Orange |
| Clashes | Ugly clashes (< 0.75× VDW sum) | Red |

H-bonds use PyMOL's built-in `cmd.distance(mode=2)` polar contact detection. All other types are detected geometrically. Non-polar hydrogens (C-H) are excluded from clash detection. Contacts/clashes are hidden by default.

Reference ligand interactions are drawn with the same color scheme but thinner dashes (65% radius) to distinguish them from pose interactions.

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

1. Load your protein and ligand(s) into PyMOL
2. Open the GUI: `ci_gui`
3. Fill in protein/ligand fields and click **Setup**
4. Step through poses with **Prev / Next**, arrow keys, or the Go-to spinner
5. Optionally: specify a scores SDF file to display docking properties per pose

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

**objects mode** — each ligand is a separate PyMOL object. The plugin cycles through them, enabling one at a time.

**states mode** — all docking poses are states of a single PyMOL object (e.g. GNINA output). The plugin steps through states.

**auto mode** — inspects loaded objects. Uses states mode if exactly one object matching the ligand selection has more than one state; otherwise uses objects mode.

---

## GUI reference

### Setup group

| Field | Description |
|---|---|
| Protein | PyMOL selection for the receptor |
| Ligand(s) | Object name(s) or selection (comma-separated for objects mode) |
| Mode | auto / objects / states |
| Scores (SDF) | Optional path to an SDF file with per-pose SD data tags (e.g. GNINA output). Browse button available. Scores are read directly from the file since open-source PyMOL does not preserve SDF properties on load. |

### Navigate group

| Control | Description |
|---|---|
| Prev / Next | Step through poses in current table sort order |
| Refresh | Re-detect interactions for the current PyMOL state |
| Go to # | Jump to pose by 1-based number |

### Reference ligand group

Selects a persistent reference ligand (e.g. co-crystal structure) that remains visible alongside every pose and always shows its own interaction lines. The reference is colored magenta (C atoms) to distinguish it from docking poses.

| Control | Description |
|---|---|
| Object dropdown | Lists all organic objects that are not the receptor. Auto-populated on Setup; can be overridden. Select `(none)` to disable and hide the reference ligand. |
| Show checkbox | Hides/shows both the reference ligand object and all its interaction lines |

### Pose Data group

Sortable, clickable table showing SD data tag properties for all poses (e.g. `minimizedAffinity`, `CNNscore` from GNINA). Clicking a row navigates to that pose. Prev/Next step in current sort order. Column headers are movable. Rank columns are excluded (redundant with score sorting). Requires a scores SDF to be loaded, or Incentive PyMOL.

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

The Display group enable checkbox hides all display elements at once (surface, labels) without changing individual settings.

---

## Detection thresholds

| Interaction | Criterion |
|---|---|
| H-bonds | PyMOL polar contacts (mode=2) |
| Halogen bonds | Cl/Br/I donor ··· O/N/S acceptor, ≤ 3.5 Å |
| Salt bridges | Formal charged N ··· Asp/Glu O or Arg/Lys/His N ··· formal charged O, ≤ 4.0 Å |
| Aromatic H-bonds | Aromatic C ··· O/N/S acceptor, ≤ 3.5 Å, C-H···A angle > 120° |
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
- **Open-source PyMOL**: SDF data fields are stripped on load (`get_property_list` and `properties` in `iterate` are incentive-only). Scores must be loaded from the original SDF file via the Scores field or `ci_load_scores`
- The shell shows residues within 5 Å of any ligand (pose + reference) as lines with CA labels; the surface covers atoms within 5 Å
