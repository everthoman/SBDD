# Contact Inspector

A PyMOL plugin for Maestro-inspired protein-ligand interaction visualization with support for multi-pose docking review.

**Authors:** Evert J. Homan, PhD; Claude (Anthropic)  
**License:** MIT

---

## Features

- Detects and visualizes all major non-covalent protein-ligand interactions
- Steps through docking poses (multi-state objects) or individual ligand objects
- Residue shell with line representation, CA labels, and a transparent surface
- Qt GUI panel with per-interaction-type toggles and display options
- Automatically detects whether to use states or objects mode

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

H-bonds use PyMOL's built-in `cmd.distance(mode=2)` polar contact detection. All other types are detected geometrically.

Contacts/clashes are hidden by default; enable them in the GUI.

---

## Installation

**Option A — Plugin Manager (recommended):**
1. In PyMOL: Plugin → Plugin Manager → Install New Plugin
2. Select `contact_inspector.py`

**Option B — Direct load:**
```
run /path/to/contact_inspector.py
ci_gui
```

---

## Quick start

1. Load your protein and ligand(s) into PyMOL
2. Open the GUI: `ci_gui`
3. Click **Setup** (auto mode detects whether ligands are separate objects or states of one object)
4. Step through poses with **Prev / Next**, arrow keys, or the Go-to spinner

---

## CLI commands

| Command | Description |
|---|---|
| `ci_gui` | Open the GUI panel |
| `ci_setup [protein [, ligands [, mode]]]` | Setup from command line |
| `ci_next` | Step to next pose |
| `ci_prev` | Step to previous pose |
| `ci_goto <index>` | Jump to pose by 0-based index |
| `ci_update` | Refresh interactions for current pose |
| `ci_clear` | Remove all contact inspector objects |

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
```

---

## Modes

**objects mode** — each ligand is a separate PyMOL object. The plugin cycles through them, enabling one at a time.

**states mode** — all docking poses are states of a single PyMOL object (e.g. gnina output). The plugin steps through states using `cmd.set("state", N)` globally.

**auto mode** — inspects loaded objects. Uses states mode if any object matching the ligand selection has more than one state; otherwise uses objects mode.

---

## GUI options

| Option | Default | Description |
|---|---|---|
| Interaction type checkboxes | Most on | Toggle individual interaction types on/off |
| Show distance labels | On | Show/hide Å labels on interaction dashes |
| Show surface | On | Show/hide the transparent pocket surface |
| Show residue labels | On | Show/hide CA residue name+number labels on the shell |

---

## Detection thresholds

| Interaction | Criterion |
|---|---|
| H-bonds | PyMOL polar contacts (mode=2) |
| Halogen bonds | Cl/Br/I donor ··· O/N/S acceptor, ≤ 3.5 Å |
| Salt bridges | Formal charged N ··· Asp/Glu O or Arg/Lys/His N ··· formal charged O, ≤ 4.0 Å |
| Aromatic H-bonds | Aromatic C-H ··· O/N/S, ≤ 3.8 Å, angle > 120° |
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
- The shell object (`shell`) shows residues within 5 Å of any loaded ligand as lines with CA labels; the surface (`_ci_surf`) covers atoms within 5 Å
- If a reference ligand is present alongside a multi-state docking poses object, both are used for shell/surface proximity and both are styled on setup
