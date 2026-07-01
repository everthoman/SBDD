# FpocketViewer v1.0

A PyMOL plugin for fpocket-based binding-site detection and pocket exploration.

**Authors:** Evert J. Homan, PhD; Claude (Anthropic)  
**License:** MIT

---

## Features

- Runs [fpocket](https://github.com/Discngine/fpocket) in the background on any loaded PyMOL protein object or an external PDB file
- Loads and visualizes alpha-sphere pockets from an existing fpocket `*_out/` directory
- Colour-coded alpha-sphere objects per pocket (up to 10 distinct colours, cycling)
- Sortable pocket metrics table: score, druggability score, volume, N spheres, SASA, hydrophobicity, and all other fpocket descriptors
- **Zoom to pocket** — centers the PyMOL view on the selected pocket
- **Toggle surface** — creates/removes a transparent surface of the protein residues lining the selected pocket (4.5 Å shell around alpha spheres)
- **Lining residues** — lists all protein residues within 4.5 Å of the selected pocket's alpha spheres
- Selecting a pocket in the table automatically highlights it and dims the others in the 3D view
- Optional fpocket detection parameter overrides (min/max α-sphere radius, min spheres per pocket, cluster distance)
- WSL support for Windows hosts

---

## Requirements

| Dependency | Notes |
|---|---|
| PyMOL ≥ 2.4 | Open-source or incentive build; Qt GUI required |
| [fpocket 4.x](https://github.com/Discngine/fpocket) | Must be on `PATH` or path entered manually |

---

## Installation

**Option A — Plugin Manager (recommended)**

`Plugin → Plugin Manager → Install New Plugin → choose FpocketViewer.py`

FpocketViewer then appears under the **Plugin** menu.

**Option B — run command**

```
PyMOL> run /path/to/FpocketViewer.py
PyMOL> fpv_gui
```

---

## Quick start

1. Load a protein structure in PyMOL.
2. Open the GUI: `Plugin → FpocketViewer` (or `fpv_gui`).
3. Select the protein object from the **PyMOL object** dropdown and click **Run fpocket**.
4. The pocket alpha spheres appear in the viewer; the table populates with scores.
5. Click a row to highlight that pocket, then use **Zoom to pocket**, **Toggle surface**, or **Get residues** to explore it.

---

## GUI reference

### Run fpocket

| Control | Description |
|---|---|
| **fpocket** field | Path to the fpocket binary. Auto-detected from `PATH` on startup. |
| PyMOL object / PDB file | Toggle between running on a loaded object (protein chain only is saved) or an external PDB. |
| **Detection parameters** | Collapsible panel for `-m`, `-M`, `-i`, `-D` overrides (leave collapsed for fpocket defaults). |
| **Run fpocket** | Launches fpocket in a background thread; progress streams to the log. |
| **Clear all** | Removes all `fpv_*` objects from the session and resets the table. |
| **Load \*_out/** | Load an already-computed fpocket output directory instead of re-running. |

### Pockets table

Each row is one fpocket pocket. Columns are sortable and reorderable.

| Column | Description |
|---|---|
| ● | Pocket colour indicator |
| Pocket | Name and alpha-sphere count |
| Score | fpocket pocket score |
| Drug. Score | Druggability score (0–1) |
| N Spheres | Number of alpha spheres |
| Volume | Estimated pocket volume (Å³) |
| Total / Polar / Apolar SASA | Solvent-accessible surface area contributions |
| … | All remaining fpocket descriptors |

**Row interactions**

- **Click a row** — highlights the selected pocket (opaque) and dims others in the 3D view.
- **Zoom to pocket** — flies the camera to the selected pocket with a 4 Å buffer.
- **Toggle surface** — creates a transparent coloured surface of the protein residues lining the pocket; click again to remove it.
- **Show all / Hide all** — enable or disable all pocket sphere objects at once.

### Lining residues

- **Protein sel** — PyMOL selection string for the receptor (default: `polymer.protein`). Change this if your protein uses a non-standard selection.
- **Get residues** — lists every unique residue within 4.5 Å of the selected pocket's alpha spheres (format: `RESN RESI:CHAIN`).

---

## CLI commands

```
fpv_load /path/to/protein_out/   Load an existing fpocket *_out/ directory
fpv_clear                        Remove all fpv_* objects from the session
fpv_gui                          Open the GUI
```

---

## PyMOL objects created

All objects are prefixed `fpv_` to avoid name collisions.

| Object | Contents |
|---|---|
| `fpv_p{n}_sph` | Alpha spheres for pocket *n* (loaded from `pocket{n}_vert.pqr`) |
| `fpv_p{n}_surf` | Transparent surface of pocket *n* lining residues (created on demand, toggle to remove) |

`fpv_clear` (or the **Clear all** button) removes all of these.

---

## Notes

- The plugin saves only `polymer.protein` atoms when exporting a PyMOL object to PDB for fpocket, so bound ligands and waters are excluded automatically.
- On Windows the plugin invokes fpocket via `wsl.exe`; fpocket must be installed inside WSL.
- The `*_out/` directory produced by fpocket is written next to the input PDB. When using the PyMOL-object path the PDB is saved to a system temp directory, so the output lands there too; the path is shown in the log and written to the **Load \*_out/** field for reference.
- Lining residues and pocket surfaces both use the **Protein sel** field, so you can restrict the receptor (e.g. `chain A and polymer.protein`) before clicking **Get residues** or **Toggle surface**.
