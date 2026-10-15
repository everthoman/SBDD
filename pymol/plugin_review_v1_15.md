# PoseViewer v1.15 — code review

**File reviewed:** `/attachments/PoseViewer_v1_15.py` (3761 lines, unmodified)
**Reviewer:** Codon · **Date:** 2026-10-15
**Scope:** correctness, PyMOL API use, UI/threading, robustness, file/subprocess security, maintainability, performance.

## How this was checked, and what it could not check

PyMOL is not installed in this workspace, so **nothing was live-tested inside PyMOL**: no session was
loaded, no GUI opened, no interaction detection run against a real structure. Everything below is
either a static reading of the source, a reproduction of a *pure* helper in isolation, or a claim
checked against PyMOL's own source.

Checks actually run:

| Check | Result |
|---|---|
| `pyflakes` on the full file | clean — no undefined names, no unused imports |
| `grep` for `subprocess`, `os.system`, `popen`, `eval(`, `exec(`, `__import__`, `pickle`, `socket`, `urllib`, `requests` | **zero hits** — no process spawning, no code evaluation, no network |
| AST scan for version-gating syntax | one walrus operator (L2244) ⇒ hard floor of Python 3.8 / PyMOL ≈ 2.4+, with no guard or message |
| Pure helpers executed against synthetic chempy-like models (fake `pymol`/`chempy` modules; no session, no GUI) | `_find_aromatic_rings`, `_ring_planarity_rmsd`, `_ligand_charges`, `_parse_sdf_records`, `_distance_object_pairs`, `_hbond_dha_angle`, `_hbond_acceptor_angle` — results below |
| PyMOL command-line parser replayed on the plugin's own documented examples (`pymol/parsing.py` `parse_arg` + `prepare_call`, stubbed) | two confirmed CLI defects |
| PyMOL upstream source consulted for API semantics | `modules/pymol/{querying,importing,controlling,parsing,parser,internal}.py`, `modules/pymol/Qt/__init__.py`, `layer2/ObjectDist.cpp`, `layer3/{Selector,Executive}.cpp` (schrodinger/pymol-open-source, master) |
| Micro-benchmark of the pairwise distance gate | 11–26 ms for 40k–160k pairs — *not* the bottleneck |

Anything that depends on PyMOL's renderer, on Incentive-only APIs (`get_property_list`,
`get_property`, `distance mode=5/6/7`), or on live Qt event ordering is marked **unverified** and
reasoned about from the source only.

---

## 1. Confirmed bugs

### 1.1 `Clear` does not reset the stepper — the scene comes back on the next keypress
**L218–225, L2533–2534, L3323–3339, L3341–3351 · severity: high**

`_clear_all()` deletes PyMOL objects and empties `_created_objects`, and `do_clear()` empties the
table, the bookmarks and `sdf_records`. Neither touches `_stepper.poses`, `current_index`, `mode`,
`state_object` or `protein_sel`.

Consequence, by direct control-flow reading: after **Clear**, press **Next**. `_table_adjacent(1)`
sees `tw_pd.rowCount() == 0` (L3344) and falls back to `ci_next()` → `_stepper.next()` → `c =
len(self.poses)` is still non-zero → `_show_current()` rebuilds the shell, the carved surface and
every interaction object. The panel still reads "Ready - click Setup". The 500 ms sync timer
(L3582) has the same exposure: its only guard is `not _stepper.poses`, which is still false.

Fix: give `LigandStepper` a `reset()` that clears `poses`, `current_index`, `mode`, `state_object`,
`protein_sel`, `_in_compare`, `_cmp_indices`, and call it from `ci_clear()` so the CLI path is
covered too.

### 1.2 `_unbind_keys` permanently disables PyMOL's own arrow-key frame stepping
**L2205–2210 · severity: medium, user-visible after every Clear**

```python
cmd.set_key("right", lambda: None)
cmd.set_key("left",  lambda: None)
```

Verified against PyMOL's `internal._invoke_key`: it looks up `key_mappings[key]`, and returns
`False` (letting the built-in frame-advance run) **only when the mapping is absent or falsy**. A
`lambda: None` is a live mapping wrapped as `(fn, (), {})` — truthy — so `_invoke_key` returns
`True` and swallows the key. After `ci_clear()` the left/right arrows are dead for the rest of the
PyMOL session, including for structures that have nothing to do with PoseViewer.

Also note `cmd.set_key(key, None)` is *not* an unbind: `controlling.set_key` returns a decorator
when `fn is None`. The actual unbind is:

```python
cmd.key_mappings.pop("right", None)
cmd.key_mappings.pop("left", None)
```

Separately, `_bind_keys` (L2200) takes those keys over without warning and without saving what was
there.

### 1.3 Command-line arguments are never unquoted or path-expanded
**L2452–2462, L2464–2497, L2108–2118, L2549–2588, L2633–2660 · severity: medium**

PyMOL's parser hands the command's value **with its quote characters intact**; every built-in
command calls `creating.unquote()` and `cmd.exp_path()` itself (see `importing.load`, L735/L751 of
`importing.py`). PoseViewer calls neither. Replayed through the real `parse_arg` + `prepare_call`:

```
'ci_load_scores "/home/me/my poses.sdf"'  -> kwargs={'path': '"/home/me/my poses.sdf"'}
'ci_setup ligands="LIG1,LIG2,LIG3"'       -> kwargs={'ligands': '"LIG1,LIG2,LIG3"'}
```

So:

* `ci_load_scores "/path/with spaces.sdf"` → `open()` fails on the literal `"` → caught by the
  broad `except` at L2148 → "could not parse SDF", no scores.
* `ci_load_poses ~/poses.sdf` → `cmd.load` expands `~` internally and the **geometry loads**, but
  `_parse_sdf_records` uses bare `open(path)` and fails → geometry present, score table silently
  empty. A confusing half-success.
* `ci_setup ligands="LIG1,LIG2"` — the only syntax that survives the comma split at L2314 — yields
  the selection strings `"LIG1` and `LIG2"`, which PyMOL cannot parse.

Fix: at the top of every path-taking or selection-taking command,
`path = cmd.exp_path(unquote(path))` (`from pymol.creating import unquote`).

### 1.4 The `ci_setup` usage example in its own docstring raises an exception
**L2283 · severity: low, documentation**

```
ci_setup protein=chain A, ligands=LIG1,LIG2,LIG3
```
replayed through `prepare_call` with the real signature:
```
QuietException: Error: too many positional arguments for ci_setup
```
PyMOL splits on top-level commas, so `LIG2` lands in `mode` and `LIG3` overflows. The multi-ligand
feature at L2314 (`if "," in ligands`) is effectively unreachable from the command line until 1.3
is fixed.

### 1.5 `_parse_sdf_records` — three defects, one of them silently mis-attributes scores
**L2108–2150 · severity: medium–high (data integrity)**

Reproduced on a synthetic SDF:

```
input: 2 records; one SD value contains the substring "$$$$fake"
parsed: 3 records
  {'_name': 'mol-A', 'Ligand_ID': 12, 'minimizedAffinity': -9.37, 'Notes': 'line one', 'SMILES': 'CC(=O)O'}
  {'_name': 'fake'}                               <-- phantom record
  {'_name': 'mol-B', 'minimizedAffinity': -8.1}
```

1. **`content.split("$$$$")`** splits on the bare substring. The SDF delimiter is `$$$$` *at the
   start of a line*. A phantom record shifts every later record by one; in states mode
   `all_properties = (recs + [{}] * n)[:n]` (L2013) then pins the wrong scores to every subsequent
   pose. The `_sdf_mismatch` warning (L2011) fires only when the counts differ — a *balanced*
   corruption is silent, and the objects-mode name-matching path (L2028–2047) never warns at all.
   Fix: `re.split(r'(?m)^\$\$\$\$\s*$', content)`.
2. **Multi-line SD values are truncated to the first line** — `Notes` lost "line two". The regex
   `>\s*<([^>]+)>[^\n]*\n([^\n]*)` captures one line; SD data runs until a blank line.
3. **Identifiers are coerced to numbers**:
   `Ligand_ID "0012" → int 12` (leading zeros lost), `"1e3" → 1000.0`, `"nan" → float('nan')`.
   The mangled value is what reaches the table, the CSV export and the SD tags written by
   `_export_poses_sdf`. Only coerce when `str(int(raw)) == raw` / the round-trip is exact, or keep a
   raw copy alongside the numeric sort key.

### 1.6 Aromaticity perception collapses when the ligand has no explicit hydrogens
**L330–365, L446–516 · severity: medium (false interactions)**

`_find_aromatic_rings` has two filters: a planarity RMSD ≤ 0.15 Å, and "an sp3 carbon has >3
bonds". The second is a **no-op on any hydrogen-free model** — a ligand read out of a PDB complex,
which is the plugin's own auto-split path (L2217). Measured:

| ring (no explicit H) | plane RMSD | called aromatic? |
|---|---|---|
| cyclopentane, flat | 0.000 | **yes** |
| cyclopentane, envelope ±0.08 Å | 0.073 | **yes** |
| succinimide ring | 0.000 | **yes** |
| tetrahydrofuran | 0.000 | **yes** |
| cyclohexane, half-chair ±0.12 Å | 0.120 | **yes** |
| piperidine, flat | 0.000 | **yes** |

The same rings **with** explicit H are correctly rejected. Only a full chair (RMSD 0.25) is caught
by geometry alone. Downstream this invents pi-pi stacking, pi-cation and aromatic H-bonds.

Second-order effect, also measured: `lig_aromatic` is passed into `_ligand_charges`, where
L513–514 requires every heavy neighbour of a candidate amine N to be a *non-aromatic* carbon. An
exocyclic `-NH2` on a hydrogen-free cyclohexane therefore loses its charge:

```
charges with the ring judged aromatic  : (set(), set())
charges if the ring were non-aromatic  : ({6}, set())      # the N correctly cationic
```

So a benzylamine-type ammonium on a saturated ring stops making salt bridges and pi-cation
contacts. Mitigation: require an actual bond-order/aromatic flag where the format supplies one
(SDF/mol2), and when hydrogens are absent fall back to element composition plus a much tighter
planarity bound (≤ 0.05 Å) and a ring-bond-length test (aromatic C–C ≈ 1.39 Å; sp3 C–C ≈ 1.54 Å) —
that last one separates these cases cleanly and needs no hydrogens.

### 1.7 `_snapshot_ligands`' fallback is a correctness fallback, not a speed fallback
**L1309–1361 vs L585 and L1481–1483 · severity: medium (silent wrong answer)**

The docstring claims: *"Falls back to the plain union if copying fails, so a surprise here costs
speed rather than function."* It costs function.

Confirmed in `layer3/Selector.cpp`, `SelectorUpdateTableImpl` (L7138 ff.): with an explicit
`state >= 0`, any object whose state count is ≤ that state is **skipped from the selector table
entirely** — no clamping, no `static_singletons` (that setting is only consulted by
`SelectorCreateObjectMolecule`, i.e. `cmd.create`). The plugin's whole design around this is
correct and well reasoned. But both call sites hard-code `state=1`:

* L585 `cmd.select(_TMP, f"({prot_sel}) within {search} of ({snap_sel})", state=1)`
* L1481 `cmd.select(_OBJ_SHELL, f"byres (({protein_sel}) within {dist} of ({snap_sel}))", ..., state=1)`

If `cmd.create` throws for every ligand, `snap_sel` becomes the raw multi-state object (L1359–1360)
and `state=1` then pins the proximity test to **pose 1 forever**. The viewer keeps stepping and
keeps drawing pose-1 interactions. Make the fallback return the state it is valid for, or raise and
let the caller skip detection.

(The comment at L583–584 — "cmd.select evaluates proximity at the current global state" — is also
stale; the call passes `state=1` explicitly.)

### 1.8 GUI checkboxes are hard-coded, not read from `_stepper`
**L2904–2939, L2944–2952 vs L1547–1580 · severity: low–medium**

`sp_hba`, `sp_sd` and `cmb_cs` are initialised from `_stepper` (L2910, L2967, L2954). The
*checkboxes* are not — `_cb(l1, "Hydrogen bonds", "#ffd900")` takes the `checked=True` default.
Close the panel after unticking H-bonds (or set the flags from the CLI), reopen it, and the panel
claims H-bonds are on while `_stepper.show_hbonds` is `False`; no signal fires to resync, so the
first click *reduces* the discrepancy only by accident. Initialise each box from its `_stepper`
attribute, ideally from one `(checkbox, attr)` table that also drives `_tog` and `_group_tog`.

### 1.9 Neutral histidine tautomers counted as cations in salt bridges
**L669–670 vs L1384–1385 · severity: low–medium (false positives)**

```python
ppos = (pa.resn in ("ARG","LYS","HIS","HID","HIE","HIP") and pa.name in (...,"ND1","NE2"))
```

`HID`/`HIE` are, by construction, the *neutral* singly-protonated tautomers. The surface-colouring
code one file-section away gets this right (`_HIS_POS`, L1384: only `HIP`/`HSP`, or an explicit-H
model carrying both `HD1` and `HE2`). Reuse that predicate here. Bare `HIS` is genuinely ambiguous
and is at least defensible; `HID`/`HIE` are not. `HSD`/`HSE`/`HSP` (CHARMM) are missing from the
salt-bridge list entirely, and `_SURF_POS_STRONG`'s HIP is likewise absent from the pi-cation donor
set (L782, ARG CZ and LYS NZ only).

### 1.10 `_prepare_scene` destroys user session content and leaves a global setting changed
**L1228, L1242, L1252–1259 · severity: medium (data loss), by design but silent**

Three irreversible side effects on the user's session, none of them undone by `ci_clear`:

* **L1252–1259** every object of type `object:measurement` not created by PoseViewer is deleted.
  `cmd.get_type` returns `"object:measurement"` for **distance, angle *and* dihedral** objects
  (confirmed in `querying.get_type`'s documented return values), so a user's hand-made angle
  measurements go too. The rationale in the comment is reasonable; the silence is not. At minimum
  print what was removed; better, hide rather than delete, or ask once.
* **L1242** `cmd.hide("surface")` with no selection hides *every* surface in the session.
* **L1228** `cmd.set("auto_zoom", 0)` is global and never restored — it survives `ci_clear`, affects
  every other object the user loads afterwards, and is written into any `.pse` saved later. The
  plugin already has the right pattern for this in `_BASELINE_RENDER_SETTINGS` (L55–58); extend it
  to `auto_zoom` and restore on clear.

---

## 2. Risks and probable defects (not reproducible without PyMOL)

### 2.1 Choosing a complex object as "Protein" makes the ligand interact with itself
**L585, L3220–3239 · unverified, but the control flow is unambiguous**

`_populate_prot_combo` offers any object containing protein atoms *by bare name* (L3231). Pick
`1abc` rather than the default `polymer.protein` and `prot_sel` then matches the het records and
waters **in that same object** — including the ligand the poses were split from. The pairwise loop
(L646 ff.) then pairs ligand atoms against their own copies at d ≈ 0: every atom becomes an "ugly
clash" with itself, plus spurious halogen/salt/H-bond hits. Defend by intersecting internally:
`prot_sel_eff = f"({prot_sel}) and polymer.protein"`, or at least `and not ({lig_union})`.

### 2.2 Drawing is O(N²) in the number of interactions
**L1174–1182 · unverified timing**

`_add_pair_points` was carefully optimised to one `load_model` call (L912–937), and then the very
next loop undoes the benefit:

```python
for i, (obj_name, _p1, _p2) in enumerate(pending):
    cmd.distance(obj_name, f"{pts} and resi {i} and name L", f"{pts} and resi {i} and name P")
```

That is one PyMOL call *and two selection evaluations over the whole `_ci_pts` object* (2N atoms)
per pair — quadratic. It is invisible with H-bonds only (a handful of pairs) and becomes the
dominant cost the moment Contacts/Clashes are ticked, which routinely produces hundreds to
thousands of pairs. Either build one `chempy` object per interaction type and use `cmd.distance`
with `mode=1` over its bonds, or emit the dashes as a CGO. Worth measuring before rewriting.

By contrast the Python pairwise loop the comments worry about (L239–241) is *not* the bottleneck:
benchmarked at 11 ms for 40 500 pairs and 26 ms for 162 000 pairs on this machine, gate only. The
decision to avoid numpy for 3-vectors is sound.

### 2.3 Re-entrancy: a modal dialog lets the 500 ms timer fire inside a running update
**L1980–1985, L3732–3735 · unverified**

`_update`'s error path opens `QtWidgets.QMessageBox.warning(...)` (L1983). A modal box spins a
nested Qt event loop, so `win._sync_timer` fires *while `_update` is still on the stack* and can
re-enter `_stepper._show_current()` → `_update` → another dialog. The transient
`_ci_snap0/_ci_snap1` objects (never `_track`ed, L1338) are live at that moment and will also show
up in the timer's object-set diff. Guard with a module-level `_busy` flag checked at the top of
`_sync_if_external_change`, and prefer a non-modal status line to a `QMessageBox` for a per-pose
error.

More generally, every long operation runs on the GUI thread with no progress feedback and no
cancel: `setup_states` on a 1600-pose SDF builds the shell, carves an SES surface, and then
`_prefetch_all_properties` makes ~2 PyMOL calls per pose (L2153–2169, L2172–2193) — several
thousand round trips with the UI frozen. PyMOL's `cmd` API is not safe to drive from a worker
thread, so the realistic fix is chunking through `QTimer.singleShot(0, ...)` plus a progress label,
not threading.

### 2.4 Interaction geometry criteria are distance-only where they should be angular
**L661–676, L717–735, L779–809 · unverified against reference implementations**

The H-bond path is careful — PyMOL's chemistry-aware `mode=2`, then a real D–H···A filter, then an
sp2 acceptor lone-pair check. The geometric types are not held to the same standard:

* **Halogen bonds (L661–666)** fire on distance alone (≤ 3.5 Å, X ∈ Cl/Br/I vs O/N/S). A halogen
  bond is a σ-hole interaction and needs C–X···A ≈ 165–180°; without it, any chlorine passing near
  a backbone carbonyl is scored. PLIP uses 140° donor / 90° acceptor minima. This is the single
  largest false-positive source in the file.
* **Pi-pi face-to-face (L725–729)** checks centroid separation ≤ 4.8 Å and inter-normal angle
  ≤ 40°, with **no lateral-offset test**. Two parallel rings 4.5 Å apart but displaced 4 Å sideways
  are reported as stacked. Standard practice caps the offset (projection of the centroid vector onto
  the ring plane) at ~2.0 Å.
* **Pi-cation (L779–809)** is distance-only (≤ 6.0 Å centroid-to-cation) with no requirement that
  the cation sit above the ring plane; same offset test applies, and 6.0 Å is generous.
* **Edge-to-face (L730–735)** has no check that a ring C–H actually points at the partner ring.
* **No metal coordination type at all**, though Fe/Zn/Mg/Ca/Mn are in `VDW_RADII` (L94). For
  kinase and metalloenzyme work this is a conspicuous gap.

These are deliberate simplifications, but the module docstring says "All other interaction types are
detected geometrically", which rather understates how loose some of the geometry is. Worth either
tightening or documenting the false-positive rate per type.

### 2.5 `n_hb = cmd.distance(...)` is not a count
**L1892, L1911 · verified semantics, benign in practice**

`cmd.distance` returns the **average distance** (`querying.py` L505, `ObjectDist.cpp` L399 ff.;
`result` starts at −1.0 and becomes the mean, so a negative value is an error signal). The plugin's
own `_distance_object_pairs` docstring says exactly this (L942–943), yet `show_comparison` names the
result `n_hb` and tests `n_hb > 0`. It happens to work — zero contacts gives 0.0 or −1.0 — but the
name invites a later "if n_hb >= 2" that would be wrong. Rename to `avg_dist`, or test
`cmd.count_atoms` on the object.

### 2.6 `table_columns` silently drops any column whose name contains "rank"
**L1662–1670 · severity: low**

```python
return (["_name"] if ... else []) + [c for c in cols if c != "_name" and "rank" not in c.lower()]
```

A substring match. `Rank`, `CNN_rank`, `pose_rank` are dropped — defensible — but so is any tag that
merely contains the letters, and the drop propagates into `ci_export` (L2565) and the SD tags
written by `_export_poses_sdf` (L2601), so **exported data silently loses columns present in the
input file**. Match the whole token (`c.lower() in {"rank", "pose_rank", ...}`) or expose it as a
hide-column toggle rather than a hard filter.

### 2.7 PyQt6 enum access
**L2707, L2760, L3011 and ~30 others · unverified**

`pymol/Qt/__init__.py` patches `QtWidgets.QShortcut = QtGui.QShortcut` for Qt6, so that use (L3559)
is fine. It does **not** patch Qt's enum scoping. PyQt6 removed unscoped enum access, so
`QtCore.Qt.WA_DeleteOnClose`, `QtCore.Qt.AlignCenter`, `QtCore.Qt.UserRole`, `QtCore.Qt.DownArrow`
etc. raise `AttributeError` under PyQt6 (PySide6 still forwards them, and PyQt5 is fine). PyMOL
selects the binding at import time by probing PyQt5 → PySide2 → PyQt6 → PySide6, so the plugin works
on today's typical PyMOL builds and breaks on a PyQt6-only one. One-line mitigation at the top of
`_open_gui`: use the fully scoped names (`QtCore.Qt.WidgetAttribute.WA_DeleteOnClose`, …), which
PyQt5 also accepts.

### 2.8 `_ligand_charges` short-circuits on the first declared formal charge
**L466–469 · severity: low**

`if any(charges): return (...)` — one annotated atom disables inference for the *whole* molecule.
Reproduced: a model declaring `N+1` but leaving an un-annotated carboxylate returns
`({0}, set())` — the anion is lost. SDF/mol2 normally annotate everything, so this mostly matters
for hand-edited or partially-annotated files. Consider inferring the missing sign rather than
all-or-nothing.

---

## 3. Security: file and subprocess handling

**No subprocess, no `eval`/`exec`, no `__import__`, no `pickle`, no network.** Verified by grep over
the whole file. The attack surface is file I/O plus PyMOL selection strings.

### 3.1 `ci_load_poses` can execute arbitrary Python via `cmd.load`
**L2490 · severity: medium (user-initiated, local), confirmed against PyMOL source**

```python
cmd.load(path, obj, multiplex=0)
```

`path` is whatever the user typed into the "Poses (SDF)" field or passed on the command line; the
file dialog offers `All files (*)` (L3205). PyMOL's `importing.loadfunctions` (L1619–1625 of
`importing.py`):

```python
'py':  lambda filename, _self: _self.do("_ run %s" % filename),
'pml': lambda filename, _self: _self.do("_ @%s" % filename),
```

So `ci_load_poses /path/to/anything.py` **runs that file as Python**, and `.pml`/`.pym`/`.p1m`/
`.pim` run as PyMOL scripts (which can contain `python ... python end` blocks). A plausible path is
a shared "poses" directory or a mistyped/renamed download; this is not a remote exploit, but the
function's contract says SDF and nothing enforces it.

Fix (one line): `cmd.load(path, obj, format="sdf", multiplex=0)`, or reject extensions outside
`{.sdf, .sd, .mol, .mol2}` before loading.

### 3.2 Selection-string injection from file-derived residue identifiers
**L2247–2250 · severity: low**

```python
sel = f"({ligands_sel}) and {chain_part}resn {resn} and resi {resi}"
```

`resn`, `resi` and `chain` come from `cmd.iterate` over a file the user loaded. PDB's fixed columns
keep these tame, but mol2/SDF-derived or hand-edited records can carry spaces and PyMOL selection
operators, which would change *which atoms get extracted* (not execute anything). Quote the values
(`resn "{resn}"`) or build the selection with `cmd.select` on an index list.

The same shape appears throughout with the user-typed `protein`/`ligands` strings — but those are
selections by design, so that is intended behaviour, not injection.

### 3.3 Encoding and file-writing hygiene
**L2118, L2603, L2580, L3106 · severity: low**

* `_parse_sdf_records` (L2118): `open(path)` with no `encoding=` uses the locale default. A UTF-8
  SDF on a `cp1252` Windows PyMOL raises `UnicodeDecodeError`, which the broad `except` at L2148
  turns into "could not parse SDF" with no hint of the real cause. Use
  `encoding="utf-8", errors="replace"`.
* `_export_poses_sdf` (L2603): `open(path, "w")` with no `encoding=` or `newline=`. CSV export gets
  both right (L2580, L3106); the SDF path does not.
* `_export_poses_sdf` sanitises newlines in the title (L2616) but **not** in the SD tag values
  (L2624–2627). A property value containing a newline or a line reading `$$$$` produces a corrupt
  SDF. Sanitise `val` the same way.
* The CLI exports (`ci_export`, `ci_export_sdf`) overwrite without confirmation — acceptable for a
  CLI — but `_export_poses_sdf` opens the file *before* checking there is anything to write, so a
  failed export leaves a truncated or empty file where a good one used to be. Write to a temp file
  and rename, or check first.
* Only `OSError` is caught around the writers (L2584, L2657, L3110). A `UnicodeEncodeError` or a
  `csv.Error` escapes to PyMOL's console as a traceback.

### 3.4 Note on `.pse` sessions
The module header reasons carefully (L46–58) about a `.pse` silently overwriting render settings.
Worth stating explicitly in the README that a `.pse` is itself untrusted executable content as far
as settings go, and that PoseViewer *deletes* measurement objects it finds in one (§1.10).

---

## 4. UI and threading

Everything runs on the Qt/PyMOL main thread, which is the correct and only safe pattern for `cmd`.
The consequences are the ones listed in §2.3 (no progress, no cancel, modal-dialog re-entrancy).
Three smaller items:

* **Spinboxes recompute on every step.** `sp_hba.valueChanged` (L3410) calls `ci_update()` — a full
  re-detect and redraw — on each 5° click, and `sp_sd.valueChanged` (L3493) calls
  `rebuild_shell()`, i.e. a full SES recomputation, on each 0.5 Å click. Hold the mouse on the
  arrow and you queue a dozen. Debounce with a ~250 ms single-shot `QTimer`, or connect
  `editingFinished`.
* **A checkbox silently exits compare mode.** Any `_tog` handler calls `ci_update()` →
  `_show_current()` → `_cleanup_cmp()` (L1745), dropping out of compare while the table still shows
  two rows selected. Either refuse to leave compare mode or clear the selection too.
* **Ref-combo desync.** `_populate_ref_combo` (L3241–3260) falls back to index 0 — "(none)" — when
  `_stepper.ref_ligand` is not among the listed objects, while `_stepper.ref_ligand` keeps its
  value. The combo then misreports the state.
* **Dead parameter.** `_section(..., show_enable=False)` (L2861–2900) is documented ("e.g. Display
  group") but never used that way; all four call sites take the default.

---

## 5. Performance

What was measured here, and what the code already gets right:

* The Python pairwise loop is **fine** — 11–26 ms for 40k–162k pairs. The flat-scalar vector helpers
  and the single squared-distance gate (L642–658) are the right call; the comments justifying them
  (L239–241) are accurate.
* The documented `within`/state pathologies and their snapshot fix (L1309–1334) are **correct** and
  confirmed against `Selector.cpp`. This is the best-reasoned part of the file.

Remaining costs, roughly in order:

1. **The per-pair `cmd.distance` loop** (§2.2) — quadratic, dominant once clashes are on.
2. **`setup_objects` builds the shell twice.** L1599 calls `_create_shell(prot, all_ligs)` over
   *every* ligand — including a full carved SES — and L1601 `_show_current()` immediately rebuilds
   it for the current pose only (L1770–1777). Drop the first call, or pass the current pose's
   ligand list to it.
3. **`_prefetch_all_properties`** makes ~2–3 PyMOL round trips per pose with no SDF loaded
   (L1994–2000) and is re-run from the sync timer on any object or state-count change
   (L3615, L3639, L3671). On a 1600-pose session that is thousands of calls, repeatedly.
4. **`CLASH_DIST_MAX = 5.46 Å`** (L141, measured) is derived from Si's 2.10 Å radius. Correct and
   future-proof as the comment says, but it widens the gate ~24% beyond the 4.42 Å that C/N/O
   actually need. A per-element cutoff, or just excluding Si/B from the max, would narrow it.
5. `_register_colors()` runs 14 `set_color` calls on every `visualize()` (L1061). Trivial, but it
   belongs in setup.

---

## 6. Maintainability

The file is unusually well commented — most non-obvious decisions carry a rationale that explains
*why*, often with the measurement that motivated it. That is genuinely rare and worth preserving.
The structural problems are about size and testability:

* **`_open_gui` is 1047 lines** (L2692–3738) containing ~40 nested closures. Nothing inside it is
  reachable from a test or from the CLI. `_detect_interactions` is 314 lines, `visualize` 153,
  `_sync_if_external_change` 149, `ci_setup` 141.
* **No tests, no test seam.** The pure helpers (`_find_aromatic_rings`, `_ligand_charges`,
  `_parse_sdf_records`, `_hbond_*`, `_distance_object_pairs`) have no PyMOL dependency and are
  directly testable — this review drove all of them from synthetic inputs in a few lines. Splitting
  the file into `poseviewer/{geometry,chemistry,sdf,pymol_scene,gui}.py` and adding ~30 pytest cases
  over the pure half would catch §1.5 and §1.6 immediately. The plugin entry point can stay a thin
  shim for Plugin Manager installation.
* **73 `except` clauses, 29 of them `except Exception:` followed by `pass`.** Several hide real
  failures: L625–627 (ring detection failure → silently no pi interactions), L634–636 (charge
  inference failure → silently no salt bridges), L711–712, L794–795, L864–865 (water bridges),
  L1523–1524 (the entire surface block). A failing surface and a surface that legitimately has
  nothing to draw are indistinguishable to the user. At minimum log once per session at a
  `PoseViewer:` prefix.
* **Module-level `cmd.get` at import time** (L55–58). The reasoning is sound, but if any of those
  three `cmd.get` calls raises, the *import* fails and the plugin never registers. Wrap in a
  `try`/`except` with per-setting fallback.
* **No Python/PyMOL version guard.** The walrus at L2244 makes 3.8 the floor; on an older PyMOL the
  user gets a bare `SyntaxError` with no indication why.
* **Duplicated colour definitions.** `COLORS` (L64–80) and the GUI swatch hex strings (L2904–2939)
  are maintained separately and have already drifted: `ci_arom_hb` is `#4cd980` in `COLORS` vs
  `#4dd97f` in the GUI, `ci_pipi` `#4cbfff` vs `#4dc0ff`. Cosmetic (1/255), but it shows the
  duplication is live. Generate the hex from `COLORS`.
* **Stale comments** worth correcting now that they are known wrong: L583–584 (claims the select
  uses the current global state; it passes `state=1`), L1332–1333 (the fallback claim, §1.7).
* `_OBJ_SNAP*` objects are never `_track`ed (L1338), so a leak survives `_clear_all`.

---

## 7. Suggested priority

| # | Item | Effort |
|---|---|---|
| 1 | §1.1 Clear doesn't reset the stepper | 10 lines |
| 2 | §1.2 restore the arrow keys via `cmd.key_mappings.pop` | 2 lines |
| 3 | §3.1 `format="sdf"` on `cmd.load` | 1 line |
| 4 | §1.5 SDF delimiter regex, multi-line values, identifier coercion | ~20 lines |
| 5 | §1.3 `unquote` + `exp_path` on every path/selection argument | ~10 lines |
| 6 | §1.6 ring-bond-length aromaticity test for H-free ligands | ~15 lines |
| 7 | §1.9 reuse the `_HIS_POS` predicate for salt bridges | 3 lines |
| 8 | §1.10 restore `auto_zoom`; warn before deleting measurement objects | ~10 lines |
| 9 | §2.1 intersect `prot_sel` with `polymer.protein` | 1 line |
| 10 | §1.7 make the snapshot fallback fail loudly | ~5 lines |
| 11 | §2.4 angular criteria for halogen bonds; offset test for pi-pi / pi-cation | ~40 lines |
| 12 | §2.2 replace the per-pair `cmd.distance` loop | ~30 lines, measure first |
| 13 | §1.8 initialise checkboxes from `_stepper` via one table | ~20 lines |
| 14 | §6 split the module and add tests for the pure half | larger |

Items 1–3 and 9 are one-sitting fixes with the best return. Item 6 changes scientific output and
should be validated against a handful of known complexes before release.
