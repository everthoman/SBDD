"""
PickPocket - PyMOL Plugin
=========================
Filter docking poses by subpocket occupancy using fpocket alpha spheres.

Three ways to load pockets
--------------------------
1. Run fpocket from within the plugin (Run fpocket section) — runs fpocket on
   any loaded PyMOL protein or an external PDB file, loads individual
   pocket*_vert.pqr files as pocket{n}_spheres objects with full sphere radii.
2. Detect from PyMOL — finds pocket{n}_spheres objects already loaded by
   Pynacea or by this plugin, or falls back to resn STP atoms from the
   fpocket-generated *_out.pdb loaded via the *.pml script.
3. Load *_out/ directory — reads the combined *_out.pdb, pocket*_vert.pqr
   files (for radii) and *_info.txt (for druggability scores).

Filtering
---------
A pose passes if at least one heavy atom falls inside any selected alpha
sphere: distance(atom, sphere_center) < sphere_radius + tolerance.
When no sphere radii are available (resn STP fallback) a fixed distance
cutoff is used instead.

Installation
------------
  Plugin > Plugin Manager > Install New Plugin > choose this file
  — or —
  run /path/to/PickPocket.py   then   pp_gui

Authors: Evert J. Homan, PhD; Claude (Anthropic)
Date:    2026-05-13
Version: 1.0
License: MIT
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from pymol import cmd

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_POCKET_HEX = [
    "#4dc0ff", "#ffd900", "#ff6633", "#33cc33", "#e633e6",
    "#9933e6", "#ff4444", "#4dd97f", "#ff99cc", "#aaccff",
]
_POCKET_PYMOL = [
    "cyan", "yellow", "orange", "green", "magenta",
    "purple", "red", "lime", "salmon", "slate",
]

_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class Pocket:
    def __init__(self, pid: int):
        self.pid = pid
        self.name = f"pocket{pid}"
        self.centers: List[Tuple[float, float, float]] = []
        self.radii: List[float] = []       # empty when loaded from *_out.pdb
        self.score: Optional[float] = None
        self.drug_score: Optional[float] = None
        self.info: Dict = {}               # full fpocket metric dict

    @property
    def n_spheres(self) -> int:
        return len(self.centers)

    def spheres_with_radius(
        self, default_r: float
    ) -> List[Tuple[float, float, float, float]]:
        out = []
        for i, (x, y, z) in enumerate(self.centers):
            r = self.radii[i] if i < len(self.radii) else default_r
            out.append((x, y, z, r))
        return out


class _State:
    def __init__(self):
        self.pockets: List[Pocket] = []
        self.has_radii: bool = False
        self.out_dir: Optional[str] = None
        self.poses: List[Tuple[str, int]] = []
        self.sdf_records: List[Dict] = []
        self.all_properties: List[Dict] = []   # one dict per pose, SDF-matched
        self.results: List[Optional[bool]] = []
        self.protein_sel: str = "polymer.protein"
        self.poses_sel: str = "organic"
        self.cutoff: float = 3.0
        self.tolerance: float = 0.0
        self.custom_spheres: Optional[List[Tuple[float,float,float,float]]] = None
        self._created_objs: set = set()


_st = _State()
_gui_window = None

# ---------------------------------------------------------------------------
# fpocket info parser  (shared with Pynacea's _parse_fpocket_info)
# ---------------------------------------------------------------------------

_FPOCKET_COL_MAP = {
    "score":                                 "Score",
    "druggability score":                    "Druggability Score",
    "number of alpha spheres":               "Number of Alpha Spheres",
    "total sasa":                            "Total SASA",
    "polar sasa":                            "Polar SASA",
    "apolar sasa":                           "Apolar SASA",
    "volume":                                "Volume",
    "mean local hydrophobic density":        "Mean local hydrophobic density",
    "mean alpha sphere radius":              "Mean alpha sphere radius",
    "mean alp. sph. solvent access":         "Mean alp. sph. solvent access",
    "apolar alpha sphere proportion":        "Apolar alpha sphere proportion",
    "hydrophobicity score":                  "Hydrophobicity score",
    "volume score":                          "Volume score",
    "polarity score":                        "Polarity score",
    "charge score":                          "Charge score",
    "proportion of polar atoms":             "Proportion of polar atoms",
    "alpha sphere density":                  "Alpha sphere density",
    "cent. of mass - alpha sphere max dist": "CoM-AlphaSphere max dist",
    "flexibility":                           "Flexibility",
}

_FPOCKET_DISPLAY_COLS = ["Pocket", "Score", "Druggability Score",
                          "Number of Alpha Spheres", "Volume"]


def _parse_fpocket_info(info_path: str) -> Dict[int, Dict]:
    """Parse *_info.txt → {pocket_id: {col: value, ...}}"""
    pockets: Dict[int, Dict] = {}
    cur: Optional[Dict] = None
    cur_id: Optional[int] = None
    try:
        with open(info_path) as fh:
            for line in fh:
                line = line.strip()
                m = re.match(r"^Pocket\s+(\d+)\s*:", line, re.IGNORECASE)
                if m:
                    if cur_id is not None:
                        pockets[cur_id] = cur
                    cur_id = int(m.group(1))
                    cur = {}
                    continue
                if cur is not None and ":" in line:
                    raw_k, _, raw_v = line.partition(":")
                    col = _FPOCKET_COL_MAP.get(raw_k.strip().lower())
                    if col:
                        try:
                            cur[col] = float(raw_v.strip())
                        except ValueError:
                            pass
        if cur_id is not None:
            pockets[cur_id] = cur
    except Exception as e:
        print(f"PickPocket: info parse error: {e}")
    return pockets


# ---------------------------------------------------------------------------
# PQR radius parser
# ---------------------------------------------------------------------------

def _parse_vert_pqr(path: str) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    centers: List[Tuple[float, float, float]] = []
    radii: List[float] = []
    try:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                parts = line.split()
                # fpocket PQR (no chain): ATOM serial name resname resseq x y z charge radius
                if len(parts) < 10:
                    continue
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                r = float(parts[9])
                centers.append((x, y, z))
                radii.append(r)
    except Exception as e:
        print(f"PickPocket: PQR read error {path}: {e}")
    return centers, radii


# ---------------------------------------------------------------------------
# fpocket pocket loading helper (shared by run & load-dir paths)
# ---------------------------------------------------------------------------

def _load_pocket_pqr(pid: int, vert_path: str) -> str:
    """Load pocket{n}_vert.pqr into PyMOL as 'pocket{n}_spheres'.

    Returns the object name, or '' on failure.
    """
    obj = f"pocket{pid}_spheres"
    if obj in cmd.get_names("objects"):
        cmd.delete(obj)
    try:
        cmd.load(vert_path, obj)
        cmd.hide("everything", obj)
        cmd.show("spheres", obj)
        cmd.set("sphere_scale", 0.3, obj)
        cmd.set("sphere_transparency", 0.1, obj)
        cmd.color(_POCKET_PYMOL[(pid - 1) % len(_POCKET_PYMOL)], obj)
        _st._created_objs.add(obj)
        return obj
    except Exception as e:
        print(f"PickPocket: load PQR error {vert_path}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Pocket discovery
# ---------------------------------------------------------------------------

def _detect_from_pymol() -> bool:
    """Detect pockets from whatever is already loaded in PyMOL.

    Priority 1: pocket{n}_spheres objects (loaded from PQR — have real radii
    in atom.b).  Created by Pynacea or by this plugin's Run fpocket path.

    Priority 2: resn STP atoms in any object (combined *_out.pdb loaded via
    fpocket's *.pml script — no radii, fixed-cutoff mode).
    """
    _st.pockets.clear()
    _st.has_radii = False

    # --- Priority 1: pocket{n}_spheres ---
    sph_objs: Dict[int, str] = {}
    for name in cmd.get_names("objects"):
        m = re.match(r"^pocket(\d+)_spheres$", name)
        if m:
            sph_objs[int(m.group(1))] = name

    if sph_objs:
        for pid in sorted(sph_objs.keys()):
            p = Pocket(pid)
            try:
                model = cmd.get_model(sph_objs[pid])
                for atom in model.atom:
                    p.centers.append(tuple(atom.coord))
                    # PyMOL stores the PQR radius column in atom.b
                    p.radii.append(atom.b if atom.b > 0.1 else 1.5)
            except Exception:
                pass
            _st.pockets.append(p)
        _st.has_radii = True
        return bool(_st.pockets)

    # --- Priority 2: resn STP (combined *_out.pdb via fpocket *.pml) ---
    # Use a plain list to avoid multi-statement exec issues in iterate_state.
    coords_list: List[Tuple] = []
    try:
        cmd.iterate_state(
            1, "resn STP",
            "coords_list.append((int(resi), x, y, z))",
            space={"coords_list": coords_list},
        )
    except Exception as e:
        print(f"PickPocket: iterate_state error: {e}")
        return False

    if not coords_list:
        return False

    pocket_map: Dict[int, List] = {}
    for pid, x, y, z in coords_list:
        pocket_map.setdefault(pid, []).append((x, y, z))

    for pid in sorted(pocket_map.keys()):
        p = Pocket(pid)
        p.centers = pocket_map[pid]
        _st.pockets.append(p)
    return bool(_st.pockets)


def _load_from_dir(out_dir: str) -> bool:
    """Load pockets from an fpocket *_out/ directory.

    Loads *_out.pdb into PyMOL if not already present, reads sphere radii
    from pocket*_vert.pqr, and reads scores from *_info.txt.
    """
    _st.pockets.clear()
    _st.has_radii = False
    _st.out_dir = out_dir

    # Locate key files
    out_pdb = info_txt = None
    for fname in os.listdir(out_dir):
        if fname.endswith("_out.pdb"):
            out_pdb = os.path.join(out_dir, fname)
        if fname.endswith("_info.txt"):
            info_txt = os.path.join(out_dir, fname)

    # Load combined PDB (visualises all pockets coloured by resi)
    if out_pdb and os.path.isfile(out_pdb):
        obj = os.path.splitext(os.path.basename(out_pdb))[0]
        if obj not in cmd.get_names("objects"):
            cmd.load(out_pdb, obj)
            cmd.hide("everything", obj)
            cmd.show("spheres", f"({obj}) and resn STP")
            cmd.set("sphere_scale", 0.3, f"({obj}) and resn STP")
            cmd.set("sphere_transparency", 0.4, f"({obj}) and resn STP")
            last = [0]
            cmd.iterate(f"({obj}) and resn STP",
                        "last[0] = max(last[0], int(resi))",
                        space={"last": last})
            for i in range(1, last[0] + 1):
                color = _POCKET_PYMOL[(i - 1) % len(_POCKET_PYMOL)]
                cmd.color(color, f"({obj}) and resn STP and resi {i}")

    if not _detect_from_pymol():
        return False

    # Enrich with PQR radii
    pockets_dir = os.path.join(out_dir, "pockets")
    if not os.path.isdir(pockets_dir):
        pockets_dir = out_dir
    pocket_map = {p.pid: p for p in _st.pockets}
    pqr_found = False
    for fname in sorted(os.listdir(pockets_dir)):
        m = re.match(r"pocket(\d+)_vert\.pqr$", fname)
        if not m:
            continue
        pid = int(m.group(1))
        if pid not in pocket_map:
            continue
        _, radii = _parse_vert_pqr(os.path.join(pockets_dir, fname))
        if radii:
            pocket_map[pid].radii = radii
            pqr_found = True
    _st.has_radii = pqr_found

    # Scores from *_info.txt
    if info_txt and os.path.isfile(info_txt):
        info = _parse_fpocket_info(info_txt)
        for p in _st.pockets:
            rec = info.get(p.pid, {})
            p.info = rec
            p.score = rec.get("Score")
            p.drug_score = rec.get("Druggability Score")

    return True


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _find_stp_object() -> Optional[str]:
    for name in cmd.get_names("objects"):
        try:
            if cmd.count_atoms(f"({name}) and resn STP") > 0:
                return name
        except Exception:
            pass
    return None


def _highlight_pockets(enabled_ids: set):
    """Colour selected pockets orange/opaque, dim others."""
    try:
        # pocket{n}_spheres objects
        for p in _st.pockets:
            obj = f"pocket{p.pid}_spheres"
            if obj in cmd.get_names("objects"):
                if p.pid in enabled_ids:
                    cmd.color("orange", obj)
                    cmd.set("sphere_transparency", 0.1, obj)
                else:
                    color = _POCKET_PYMOL[(p.pid - 1) % len(_POCKET_PYMOL)]
                    cmd.color(color, obj)
                    cmd.set("sphere_transparency", 0.65, obj)

        # resn STP in combined *_out.pdb
        stp_obj = _find_stp_object()
        if stp_obj:
            for p in _st.pockets:
                sel = f"({stp_obj}) and resn STP and resi {p.pid}"
                if p.pid in enabled_ids:
                    cmd.color("orange", sel)
                    cmd.set("sphere_transparency", 0.1, sel)
                else:
                    color = _POCKET_PYMOL[(p.pid - 1) % len(_POCKET_PYMOL)]
                    cmd.color(color, sel)
                    cmd.set("sphere_transparency", 0.65, sel)
    except Exception as e:
        print(f"PickPocket: highlight error: {e}")


# ---------------------------------------------------------------------------
# Pose discovery & SDF parsing
# ---------------------------------------------------------------------------

def _discover_poses(poses_sel: str, protein_sel: str) -> List[Tuple[str, int]]:
    poses = []
    ignore = _st._created_objs | {_find_stp_object() or ""}
    for name in cmd.get_names("objects"):
        if name in ignore:
            continue
        try:
            if (cmd.count_atoms(f"({name}) and ({poses_sel})") > 0 and
                    cmd.count_atoms(f"({name}) and ({protein_sel})") == 0):
                n_st = cmd.count_states(name)
                for st in range(1, n_st + 1):
                    poses.append((name, st))
        except Exception:
            pass
    return poses


def _parse_sdf_records(path: str) -> List[Dict]:
    records = []
    try:
        with open(path) as fh:
            content = fh.read()
        for block in content.split("$$$$"):
            block = block.strip()
            if not block:
                continue
            props: Dict = {}
            lines = block.splitlines()
            if lines and lines[0].strip():
                props["_name"] = lines[0].strip()
            for m in re.finditer(r">\s*<([^>]+)>\s*\n([^\n]*)", block):
                key = m.group(1).strip()
                raw = m.group(2).strip()
                try:
                    props[key] = int(raw)
                except (ValueError, OverflowError):
                    try:
                        props[key] = float(raw)
                    except (ValueError, OverflowError):
                        props[key] = raw
            records.append(props)
    except Exception as e:
        print(f"PickPocket: SDF parse error: {e}")
    return records


# ---------------------------------------------------------------------------
# SDF-to-pose property matching  (mirrors PoseViewer._prefetch_all_properties)
# ---------------------------------------------------------------------------

def _build_all_properties() -> List[Dict]:
    """Return one property dict per pose, aligned to _st.poses.

    States mode (single object, multiple states): sequential 1-to-1 with SDF.
    Objects mode (multiple objects): match by SDF molecule title (_name field),
    fall back to sequential if no names match.
    """
    from collections import defaultdict
    n = len(_st.poses)
    if not _st.sdf_records:
        return [{}] * n

    unique_objs = list(dict.fromkeys(obj for obj, _ in _st.poses))
    if len(unique_objs) == 1:
        # States mode — sequential
        recs = list(_st.sdf_records)
        return (recs + [{}] * n)[:n]

    # Objects mode — try name-based matching first
    by_name: Dict = defaultdict(list)
    for rec in _st.sdf_records:
        by_name[rec.get("_name", "")].append(rec)

    obj_names = {obj for obj, _ in _st.poses}
    if obj_names & set(by_name.keys()):
        counters: Dict = defaultdict(int)
        result = []
        for obj, _state in _st.poses:
            recs = by_name.get(obj, [])
            i = counters[obj]
            result.append(recs[i] if i < len(recs) else {"_name": obj})
            counters[obj] += 1
        return result

    # Sequential fallback
    return (list(_st.sdf_records) + [{}] * n)[:n]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def _pose_in_subpocket(
    obj: str,
    state: int,
    selected: List[Pocket],
    cutoff: float,
    tolerance: float,
) -> bool:
    try:
        model = cmd.get_model(f"({obj}) and not elem H", state=state)
    except Exception:
        return False
    for atom in model.atom:
        ax, ay, az = atom.coord
        for pocket in selected:
            for sx, sy, sz, sr in pocket.spheres_with_radius(cutoff):
                d = math.sqrt((ax - sx) ** 2 + (ay - sy) ** 2 + (az - sz) ** 2)
                if d < sr + tolerance:
                    return True
    return False


def pp_filter(enabled_ids: Optional[set] = None) -> Dict:
    """Run the subpocket filter.

USAGE
    pp_filter
    """
    if _st.custom_spheres is not None:
        # Use manually selected spheres (overrides pocket checkboxes)
        proxy = Pocket(0)
        proxy.centers = [(x, y, z) for x, y, z, r in _st.custom_spheres]
        proxy.radii   = [r          for x, y, z, r in _st.custom_spheres]
        selected = [proxy]
    else:
        if enabled_ids is None:
            enabled_ids = {p.pid for p in _st.pockets}
        selected = [p for p in _st.pockets if p.pid in enabled_ids]
    results = [
        _pose_in_subpocket(obj, st, selected, _st.cutoff, _st.tolerance)
        for obj, st in _st.poses
    ]
    _st.results = results
    n = sum(results)
    print(f"PickPocket: {n}/{len(results)} poses occupy the subpocket.")
    return {
        "pass": [p for p, r in zip(_st.poses, results) if r],
        "fail": [p for p, r in zip(_st.poses, results) if not r],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def pp_load(path=""):
    """Load fpocket *_out/ directory.  Usage: pp_load /path/to/protein_out/"""
    path = path.strip()
    if not path or not os.path.isdir(path):
        print("PickPocket: provide a valid *_out/ directory.")
        return
    ok = _load_from_dir(path)
    note = " (radii from PQR)" if _st.has_radii else " (cutoff mode)"
    print(f"PickPocket: {'loaded' if ok else 'failed'} {len(_st.pockets)} pockets{note}.")


def pp_detect():
    """Detect pockets from PyMOL (pocket{n}_spheres or resn STP).  Usage: pp_detect"""
    ok = _detect_from_pymol()
    print(f"PickPocket: {'detected' if ok else 'no'} {len(_st.pockets)} pockets.")


def pp_setup(protein="polymer.protein", poses="organic"):
    """Discover docking poses.  Usage: pp_setup [protein [, poses]]"""
    _st.protein_sel = protein
    _st.poses_sel   = poses
    _st.poses = _discover_poses(poses, protein)
    _st.results = [None] * len(_st.poses)
    print(f"PickPocket: {len(_st.poses)} pose(s).")


def pp_gui():
    """Open the PickPocket GUI."""
    _open_gui()


cmd.extend("pp_load",   pp_load)
cmd.extend("pp_detect", pp_detect)
cmd.extend("pp_setup",  pp_setup)
cmd.extend("pp_filter", pp_filter)
cmd.extend("pp_gui",    pp_gui)


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------

def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("PickPocket", _open_gui)


def _set_gui_none():
    global _gui_window
    _gui_window = None


# ── fpocket background worker ────────────────────────────────────────────────

def _make_worker_class():
    """Build FpocketWorker using Qt signals (deferred to avoid import-time Qt dep)."""
    from pymol.Qt import QtCore

    class FpocketWorker(QtCore.QThread):
        log_line = QtCore.Signal(str)
        finished = QtCore.Signal(str, str)   # out_dir, stem
        failed   = QtCore.Signal(str)

        def __init__(self, fpocket_bin: str, pdb_path: str):
            super().__init__()
            self._bin  = fpocket_bin
            self._pdb  = pdb_path

        def run(self):
            from pathlib import Path
            try:
                stem    = Path(self._pdb).stem
                out_dir = Path(self._pdb).parent / f"{stem}_out"
                if out_dir.exists():
                    shutil.rmtree(str(out_dir))
                args = [self._bin, "-f", self._pdb]
                if _IS_WINDOWS:
                    # Run via WSL
                    def _to_wsl(p):
                        p = os.path.abspath(p).replace("\\", "/")
                        if len(p) >= 2 and p[1] == ":":
                            return f"/mnt/{p[0].lower()}{p[2:]}"
                        return p
                    args = ["wsl.exe", self._bin, "-f", _to_wsl(self._pdb)]
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                for line in proc.stdout:
                    self.log_line.emit(line.rstrip())
                proc.wait()
                if proc.returncode == 0 and out_dir.exists():
                    self.finished.emit(str(out_dir), stem)
                else:
                    self.failed.emit(
                        f"fpocket exited with code {proc.returncode}")
            except FileNotFoundError:
                self.failed.emit(f"fpocket not found: {self._bin}")
            except Exception as exc:
                self.failed.emit(str(exc))

    return FpocketWorker


def _open_gui():
    global _gui_window

    if _gui_window is not None:
        try:
            _gui_window.raise_()
            _gui_window.activateWindow()
            return
        except RuntimeError:
            _gui_window = None

    from pymol.Qt import QtWidgets, QtCore, QtGui

    FpocketWorker = _make_worker_class()

    win = QtWidgets.QWidget()
    _gui_window = win
    win.setWindowTitle("PickPocket")
    win.setMinimumWidth(480)
    win.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    win.destroyed.connect(_set_gui_none)

    root = QtWidgets.QVBoxLayout(win)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(5)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    root.addWidget(splitter)

    top_w = QtWidgets.QWidget()
    top_l = QtWidgets.QVBoxLayout(top_w)
    top_l.setContentsMargins(0, 0, 0, 0)
    top_l.setSpacing(4)
    splitter.addWidget(top_w)

    bot_w = QtWidgets.QWidget()
    bot_l = QtWidgets.QVBoxLayout(bot_w)
    bot_l.setContentsMargins(0, 0, 0, 0)
    bot_l.setSpacing(4)
    splitter.addWidget(bot_w)

    # ── collapsible section helper ────────────────────────────────────────────
    def _section(title, expanded=True):
        g = QtWidgets.QGroupBox()
        gl = QtWidgets.QVBoxLayout(g)
        gl.setContentsMargins(6, 4, 6, 6)
        gl.setSpacing(2)
        hl = QtWidgets.QHBoxLayout()
        hdr = QtWidgets.QLabel(title)
        f = hdr.font(); f.setBold(True); hdr.setFont(f)
        hl.addWidget(hdr); hl.addStretch()
        btn = QtWidgets.QToolButton()
        btn.setCheckable(True); btn.setChecked(expanded)
        btn.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        btn.setAutoRaise(True)
        hl.addWidget(btn)
        gl.addLayout(hl)
        body = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(body)
        bl.setContentsMargins(0, 2, 0, 0)
        bl.setSpacing(3)
        body.setVisible(expanded)
        gl.addWidget(body)
        btn.toggled.connect(lambda on: (
            body.setVisible(on),
            btn.setArrowType(
                QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)))
        return g, body, bl

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Run fpocket  (collapsible, collapsed by default)
    # ══════════════════════════════════════════════════════════════════════════
    g_run, _run_body, l_run = _section("Run fpocket", expanded=True)

    # fpocket binary
    hl_fp = QtWidgets.QHBoxLayout()
    hl_fp.addWidget(QtWidgets.QLabel("fpocket:"))
    e_fpbin = QtWidgets.QLineEdit(shutil.which("fpocket") or "fpocket")
    b_fpbin = QtWidgets.QPushButton("…"); b_fpbin.setFixedWidth(26)
    hl_fp.addWidget(e_fpbin, 1); hl_fp.addWidget(b_fpbin)
    l_run.addLayout(hl_fp)

    # Protein source toggle
    hl_mode = QtWidgets.QHBoxLayout()
    rb_obj  = QtWidgets.QRadioButton("PyMOL object")
    rb_file = QtWidgets.QRadioButton("PDB file")
    rb_obj.setChecked(True)
    hl_mode.addWidget(rb_obj); hl_mode.addWidget(rb_file); hl_mode.addStretch()
    l_run.addLayout(hl_mode)

    prot_stack = QtWidgets.QStackedWidget()

    # page 0 — PyMOL object
    obj_page = QtWidgets.QWidget()
    ol = QtWidgets.QHBoxLayout(obj_page); ol.setContentsMargins(0,0,0,0)
    cb_prot = QtWidgets.QComboBox(); cb_prot.setMinimumWidth(160)
    b_refresh = QtWidgets.QPushButton("Refresh"); b_refresh.setFixedWidth(60)
    ol.addWidget(cb_prot, 1); ol.addWidget(b_refresh)
    prot_stack.addWidget(obj_page)

    # page 1 — PDB file
    file_page = QtWidgets.QWidget()
    fl = QtWidgets.QHBoxLayout(file_page); fl.setContentsMargins(0,0,0,0)
    e_pdbfile = QtWidgets.QLineEdit(); e_pdbfile.setPlaceholderText("path/to/protein.pdb")
    b_pdbfile = QtWidgets.QPushButton("…"); b_pdbfile.setFixedWidth(26)
    fl.addWidget(e_pdbfile, 1); fl.addWidget(b_pdbfile)
    prot_stack.addWidget(file_page)

    rb_obj.toggled.connect(lambda on: prot_stack.setCurrentIndex(0 if on else 1))
    l_run.addWidget(prot_stack)

    hl_run_btn = QtWidgets.QHBoxLayout()
    b_run_fp = QtWidgets.QPushButton("Run fpocket")
    f_run_fp = b_run_fp.font(); f_run_fp.setBold(True); b_run_fp.setFont(f_run_fp)
    hl_run_btn.addWidget(b_run_fp); hl_run_btn.addStretch()
    l_run.addLayout(hl_run_btn)

    pb_run = QtWidgets.QProgressBar()
    pb_run.setRange(0, 0); pb_run.setVisible(False)
    l_run.addWidget(pb_run)

    log_run = QtWidgets.QPlainTextEdit()
    log_run.setReadOnly(True)
    f_log = QtGui.QFont("Monospace"); f_log.setStyleHint(QtGui.QFont.Monospace)
    f_log.setPointSize(8); log_run.setFont(f_log)
    log_run.setMaximumBlockCount(2000)
    log_run.setMaximumHeight(100)
    l_run.addWidget(log_run)

    top_l.addWidget(g_run)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Input (detect / load from dir / poses)
    # ══════════════════════════════════════════════════════════════════════════
    g_in = QtWidgets.QGroupBox("Input")
    l_in = QtWidgets.QGridLayout(g_in)

    l_in.addWidget(QtWidgets.QLabel("fpocket *_out/:"), 0, 0)
    hl_dir = QtWidgets.QHBoxLayout()
    e_dir = QtWidgets.QLineEdit()
    e_dir.setPlaceholderText("optional — for radii & scores")
    b_dir = QtWidgets.QPushButton("…"); b_dir.setFixedWidth(26)
    hl_dir.addWidget(e_dir); hl_dir.addWidget(b_dir)
    l_in.addLayout(hl_dir, 0, 1)

    l_in.addWidget(QtWidgets.QLabel("Protein:"), 1, 0)
    e_prot = QtWidgets.QLineEdit("polymer.protein")
    l_in.addWidget(e_prot, 1, 1)

    l_in.addWidget(QtWidgets.QLabel("Poses:"), 2, 0)
    e_poses = QtWidgets.QLineEdit("organic")
    l_in.addWidget(e_poses, 2, 1)

    l_in.addWidget(QtWidgets.QLabel("Scores SDF:"), 3, 0)
    hl_sdf = QtWidgets.QHBoxLayout()
    e_sdf = QtWidgets.QLineEdit(); e_sdf.setPlaceholderText("optional")
    b_sdf = QtWidgets.QPushButton("…"); b_sdf.setFixedWidth(26)
    hl_sdf.addWidget(e_sdf); hl_sdf.addWidget(b_sdf)
    l_in.addLayout(hl_sdf, 3, 1)

    hl_in_btns = QtWidgets.QHBoxLayout()
    b_detect = QtWidgets.QPushButton("Detect from PyMOL")
    b_detect.setToolTip(
        "Finds pocket{n}_spheres objects (created by 'Run fpocket' or Pynacea),\n"
        "or resn STP atoms from an fpocket *_out.pdb loaded via *.pml.\n\n"
        "If you haven't run fpocket yet, use the 'Run fpocket' section above.")
    b_load   = QtWidgets.QPushButton("Load *_out/ dir")
    b_load.setToolTip(
        "Load an existing fpocket *_out/ directory directly\n"
        "(reads *_out.pdb, pocket*_vert.pqr radii, and *_info.txt scores).")
    b_setup_poses = QtWidgets.QPushButton("Setup Poses")
    b_setup_poses.setToolTip(
        "Discover docking poses using the current Protein/Poses/SDF fields.\n"
        "Use this after changing the SDF path without re-running fpocket.")
    b_clear  = QtWidgets.QPushButton("Clear")
    hl_in_btns.addWidget(b_detect); hl_in_btns.addWidget(b_load)
    hl_in_btns.addWidget(b_setup_poses); hl_in_btns.addWidget(b_clear)
    l_in.addLayout(hl_in_btns, 4, 0, 1, 2)

    lbl_in_status = QtWidgets.QLabel("")
    lbl_in_status.setStyleSheet("color: grey; font-style: italic;")
    l_in.addWidget(lbl_in_status, 5, 0, 1, 2)
    top_l.addWidget(g_in)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Pockets list
    # ══════════════════════════════════════════════════════════════════════════
    g_pk, _pk_body, l_pk = _section("Pockets — select to define subpocket",
                                     expanded=True)

    _PK_INFO_COLS = list(_FPOCKET_COL_MAP.values())

    tw_pockets = QtWidgets.QTableWidget(0, 2 + len(_PK_INFO_COLS))
    tw_pockets.setHorizontalHeaderLabels(["", "Pocket"] + _PK_INFO_COLS)
    tw_pockets.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    tw_pockets.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    tw_pockets.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    tw_pockets.setMinimumHeight(90); tw_pockets.setMaximumHeight(220)
    tw_pockets.horizontalHeader().setStretchLastSection(False)
    tw_pockets.horizontalHeader().setSectionsMovable(True)
    tw_pockets.setSortingEnabled(True)
    tw_pockets.verticalHeader().setDefaultSectionSize(20)
    tw_pockets.setAlternatingRowColors(True)
    tw_pockets.setColumnWidth(0, 28)
    l_pk.addWidget(tw_pockets)

    hl_pk = QtWidgets.QHBoxLayout()
    b_all  = QtWidgets.QPushButton("All")
    b_none = QtWidgets.QPushButton("None")
    lbl_nsph = QtWidgets.QLabel("0 spheres")
    hl_pk.addWidget(b_all); hl_pk.addWidget(b_none)
    hl_pk.addStretch(); hl_pk.addWidget(lbl_nsph)
    l_pk.addLayout(hl_pk)

    hl_sele = QtWidgets.QHBoxLayout()
    b_from_sele  = QtWidgets.QPushButton("From PyMOL sele")
    b_from_sele.setToolTip(
        "Click sphere atoms in the PyMOL 3D view to build a selection,\n"
        "then click this button to use those spheres as a custom subpocket.\n"
        "Overrides the pocket checkboxes above.")
    b_clear_sele = QtWidgets.QPushButton("Clear custom")
    b_clear_sele.setToolTip("Remove custom sphere selection; revert to pocket checkboxes.")
    lbl_sele_status = QtWidgets.QLabel("using pocket checkboxes")
    lbl_sele_status.setStyleSheet("color: grey; font-style: italic;")
    hl_sele.addWidget(b_from_sele); hl_sele.addWidget(b_clear_sele)
    hl_sele.addStretch(); hl_sele.addWidget(lbl_sele_status)
    l_pk.addLayout(hl_sele)

    top_l.addWidget(g_pk)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Filter options
    # ══════════════════════════════════════════════════════════════════════════
    g_flt, _flt_body, l_flt = _section("Filter", expanded=True)

    hl_params = QtWidgets.QHBoxLayout()
    lbl_cut = QtWidgets.QLabel("Distance cutoff (Å):")
    lbl_cut.setToolTip(
        "Used when no PQR sphere radii are loaded.\n"
        "A pose atom must be within this distance of a sphere centre.")
    hl_params.addWidget(lbl_cut)
    sb_cut = QtWidgets.QDoubleSpinBox()
    sb_cut.setRange(0.5, 10.0); sb_cut.setSingleStep(0.5)
    sb_cut.setDecimals(1); sb_cut.setValue(3.0); sb_cut.setFixedWidth(62)
    hl_params.addWidget(sb_cut)
    hl_params.addSpacing(14)
    hl_params.addWidget(QtWidgets.QLabel("Tolerance (Å):"))
    sb_tol = QtWidgets.QDoubleSpinBox()
    sb_tol.setRange(0.0, 5.0); sb_tol.setSingleStep(0.1)
    sb_tol.setDecimals(1); sb_tol.setValue(0.0); sb_tol.setFixedWidth(55)
    sb_tol.setToolTip(
        "Extra cushion added to each sphere radius (or cutoff).")
    hl_params.addWidget(sb_tol); hl_params.addStretch()
    l_flt.addLayout(hl_params)

    lbl_rmode = QtWidgets.QLabel("Radius mode: —")
    lbl_rmode.setStyleSheet("color: grey; font-style: italic;")
    l_flt.addWidget(lbl_rmode)

    b_filter = QtWidgets.QPushButton("Run Filter")
    f_flt = b_filter.font(); f_flt.setBold(True); b_filter.setFont(f_flt)
    l_flt.addWidget(b_filter)

    lbl_result = QtWidgets.QLabel("–")
    lbl_result.setAlignment(QtCore.Qt.AlignCenter)
    f_res = lbl_result.font(); f_res.setBold(True); lbl_result.setFont(f_res)
    l_flt.addWidget(lbl_result)
    top_l.addWidget(g_flt)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Results table
    # ══════════════════════════════════════════════════════════════════════════
    g_res, _res_body, l_res = _section("Results", expanded=True)

    class _SortItem(QtWidgets.QTableWidgetItem):
        def __lt__(self, other):
            a = self.data(QtCore.Qt.UserRole + 1)
            b = other.data(QtCore.Qt.UserRole + 1)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a < b
            return self.text() < other.text()

    tw = QtWidgets.QTableWidget(0, 0)
    tw.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    tw.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    tw.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    tw.setMinimumHeight(90)
    tw.horizontalHeader().setStretchLastSection(True)
    tw.horizontalHeader().setSectionsMovable(True)
    tw.verticalHeader().setDefaultSectionSize(18)
    tw.setAlternatingRowColors(True)
    l_res.addWidget(tw)

    hl_res = QtWidgets.QHBoxLayout()
    b_goto   = QtWidgets.QPushButton("Go to selected")
    b_export = QtWidgets.QPushButton("Export passing SDF…")
    hl_res.addWidget(b_goto); hl_res.addStretch(); hl_res.addWidget(b_export)
    l_res.addLayout(hl_res)
    bot_l.addWidget(g_res)
    bot_l.addStretch()

    splitter.setSizes([400, 200])

    # ── shared helpers ────────────────────────────────────────────────────────
    _worker: list = [None]   # mutable container so inner functions can rebind

    def _enabled_ids() -> set:
        ids = set()
        for row in range(tw_pockets.rowCount()):
            item = tw_pockets.item(row, 0)
            if item and item.checkState() == QtCore.Qt.Checked:
                ids.add(item.data(QtCore.Qt.UserRole))
        return ids

    def _refresh_sphere_count():
        ids = _enabled_ids()
        n = sum(p.n_spheres for p in _st.pockets if p.pid in ids)
        lbl_nsph.setText(f"{n} spheres")
        _highlight_pockets(ids)

    def _update_rmode():
        if not _st.pockets:
            lbl_rmode.setText("Radius mode: —")
        elif _st.has_radii:
            lbl_rmode.setText("Radius mode: sphere radius from PQR + tolerance")
        else:
            lbl_rmode.setText("Radius mode: fixed distance cutoff (no PQR)")

    def _rebuild_pocket_list():
        tw_pockets.setSortingEnabled(False)
        tw_pockets.blockSignals(True)
        tw_pockets.setRowCount(0)
        for i, p in enumerate(_st.pockets):
            row = tw_pockets.rowCount()
            tw_pockets.insertRow(row)
            # Col 0: checkbox
            chk = QtWidgets.QTableWidgetItem()
            chk.setData(QtCore.Qt.UserRole, p.pid)
            chk.setCheckState(QtCore.Qt.Unchecked)
            chk.setFlags(
                QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)
            tw_pockets.setItem(row, 0, chk)
            # Col 1: pocket name (colored, sortable by pid)
            name_item = _SortItem(f"{p.name}  ({p.n_spheres} sph)")
            name_item.setForeground(QtGui.QColor(_POCKET_HEX[i % len(_POCKET_HEX)]))
            f_name = name_item.font(); f_name.setBold(True); name_item.setFont(f_name)
            name_item.setData(QtCore.Qt.UserRole + 1, p.pid)
            tw_pockets.setItem(row, 1, name_item)
            # Cols 2+: fpocket info metrics
            for ci, col_name in enumerate(_PK_INFO_COLS):
                val = p.info.get(col_name, "")
                if val == "":
                    cell = _SortItem("")
                else:
                    disp = f"{val:.3f}" if isinstance(val, float) else str(val)
                    cell = _SortItem(disp)
                    cell.setData(QtCore.Qt.UserRole + 1, float(val))
                    cell.setTextAlignment(
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                tw_pockets.setItem(row, 2 + ci, cell)
        tw_pockets.resizeColumnsToContents()
        tw_pockets.setColumnWidth(0, 28)
        tw_pockets.blockSignals(False)
        tw_pockets.setSortingEnabled(True)
        _refresh_sphere_count()
        _update_rmode()

    def _rebuild_table():
        tw.setSortingEnabled(False)
        tw.clearContents(); tw.setRowCount(0)
        if not _st.poses:
            tw.setColumnCount(0); return
        sdf_keys: List[str] = []
        if _st.all_properties:
            seen: Dict = {}
            for r in _st.all_properties:
                for k in r:
                    if k not in seen:
                        seen[k] = None
            sdf_keys = [k for k in seen if k != "_name"]
        cols = ["Pass", "Pose", "State"] + sdf_keys
        tw.setColumnCount(len(cols))
        tw.setHorizontalHeaderLabels(cols)
        tw.setRowCount(len(_st.poses))
        for row, (obj, state) in enumerate(_st.poses):
            result = _st.results[row] if row < len(_st.results) else None
            if result is True:
                cell = QtWidgets.QTableWidgetItem("✓")
                cell.setForeground(QtGui.QColor("#33cc33"))
            elif result is False:
                cell = QtWidgets.QTableWidgetItem("✗")
                cell.setForeground(QtGui.QColor("#cc3333"))
            else:
                cell = QtWidgets.QTableWidgetItem("–")
            cell.setData(QtCore.Qt.UserRole, row)
            cell.setTextAlignment(QtCore.Qt.AlignCenter)
            tw.setItem(row, 0, cell)
            pi = _SortItem(obj); pi.setData(QtCore.Qt.UserRole, row)
            tw.setItem(row, 1, pi)
            si = _SortItem(str(state))
            si.setData(QtCore.Qt.UserRole + 1, state)
            si.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            tw.setItem(row, 2, si)
            sdf_rec = _st.all_properties[row] if row < len(_st.all_properties) else {}
            for ci, key in enumerate(sdf_keys):
                val = sdf_rec.get(key, "")
                disp = f"{val:.2f}" if isinstance(val, float) else (
                    str(val) if val != "" else "")
                ci_item = _SortItem(disp)
                ci_item.setData(QtCore.Qt.UserRole, row)
                if isinstance(val, (int, float)):
                    ci_item.setData(QtCore.Qt.UserRole + 1, val)
                    ci_item.setTextAlignment(
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                tw.setItem(row, 3 + ci, ci_item)
        tw.resizeColumnsToContents()
        tw.setSortingEnabled(True)

    def _finalize(ok: bool):
        if not ok:
            QtWidgets.QMessageBox.warning(
                win, "PickPocket",
                "No fpocket pocket spheres found in the current PyMOL session.\n\n"
                "fpocket must be run first.  Three ways to do this:\n\n"
                "1. Use 'Run fpocket' above — select your protein object\n"
                "   and click 'Run fpocket'.\n\n"
                "2. Load an existing *_out/ directory with 'Load *_out/ dir'.\n\n"
                "3. Run the fpocket *.pml script in PyMOL manually, then\n"
                "   click 'Detect from PyMOL'.")
            lbl_in_status.setText("No pockets found — run fpocket first")
            return
        _st.protein_sel = e_prot.text().strip() or "polymer.protein"
        _st.poses_sel   = e_poses.text().strip() or "organic"
        _st.poses = _discover_poses(_st.poses_sel, _st.protein_sel)
        _st.results = [None] * len(_st.poses)
        sdf = e_sdf.text().strip()
        _st.sdf_records = _parse_sdf_records(sdf) if sdf and os.path.isfile(sdf) else []
        _st.all_properties = _build_all_properties()
        _rebuild_pocket_list()
        _rebuild_table()
        note = " (radii from PQR)" if _st.has_radii else " (cutoff mode)"
        lbl_in_status.setText(
            f"{len(_st.pockets)} pockets{note}, {len(_st.poses)} poses")

    # ── Run fpocket callbacks ─────────────────────────────────────────────────
    def _refresh_prot_combo():
        prev = cb_prot.currentText()
        cb_prot.blockSignals(True); cb_prot.clear()
        for n in cmd.get_names("objects"):
            if n.startswith("_pp_") or "pocket" in n.lower():
                continue
            try:
                if cmd.count_atoms(f"({n}) and polymer") > 0:
                    cb_prot.addItem(n)
            except Exception:
                pass
        idx = cb_prot.findText(prev)
        if idx >= 0:
            cb_prot.setCurrentIndex(idx)
        cb_prot.blockSignals(False)

    def do_browse_fpbin():
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            win, "Select fpocket binary", "", "All (*)")
        if p:
            e_fpbin.setText(p)

    def do_browse_pdbfile():
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            win, "Select PDB file", "", "PDB (*.pdb *.ent);;All (*)")
        if p:
            e_pdbfile.setText(p)

    def _on_fpocket_done(out_dir: str, stem: str):
        b_run_fp.setEnabled(True)
        pb_run.setVisible(False)
        log_run.appendPlainText(f"✓ fpocket finished → {out_dir}")

        # Load all pocket PQR files
        pockets_dir = os.path.join(out_dir, "pockets")
        info_txt    = os.path.join(out_dir, f"{stem}_info.txt")
        _st.pockets.clear()
        _st.has_radii = False

        # Determine pocket count from vert files
        pids = []
        if os.path.isdir(pockets_dir):
            for fname in sorted(os.listdir(pockets_dir)):
                m = re.match(r"pocket(\d+)_vert\.pqr$", fname)
                if m:
                    pids.append(int(m.group(1)))

        info_data = {}
        if os.path.isfile(info_txt):
            info_data = _parse_fpocket_info(info_txt)

        for pid in sorted(pids):
            vert = os.path.join(pockets_dir, f"pocket{pid}_vert.pqr")
            _load_pocket_pqr(pid, vert)
            p = Pocket(pid)
            centers, radii = _parse_vert_pqr(vert)
            p.centers = centers
            p.radii   = radii
            rec = info_data.get(pid, {})
            p.info       = rec
            p.score      = rec.get("Score")
            p.drug_score = rec.get("Druggability Score")
            _st.pockets.append(p)

        _st.has_radii = bool(pids)
        _st.out_dir   = out_dir
        e_dir.setText(out_dir)
        _finalize(bool(_st.pockets))
        log_run.appendPlainText(f"✓ {len(_st.pockets)} pocket(s) loaded")

    def _on_fpocket_failed(msg: str):
        b_run_fp.setEnabled(True)
        pb_run.setVisible(False)
        log_run.appendPlainText(f"✗ {msg}")
        QtWidgets.QMessageBox.critical(win, "fpocket failed", msg)

    def do_run_fpocket():
        fpbin = e_fpbin.text().strip()
        if not fpbin:
            QtWidgets.QMessageBox.warning(win, "PickPocket",
                                          "Specify the fpocket binary path.")
            return
        if rb_file.isChecked():
            pdb_path = e_pdbfile.text().strip()
            if not pdb_path or not os.path.isfile(pdb_path):
                QtWidgets.QMessageBox.warning(win, "PickPocket",
                                              "Select a valid PDB file.")
                return
        else:
            prot = cb_prot.currentText()
            if not prot:
                QtWidgets.QMessageBox.warning(win, "PickPocket",
                                              "Select a protein object.")
                return
            tmpdir = tempfile.mkdtemp(prefix="pp_fpocket_")
            pdb_path = os.path.join(tmpdir, f"{prot}.pdb")
            try:
                cmd.save(pdb_path, prot, state=-1)
            except Exception as e:
                QtWidgets.QMessageBox.warning(win, "PickPocket",
                                              f"Could not save {prot}: {e}")
                return

        log_run.clear()
        log_run.appendPlainText(f"Running fpocket on {os.path.basename(pdb_path)} …")
        b_run_fp.setEnabled(False)
        pb_run.setVisible(True)

        w = FpocketWorker(fpbin, pdb_path)
        w.log_line.connect(log_run.appendPlainText)
        w.finished.connect(_on_fpocket_done)
        w.failed.connect(_on_fpocket_failed)
        w.start()
        _worker[0] = w    # keep alive

    # ── Input section callbacks ───────────────────────────────────────────────
    def do_browse_dir():
        p = QtWidgets.QFileDialog.getExistingDirectory(
            win, "Select fpocket *_out/ directory")
        if p:
            e_dir.setText(p)

    def do_browse_sdf():
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            win, "Select scores SDF", "",
            "SDF files (*.sdf *.SDF);;All files (*)")
        if p:
            e_sdf.setText(p)

    def do_detect():
        ok = _detect_from_pymol()
        _finalize(ok)

    def do_load_dir():
        path = e_dir.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(
                win, "PickPocket",
                "Specify the fpocket *_out/ directory first.")
            return
        if not os.path.isdir(path):
            QtWidgets.QMessageBox.warning(
                win, "PickPocket", f"Directory not found:\n{path}")
            return
        _finalize(_load_from_dir(path))

    def do_setup_poses():
        _st.protein_sel = e_prot.text().strip() or "polymer.protein"
        _st.poses_sel   = e_poses.text().strip() or "organic"
        _st.poses = _discover_poses(_st.poses_sel, _st.protein_sel)
        _st.results = [None] * len(_st.poses)
        sdf = e_sdf.text().strip()
        _st.sdf_records = _parse_sdf_records(sdf) if sdf and os.path.isfile(sdf) else []
        _st.all_properties = _build_all_properties()
        _rebuild_table()
        note = " (radii from PQR)" if _st.has_radii else " (cutoff mode)"
        lbl_in_status.setText(
            f"{len(_st.pockets)} pockets{note}, {len(_st.poses)} poses, "
            f"{len(_st.sdf_records)} SDF records")

    def do_clear():
        # Remove pocket objects created by this plugin
        for obj in list(_st._created_objs):
            if obj in cmd.get_names("objects"):
                try: cmd.delete(obj)
                except Exception: pass
        _st._created_objs.clear()
        _st.pockets.clear(); _st.poses.clear()
        _st.results.clear(); _st.sdf_records.clear(); _st.all_properties.clear()
        _st.custom_spheres = None
        _st.has_radii = False
        tw_pockets.setRowCount(0)
        tw.clearContents(); tw.setRowCount(0); tw.setColumnCount(0)
        lbl_result.setText("–"); lbl_nsph.setText("0 spheres")
        lbl_sele_status.setText("using pocket checkboxes")
        lbl_sele_status.setStyleSheet("color: grey; font-style: italic;")
        lbl_in_status.setText(""); _update_rmode()

    # ── Pocket list callbacks ─────────────────────────────────────────────────
    def do_from_sele():
        spheres = []
        try:
            model = cmd.get_model("sele")
            for atom in model.atom:
                r = atom.b if atom.b > 0.5 else _st.cutoff
                spheres.append((atom.coord[0], atom.coord[1], atom.coord[2], r))
        except Exception as e:
            print(f"PickPocket: sele read error: {e}")
        if not spheres:
            QtWidgets.QMessageBox.information(
                win, "PickPocket",
                "Nothing in PyMOL's 'sele'.\n\n"
                "Click sphere atoms in the 3D view first (Ctrl+click to add to\n"
                "an existing selection), then click 'From PyMOL sele'.")
            return
        _st.custom_spheres = spheres
        lbl_sele_status.setText(f"{len(spheres)} custom spheres locked")
        lbl_sele_status.setStyleSheet("color: #cc7700; font-style: italic; font-weight: bold;")
        lbl_nsph.setText(f"{len(spheres)} spheres (custom)")

    def do_clear_sele():
        _st.custom_spheres = None
        lbl_sele_status.setText("using pocket checkboxes")
        lbl_sele_status.setStyleSheet("color: grey; font-style: italic;")
        _refresh_sphere_count()

    def do_all():
        tw_pockets.blockSignals(True)
        for i in range(tw_pockets.rowCount()):
            item = tw_pockets.item(i, 0)
            if item:
                item.setCheckState(QtCore.Qt.Checked)
        tw_pockets.blockSignals(False)
        _refresh_sphere_count()

    def do_none():
        tw_pockets.blockSignals(True)
        for i in range(tw_pockets.rowCount()):
            item = tw_pockets.item(i, 0)
            if item:
                item.setCheckState(QtCore.Qt.Unchecked)
        tw_pockets.blockSignals(False)
        _refresh_sphere_count()

    # ── Filter callbacks ──────────────────────────────────────────────────────
    def do_filter():
        if not _st.poses:
            QtWidgets.QMessageBox.information(
                win, "PickPocket",
                "No poses loaded. Configure Protein/Poses fields and click Detect.")
            return
        ids = _enabled_ids()
        if not ids:
            QtWidgets.QMessageBox.information(
                win, "PickPocket", "No pockets selected.")
            return
        _st.cutoff    = sb_cut.value()
        _st.tolerance = sb_tol.value()
        res = pp_filter(enabled_ids=ids)
        lbl_result.setText(f"{len(res['pass'])} / {len(_st.poses)} poses pass")
        _rebuild_table()

    def do_goto():
        rows = [idx.row() for idx in tw.selectionModel().selectedRows()]
        if not rows:
            return
        item0 = tw.item(rows[0], 0)
        if item0 is None:
            return
        pi = item0.data(QtCore.Qt.UserRole)
        if pi is None or pi >= len(_st.poses):
            return
        obj, state = _st.poses[pi]
        try:
            active = {o for o, _ in _st.poses}
            for n in active:
                if n in cmd.get_names("objects"):
                    cmd.disable(n)
            cmd.enable(obj)
            cmd.set("state", state)
            cmd.zoom(obj, buffer=3.0, animate=1, state=state)
        except Exception as e:
            print(f"PickPocket: goto error: {e}")

    def do_export():
        if not _st.results or not any(_st.results):
            QtWidgets.QMessageBox.information(
                win, "PickPocket", "No passing poses to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            win, "Export passing poses", "",
            "SDF files (*.sdf);;All files (*)")
        if not path:
            return
        sdf_src = e_sdf.text().strip()
        if sdf_src and os.path.isfile(sdf_src):
            _export_from_sdf(sdf_src, path)
        else:
            _export_from_pymol(path)

    def _export_from_sdf(src: str, dst: str):
        pass_set = {i for i, r in enumerate(_st.results) if r}
        try:
            with open(src) as fh:
                blocks = [b for b in fh.read().split("$$$$") if b.strip()]
            with open(dst, "w") as out:
                for i, block in enumerate(blocks):
                    if i in pass_set:
                        out.write(block.strip() + "\n$$$$\n")
            print(f"PickPocket: exported {len(pass_set)} records to {dst}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(win, "PickPocket", str(e))

    def _export_from_pymol(dst: str):
        written = 0
        try:
            with open(dst, "w") as out:
                for (obj, state), r in zip(_st.poses, _st.results):
                    if not r:
                        continue
                    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
                    tmp.close()
                    try:
                        cmd.save(tmp.name, f"({obj})", state=state)
                        with open(tmp.name) as fh:
                            text = fh.read().strip()
                        lines = text.splitlines()
                        if lines:
                            lines[0] = f"{obj}_s{state}"
                        out.write("\n".join(lines) + "\n$$$$\n")
                        written += 1
                    except Exception as e:
                        print(f"PickPocket: save error {obj} s{state}: {e}")
                    finally:
                        os.unlink(tmp.name)
            print(f"PickPocket: exported {written} poses to {dst}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(win, "PickPocket", str(e))

    # ── Wire up signals ───────────────────────────────────────────────────────
    b_fpbin.clicked.connect(do_browse_fpbin)
    b_refresh.clicked.connect(_refresh_prot_combo)
    b_pdbfile.clicked.connect(do_browse_pdbfile)
    b_run_fp.clicked.connect(do_run_fpocket)

    b_dir.clicked.connect(do_browse_dir)
    b_sdf.clicked.connect(do_browse_sdf)
    b_detect.clicked.connect(do_detect)
    b_load.clicked.connect(do_load_dir)
    b_setup_poses.clicked.connect(do_setup_poses)
    b_clear.clicked.connect(do_clear)

    b_from_sele.clicked.connect(do_from_sele)
    b_clear_sele.clicked.connect(do_clear_sele)
    b_all.clicked.connect(do_all)
    b_none.clicked.connect(do_none)
    tw_pockets.itemChanged.connect(lambda _: _refresh_sphere_count())

    b_filter.clicked.connect(do_filter)
    b_goto.clicked.connect(do_goto)
    b_export.clicked.connect(do_export)

    _refresh_prot_combo()
    win.show()
    return win
