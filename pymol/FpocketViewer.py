"""
FpocketViewer - PyMOL Plugin
============================
Standalone fpocket binding-site detection and pocket explorer.

Features
--------
- Run fpocket on any loaded protein object or external PDB file (background)
- Load an existing fpocket *_out/ directory
- Colour-coded alpha-sphere visualization per pocket
- Sortable pocket table: score, druggability, volume, N spheres, SASA…
- Per-pocket: lining residue list, zoom-to-pocket, toggle surface

Installation
------------
  Plugin > Plugin Manager > Install New Plugin > choose this file
  — or —
  run /path/to/FpocketViewer.py   then   fpv_gui

CLI commands
------------
  fpv_load /path/to/protein_out/   load existing *_out/ directory
  fpv_clear                        remove all FpocketViewer objects
  fpv_gui                          open the GUI

Authors: Evert J. Homan, PhD; Claude (Anthropic)
Date:    2026-07-01
Version: 1.0
License: MIT
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from pymol import cmd

_IS_WINDOWS = platform.system() == "Windows"
_PREFIX = "fpv_"

_POCKET_PYMOL = [
    "cyan", "yellow", "orange", "green", "magenta",
    "purple", "red", "lime", "salmon", "slate",
]
_POCKET_HEX = [
    "#4dc0ff", "#ffd900", "#ff6633", "#33cc33", "#e633e6",
    "#9933e6", "#ff4444", "#4dd97f", "#ff99cc", "#aaccff",
]

_FPOCKET_COL_MAP = {
    "score":                                 "Score",
    "druggability score":                    "Drug. Score",
    "number of alpha spheres":               "N Spheres",
    "volume":                                "Volume",
    "total sasa":                            "Total SASA",
    "polar sasa":                            "Polar SASA",
    "apolar sasa":                           "Apolar SASA",
    "mean local hydrophobic density":        "Hydrophobic Dens.",
    "mean alpha sphere radius":              "Mean α Radius",
    "mean alp. sph. solvent access":         "Mean Solv. Access",
    "apolar alpha sphere proportion":        "Apolar Proportion",
    "hydrophobicity score":                  "Hydrophobicity",
    "volume score":                          "Volume Score",
    "polarity score":                        "Polarity Score",
    "charge score":                          "Charge Score",
    "proportion of polar atoms":             "Polar Atom Prop.",
    "alpha sphere density":                  "Sphere Density",
    "cent. of mass - alpha sphere max dist": "CoM-AS Dist",
    "flexibility":                           "Flexibility",
}

_INFO_COLS = list(_FPOCKET_COL_MAP.values())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class Pocket:
    def __init__(self, pid: int):
        self.pid = pid
        self.centers: List[Tuple[float, float, float]] = []
        self.radii: List[float] = []
        self.info: Dict = {}
        self.score: Optional[float] = None
        self.drug_score: Optional[float] = None

    @property
    def n_spheres(self) -> int:
        return len(self.centers)

    @property
    def sph_obj(self) -> str:
        return f"{_PREFIX}p{self.pid}_sph"

    @property
    def surf_obj(self) -> str:
        return f"{_PREFIX}p{self.pid}_surf"


class _State:
    def __init__(self):
        self.pockets: List[Pocket] = []
        self.out_dir: Optional[str] = None
        self._created_objs: set = set()


_st = _State()
_gui_window = None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_vert_pqr(path: str) -> Tuple[List[Tuple], List[float]]:
    centers: List[Tuple] = []
    radii: List[float] = []
    try:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                r = float(parts[9])
                centers.append((x, y, z))
                radii.append(r)
    except Exception as e:
        print(f"FpocketViewer: PQR read error {path}: {e}")
    return centers, radii


def _parse_fpocket_info(info_path: str) -> Dict[int, Dict]:
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
        print(f"FpocketViewer: info parse error: {e}")
    return pockets


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _load_pocket_spheres(p: Pocket, vert_path: str):
    obj = p.sph_obj
    if obj in cmd.get_names("objects"):
        cmd.delete(obj)
    try:
        cmd.load(vert_path, obj)
        cmd.hide("everything", obj)
        cmd.show("spheres", obj)
        cmd.set("sphere_scale", 0.3, obj)
        cmd.set("sphere_transparency", 0.35, obj)
        cmd.color(_POCKET_PYMOL[(p.pid - 1) % len(_POCKET_PYMOL)], obj)
        _st._created_objs.add(obj)
    except Exception as e:
        print(f"FpocketViewer: sphere load error {obj}: {e}")


def _toggle_pocket_surface(p: Pocket, protein_sel: str = "polymer.protein"):
    """Create pocket surface if absent; delete it if already shown (toggle)."""
    obj = p.surf_obj
    if obj in cmd.get_names("objects"):
        cmd.delete(obj)
        _st._created_objs.discard(obj)
        return

    sph = p.sph_obj
    if sph not in cmd.get_names("objects"):
        print("FpocketViewer: load pocket spheres first.")
        return

    tmp = f"_fpv_tmp_{p.pid}"
    try:
        cmd.select(tmp, f"({protein_sel}) within 4.5 of {sph}")
        if cmd.count_atoms(tmp) == 0:
            cmd.delete(tmp)
            print(f"FpocketViewer: no protein atoms within 4.5 Å of pocket {p.pid}.")
            return
        cmd.create(obj, tmp)
        cmd.delete(tmp)
        cmd.hide("everything", obj)
        cmd.show("surface", obj)
        cmd.color(_POCKET_PYMOL[(p.pid - 1) % len(_POCKET_PYMOL)], obj)
        cmd.set("surface_transparency", 0.5, obj)
        _st._created_objs.add(obj)
    except Exception as e:
        print(f"FpocketViewer: surface error: {e}")
        try:
            cmd.delete(tmp)
        except Exception:
            pass


def _get_lining_residues(p: Pocket, protein_sel: str = "polymer.protein") -> List[str]:
    sph = p.sph_obj
    if sph not in cmd.get_names("objects"):
        return []
    tmp = f"_fpv_lint_{p.pid}"
    residues: List[str] = []
    try:
        cmd.select(tmp, f"({protein_sel}) within 4.5 of {sph}")
        entries: List[Tuple] = []
        cmd.iterate(tmp, "entries.append((chain, resn, resi))",
                    space={"entries": entries})
        cmd.delete(tmp)
        seen: set = set()
        for chain, resn, resi in entries:
            key = (chain, resn, resi)
            if key not in seen:
                seen.add(key)
                label = f"{resn} {resi}" + (f":{chain}" if chain.strip() else "")
                residues.append(label)
    except Exception as e:
        print(f"FpocketViewer: lining residues error: {e}")
        try:
            cmd.delete(tmp)
        except Exception:
            pass
    return residues


def _highlight_pocket(pid: int):
    for p in _st.pockets:
        obj = p.sph_obj
        if obj not in cmd.get_names("objects"):
            continue
        cmd.set("sphere_transparency", 0.15 if p.pid == pid else 0.75, obj)


def _reset_transparency():
    for p in _st.pockets:
        obj = p.sph_obj
        if obj in cmd.get_names("objects"):
            cmd.set("sphere_transparency", 0.35, obj)


# ---------------------------------------------------------------------------
# Load from *_out/ directory
# ---------------------------------------------------------------------------

def _load_from_dir(out_dir: str) -> bool:
    # Clean up previous objects
    for obj in list(_st._created_objs):
        if obj in cmd.get_names("objects"):
            try:
                cmd.delete(obj)
            except Exception:
                pass
    _st._created_objs.clear()
    _st.pockets.clear()
    _st.out_dir = out_dir

    info_txt: Optional[str] = None
    for fname in os.listdir(out_dir):
        if fname.endswith("_info.txt"):
            info_txt = os.path.join(out_dir, fname)
            break

    pockets_dir = os.path.join(out_dir, "pockets")
    if not os.path.isdir(pockets_dir):
        pockets_dir = out_dir

    pids: List[int] = []
    for fname in sorted(os.listdir(pockets_dir)):
        m = re.match(r"pocket(\d+)_vert\.pqr$", fname)
        if m:
            pids.append(int(m.group(1)))

    if not pids:
        return False

    info_data = _parse_fpocket_info(info_txt) if info_txt else {}

    for pid in sorted(pids):
        vert = os.path.join(pockets_dir, f"pocket{pid}_vert.pqr")
        p = Pocket(pid)
        p.centers, p.radii = _parse_vert_pqr(vert)
        rec = info_data.get(pid, {})
        p.info = rec
        p.score = rec.get("Score")
        p.drug_score = rec.get("Drug. Score")
        _st.pockets.append(p)
        _load_pocket_spheres(p, vert)

    return True


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def fpv_load(path=""):
    """Load fpocket *_out/ directory.  Usage: fpv_load /path/to/protein_out/"""
    path = path.strip()
    if not path or not os.path.isdir(path):
        print("FpocketViewer: provide a valid *_out/ directory.")
        return
    ok = _load_from_dir(path)
    print(f"FpocketViewer: {'loaded' if ok else 'no pockets found in'} "
          f"{len(_st.pockets)} pocket(s) from {path}.")


def fpv_clear():
    """Remove all FpocketViewer objects from the session.  Usage: fpv_clear"""
    for obj in list(_st._created_objs):
        if obj in cmd.get_names("objects"):
            try:
                cmd.delete(obj)
            except Exception:
                pass
    _st._created_objs.clear()
    _st.pockets.clear()
    _st.out_dir = None
    print("FpocketViewer: cleared.")


def fpv_gui():
    """Open the FpocketViewer GUI.  Usage: fpv_gui"""
    _open_gui()


cmd.extend("fpv_load",  fpv_load)
cmd.extend("fpv_clear", fpv_clear)
cmd.extend("fpv_gui",   fpv_gui)


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------

def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("FpocketViewer", _open_gui)


def _set_gui_none():
    global _gui_window
    _gui_window = None


def _make_worker_class():
    from pymol.Qt import QtCore

    class FpocketWorker(QtCore.QThread):
        log_line = QtCore.Signal(str)
        finished = QtCore.Signal(str, str)   # out_dir, stem
        failed   = QtCore.Signal(str)

        def __init__(self, fpocket_bin: str, pdb_path: str, extra_args: list):
            super().__init__()
            self._bin   = fpocket_bin
            self._pdb   = pdb_path
            self._extra = extra_args

        def run(self):
            from pathlib import Path
            try:
                stem    = Path(self._pdb).stem
                out_dir = Path(self._pdb).parent / f"{stem}_out"
                if out_dir.exists():
                    shutil.rmtree(str(out_dir))

                if _IS_WINDOWS:
                    def _to_wsl(p: str) -> str:
                        p = os.path.abspath(p).replace("\\", "/")
                        if len(p) >= 2 and p[1] == ":":
                            return f"/mnt/{p[0].lower()}{p[2:]}"
                        return p
                    args = (["wsl.exe", self._bin, "-f", _to_wsl(self._pdb)]
                            + self._extra)
                else:
                    args = [self._bin, "-f", self._pdb] + self._extra

                proc = subprocess.Popen(
                    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.log_line.emit(line.rstrip())
                proc.wait()

                if proc.returncode == 0 and out_dir.exists():
                    self.finished.emit(str(out_dir), stem)
                else:
                    self.failed.emit(f"fpocket exited with code {proc.returncode}")
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
    win.setWindowTitle("FpocketViewer 1.0")
    win.setMinimumWidth(540)
    win.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    win.destroyed.connect(_set_gui_none)

    root = QtWidgets.QVBoxLayout(win)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(6)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Run fpocket
    # ══════════════════════════════════════════════════════════════════════════
    g_run = QtWidgets.QGroupBox("Run fpocket")
    l_run = QtWidgets.QVBoxLayout(g_run)
    l_run.setSpacing(4)

    # Binary path
    hl_fp = QtWidgets.QHBoxLayout()
    hl_fp.addWidget(QtWidgets.QLabel("fpocket:"))
    e_fpbin = QtWidgets.QLineEdit(shutil.which("fpocket") or "fpocket")
    b_fpbin = QtWidgets.QPushButton("…"); b_fpbin.setFixedWidth(26)
    hl_fp.addWidget(e_fpbin, 1); hl_fp.addWidget(b_fpbin)
    l_run.addLayout(hl_fp)

    # Input source toggle
    hl_mode = QtWidgets.QHBoxLayout()
    rb_obj  = QtWidgets.QRadioButton("PyMOL object")
    rb_file = QtWidgets.QRadioButton("PDB file")
    rb_obj.setChecked(True)
    hl_mode.addWidget(rb_obj); hl_mode.addWidget(rb_file); hl_mode.addStretch()
    l_run.addLayout(hl_mode)

    prot_stack = QtWidgets.QStackedWidget()

    obj_page = QtWidgets.QWidget()
    ol = QtWidgets.QHBoxLayout(obj_page); ol.setContentsMargins(0, 0, 0, 0)
    cb_prot = QtWidgets.QComboBox(); cb_prot.setMinimumWidth(160)
    b_refresh = QtWidgets.QPushButton("Refresh"); b_refresh.setFixedWidth(60)
    ol.addWidget(cb_prot, 1); ol.addWidget(b_refresh)
    prot_stack.addWidget(obj_page)

    file_page = QtWidgets.QWidget()
    fl = QtWidgets.QHBoxLayout(file_page); fl.setContentsMargins(0, 0, 0, 0)
    e_pdbfile = QtWidgets.QLineEdit()
    e_pdbfile.setPlaceholderText("path/to/protein.pdb")
    b_pdbfile = QtWidgets.QPushButton("…"); b_pdbfile.setFixedWidth(26)
    fl.addWidget(e_pdbfile, 1); fl.addWidget(b_pdbfile)
    prot_stack.addWidget(file_page)

    rb_obj.toggled.connect(lambda on: prot_stack.setCurrentIndex(0 if on else 1))
    l_run.addWidget(prot_stack)

    # Optional fpocket parameters
    g_params = QtWidgets.QGroupBox("Detection parameters")
    g_params.setCheckable(True); g_params.setChecked(False)
    l_params = QtWidgets.QGridLayout(g_params)
    l_params.setSpacing(3)
    param_fields: Dict[str, QtWidgets.QLineEdit] = {}
    for (flag, label, default), (row, col) in zip(
        [("-m", "Min α-sphere radius (Å)", "3.4"),
         ("-M", "Max α-sphere radius (Å)", "6.2"),
         ("-i", "Min spheres/pocket",       "15"),
         ("-D", "Cluster distance (Å)",     "2.4")],
        [(0, 0), (0, 2), (1, 0), (1, 2)],
    ):
        l_params.addWidget(QtWidgets.QLabel(label + ":"), row, col)
        le = QtWidgets.QLineEdit(default); le.setFixedWidth(52)
        param_fields[flag] = le
        l_params.addWidget(le, row, col + 1)
    l_run.addWidget(g_params)

    hl_run_btns = QtWidgets.QHBoxLayout()
    b_run_fp = QtWidgets.QPushButton("Run fpocket")
    f_bold = b_run_fp.font(); f_bold.setBold(True); b_run_fp.setFont(f_bold)
    b_clear = QtWidgets.QPushButton("Clear all")
    hl_run_btns.addWidget(b_run_fp); hl_run_btns.addWidget(b_clear)
    hl_run_btns.addStretch()
    l_run.addLayout(hl_run_btns)

    pb_run = QtWidgets.QProgressBar()
    pb_run.setRange(0, 0); pb_run.setVisible(False); pb_run.setMaximumHeight(14)
    l_run.addWidget(pb_run)

    log_run = QtWidgets.QPlainTextEdit()
    log_run.setReadOnly(True)
    f_log = QtGui.QFont("Monospace"); f_log.setStyleHint(QtGui.QFont.Monospace)
    f_log.setPointSize(8); log_run.setFont(f_log)
    log_run.setMaximumBlockCount(2000); log_run.setMaximumHeight(72)
    l_run.addWidget(log_run)

    # Load existing *_out/ dir
    hl_load = QtWidgets.QHBoxLayout()
    hl_load.addWidget(QtWidgets.QLabel("Load *_out/:"))
    e_dir = QtWidgets.QLineEdit()
    e_dir.setPlaceholderText("path/to/protein_out/  (skip if you just ran fpocket above)")
    b_dir = QtWidgets.QPushButton("…"); b_dir.setFixedWidth(26)
    b_load = QtWidgets.QPushButton("Load")
    hl_load.addWidget(e_dir, 1); hl_load.addWidget(b_dir); hl_load.addWidget(b_load)
    l_run.addLayout(hl_load)

    root.addWidget(g_run)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Pockets table
    # ══════════════════════════════════════════════════════════════════════════
    g_pk = QtWidgets.QGroupBox("Pockets")
    l_pk = QtWidgets.QVBoxLayout(g_pk)
    l_pk.setSpacing(4)

    class _SortItem(QtWidgets.QTableWidgetItem):
        def __lt__(self, other):
            a = self.data(QtCore.Qt.UserRole + 1)
            b = other.data(QtCore.Qt.UserRole + 1)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a < b
            return self.text() < other.text()

    tw = QtWidgets.QTableWidget(0, 2 + len(_INFO_COLS))
    tw.setHorizontalHeaderLabels(["", "Pocket"] + _INFO_COLS)
    tw.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    tw.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    tw.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    tw.setMinimumHeight(130); tw.setMaximumHeight(280)
    tw.horizontalHeader().setStretchLastSection(False)
    tw.horizontalHeader().setSectionsMovable(True)
    tw.setSortingEnabled(True)
    tw.verticalHeader().setDefaultSectionSize(20)
    tw.setAlternatingRowColors(True)
    tw.setColumnWidth(0, 24)
    l_pk.addWidget(tw)

    hl_vis = QtWidgets.QHBoxLayout()
    b_show_all = QtWidgets.QPushButton("Show all")
    b_hide_all = QtWidgets.QPushButton("Hide all")
    b_zoom     = QtWidgets.QPushButton("Zoom to pocket")
    b_surf     = QtWidgets.QPushButton("Toggle surface")
    hl_vis.addWidget(b_show_all); hl_vis.addWidget(b_hide_all)
    hl_vis.addStretch()
    hl_vis.addWidget(b_zoom); hl_vis.addWidget(b_surf)
    l_pk.addLayout(hl_vis)

    lbl_status = QtWidgets.QLabel("")
    lbl_status.setStyleSheet("color: grey; font-style: italic;")
    l_pk.addWidget(lbl_status)

    root.addWidget(g_pk)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Lining residues
    # ══════════════════════════════════════════════════════════════════════════
    g_res = QtWidgets.QGroupBox("Lining residues  (within 4.5 Å of alpha spheres)")
    l_res = QtWidgets.QVBoxLayout(g_res)
    l_res.setSpacing(3)

    hl_psel = QtWidgets.QHBoxLayout()
    hl_psel.addWidget(QtWidgets.QLabel("Protein sel:"))
    e_prot_sel = QtWidgets.QLineEdit("polymer.protein")
    b_get_res = QtWidgets.QPushButton("Get residues")
    hl_psel.addWidget(e_prot_sel, 1); hl_psel.addWidget(b_get_res)
    l_res.addLayout(hl_psel)

    te_res = QtWidgets.QTextEdit()
    te_res.setReadOnly(True); te_res.setMaximumHeight(72)
    f_mono = QtGui.QFont("Monospace"); f_mono.setStyleHint(QtGui.QFont.Monospace)
    f_mono.setPointSize(8); te_res.setFont(f_mono)
    l_res.addWidget(te_res)

    root.addWidget(g_res)

    # ── Shared helpers ────────────────────────────────────────────────────────
    _worker: list = [None]

    def _selected_pocket() -> Optional[Pocket]:
        rows = [i.row() for i in tw.selectionModel().selectedRows()]
        if not rows:
            return None
        item = tw.item(rows[0], 0)
        if item is None:
            return None
        pid = item.data(QtCore.Qt.UserRole)
        for p in _st.pockets:
            if p.pid == pid:
                return p
        return None

    def _rebuild_table():
        tw.setSortingEnabled(False)
        tw.blockSignals(True)
        tw.setRowCount(0)
        for idx, p in enumerate(_st.pockets):
            row = tw.rowCount(); tw.insertRow(row)

            dot = QtWidgets.QTableWidgetItem("●")
            dot.setData(QtCore.Qt.UserRole, p.pid)
            dot.setForeground(QtGui.QColor(_POCKET_HEX[idx % len(_POCKET_HEX)]))
            dot.setTextAlignment(QtCore.Qt.AlignCenter)
            dot.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            tw.setItem(row, 0, dot)

            name = _SortItem(f"Pocket {p.pid}  ({p.n_spheres} sph)")
            name.setForeground(QtGui.QColor(_POCKET_HEX[idx % len(_POCKET_HEX)]))
            f = name.font(); f.setBold(True); name.setFont(f)
            name.setData(QtCore.Qt.UserRole + 1, p.pid)
            tw.setItem(row, 1, name)

            for ci, col_name in enumerate(_INFO_COLS):
                val = p.info.get(col_name, "")
                if val == "":
                    cell = _SortItem("")
                else:
                    disp = f"{val:.3f}" if isinstance(val, float) else str(val)
                    cell = _SortItem(disp)
                    cell.setData(QtCore.Qt.UserRole + 1, float(val))
                    cell.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                tw.setItem(row, 2 + ci, cell)

        tw.resizeColumnsToContents()
        tw.setColumnWidth(0, 24)
        tw.blockSignals(False)
        tw.setSortingEnabled(True)
        n = len(_st.pockets)
        lbl_status.setText(f"{n} pocket{'s' if n != 1 else ''} loaded" if n else "")

    # ── Run fpocket callbacks ─────────────────────────────────────────────────
    def _refresh_prot_combo():
        prev = cb_prot.currentText()
        cb_prot.blockSignals(True); cb_prot.clear()
        for n in cmd.get_names("objects"):
            if n.startswith(_PREFIX):
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

    def _extra_args() -> list:
        if not g_params.isChecked():
            return []
        args = []
        for flag, le in param_fields.items():
            val = le.text().strip()
            if val:
                args += [flag, val]
        return args

    def _on_fpocket_done(out_dir: str, stem: str):
        b_run_fp.setEnabled(True); pb_run.setVisible(False)
        log_run.appendPlainText(f"✓ Done → {out_dir}")
        e_dir.setText(out_dir)
        ok = _load_from_dir(out_dir)
        _rebuild_table()
        te_res.clear()
        log_run.appendPlainText(
            f"✓ {len(_st.pockets)} pocket(s) loaded." if ok
            else "✗ No pockets found in output directory.")

    def _on_fpocket_failed(msg: str):
        b_run_fp.setEnabled(True); pb_run.setVisible(False)
        log_run.appendPlainText(f"✗ {msg}")
        QtWidgets.QMessageBox.critical(win, "fpocket error", msg)

    def do_run_fpocket():
        fpbin = e_fpbin.text().strip()
        if not fpbin:
            QtWidgets.QMessageBox.warning(win, "FpocketViewer",
                                          "Specify the fpocket binary path."); return
        if rb_file.isChecked():
            pdb_path = e_pdbfile.text().strip()
            if not pdb_path or not os.path.isfile(pdb_path):
                QtWidgets.QMessageBox.warning(win, "FpocketViewer",
                                              "Select a valid PDB file."); return
        else:
            prot = cb_prot.currentText()
            if not prot:
                QtWidgets.QMessageBox.warning(win, "FpocketViewer",
                                              "Select a protein object."); return
            tmpdir = tempfile.mkdtemp(prefix="fpv_")
            pdb_path = os.path.join(tmpdir, f"{prot}.pdb")
            try:
                cmd.save(pdb_path, f"({prot}) and polymer.protein", state=-1)
            except Exception as e:
                QtWidgets.QMessageBox.warning(win, "FpocketViewer",
                                              f"Could not save {prot}: {e}"); return

        log_run.clear()
        log_run.appendPlainText(f"Running fpocket on {os.path.basename(pdb_path)} …")
        b_run_fp.setEnabled(False); pb_run.setVisible(True)
        w = FpocketWorker(fpbin, pdb_path, _extra_args())
        w.log_line.connect(log_run.appendPlainText)
        w.finished.connect(_on_fpocket_done)
        w.failed.connect(_on_fpocket_failed)
        w.start()
        _worker[0] = w

    def do_load_dir():
        path = e_dir.text().strip()
        if not path or not os.path.isdir(path):
            QtWidgets.QMessageBox.warning(win, "FpocketViewer",
                                          "Specify a valid *_out/ directory."); return
        ok = _load_from_dir(path)
        _rebuild_table()
        te_res.clear()
        if not ok:
            lbl_status.setText("No pockets found.")

    def do_clear():
        fpv_clear()
        tw.clearContents(); tw.setRowCount(0)
        te_res.clear()
        lbl_status.setText("")
        log_run.clear()

    # ── Pocket table callbacks ────────────────────────────────────────────────
    def do_selection_changed():
        p = _selected_pocket()
        if p is None:
            _reset_transparency()
        else:
            _highlight_pocket(p.pid)
            te_res.clear()

    def do_zoom():
        p = _selected_pocket()
        if p is None:
            QtWidgets.QMessageBox.information(win, "FpocketViewer",
                                              "Select a pocket row first."); return
        if p.sph_obj in cmd.get_names("objects"):
            cmd.zoom(p.sph_obj, buffer=4.0, animate=1)

    def do_toggle_surface():
        p = _selected_pocket()
        if p is None:
            QtWidgets.QMessageBox.information(win, "FpocketViewer",
                                              "Select a pocket row first."); return
        prot = e_prot_sel.text().strip() or "polymer.protein"
        _toggle_pocket_surface(p, prot)

    def do_show_all():
        for p in _st.pockets:
            if p.sph_obj in cmd.get_names("objects"):
                cmd.enable(p.sph_obj)
        _reset_transparency()

    def do_hide_all():
        for p in _st.pockets:
            if p.sph_obj in cmd.get_names("objects"):
                cmd.disable(p.sph_obj)

    def do_get_residues():
        p = _selected_pocket()
        if p is None:
            QtWidgets.QMessageBox.information(win, "FpocketViewer",
                                              "Select a pocket row first."); return
        prot = e_prot_sel.text().strip() or "polymer.protein"
        residues = _get_lining_residues(p, prot)
        if residues:
            te_res.setPlainText("  ".join(residues))
        else:
            te_res.setPlainText("(none within 4.5 Å of this pocket)")

    # ── Wire up signals ───────────────────────────────────────────────────────
    b_fpbin.clicked.connect(
        lambda: e_fpbin.setText(
            QtWidgets.QFileDialog.getOpenFileName(win, "Select fpocket binary", "", "All (*)")[0]
            or e_fpbin.text()))
    b_refresh.clicked.connect(_refresh_prot_combo)
    b_pdbfile.clicked.connect(
        lambda: e_pdbfile.setText(
            QtWidgets.QFileDialog.getOpenFileName(
                win, "Select PDB file", "", "PDB (*.pdb *.ent);;All (*)")[0]
            or e_pdbfile.text()))
    b_run_fp.clicked.connect(do_run_fpocket)
    b_clear.clicked.connect(do_clear)
    b_dir.clicked.connect(
        lambda: e_dir.setText(
            QtWidgets.QFileDialog.getExistingDirectory(
                win, "Select fpocket *_out/ directory") or e_dir.text()))
    b_load.clicked.connect(do_load_dir)

    tw.selectionModel().selectionChanged.connect(lambda *_: do_selection_changed())
    b_zoom.clicked.connect(do_zoom)
    b_surf.clicked.connect(do_toggle_surface)
    b_show_all.clicked.connect(do_show_all)
    b_hide_all.clicked.connect(do_hide_all)
    b_get_res.clicked.connect(do_get_residues)

    _refresh_prot_combo()
    _rebuild_table()
    win.show()
    return win
