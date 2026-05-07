"""
Pynacea — PyMOL Plugin
=========================
Structure-Based Drug Design workflow plugin for PyMOL.  Four tabs follow
the standard pipeline:

  1. Protein Prep   — clean, fix, protonate, minimize receptor (protprep.py)
  2. Pocket Detect  — fpocket binding-site detection, load pockets into PyMOL
  3. Ligand Prep    — protonate + 3-D embed ligands from SMILES or SDF (obabel)
  4. Docking        — GNINA docking: receptor + ref ligand → ranked pose table
  5. Pose Viewer    — interaction visualisation (H-bonds, pi, salt bridges…)
  6. Design         — load a pose, edit in PyMOL builder, minimize & rescore

Requirements
------------
  protprep.py  — SBDD protein preparation script (openmmdl conda env)
  obabel       — OpenBabel CLI, must be on PATH or configured below
  GNINA        — https://github.com/gnina/gnina

Installation
------------
  Plugin > Plugin Manager > Install New Plugin > choose this file
  — or —
  run /path/to/Pynacea.py   then   pynacea

Authors: Evert J. Homan, PhD; Claude (Anthropic)
Date:    2026-04-27
Version: 0.2
License: MIT
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rdkit import Chem
    _RDKIT = True
except ImportError:
    _RDKIT = False

from pymol import cmd
from pymol.Qt import QtCore, QtGui, QtWidgets

# ─── Constants ────────────────────────────────────────────────────────────────

PLUGIN_VERSION = "0.2"
_IS_WINDOWS    = platform.system() == "Windows"

_CFG_FILE = Path.home() / ".config" / "pynacea.json"

_DEFAULTS: Dict[str, str] = {
    "gnina_path":      os.environ.get("GNINA_PATH", "/opt/gnina/gnina"),
    "protprep_script": os.environ.get("PROTPREP_SCRIPT", ""),
    "openmmdl_python": os.environ.get("OPENMMDL_PYTHON", ""),
    "obabel_path":     os.environ.get("OBABEL_PATH", shutil.which("obabel") or "obabel"),
    "fpocket_path":    os.environ.get("FPOCKET_PATH", shutil.which("fpocket") or "fpocket"),
}

# ─── Persistent config ────────────────────────────────────────────────────────

def _load_cfg() -> Dict[str, str]:
    cfg = dict(_DEFAULTS)
    try:
        if _CFG_FILE.exists():
            cfg.update(json.loads(_CFG_FILE.read_text()))
    except Exception:
        pass
    return cfg


def _save_cfg(cfg: Dict[str, str]):
    try:
        _CFG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CFG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


_cfg = _load_cfg()

# ─── Shared utilities ─────────────────────────────────────────────────────────

def _pymol_objects() -> List[str]:
    return list(cmd.get_object_list() or [])


def _save_sel(sel: str, path: str, state: int = -1) -> bool:
    try:
        cmd.save(path, sel, state=state)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def _sanitize_obj_name(name: str) -> str:
    """Return a PyMOL-safe object name (alphanumeric + underscores only)."""
    name = re.sub(r"[^A-Za-z0-9_.+-]", "_", name.strip())
    if name and name[0].isdigit():
        name = "mol_" + name
    return name or "mol"


def _load_sdf_as_objects(sdf_path: str) -> List[str]:
    """Load each record in an SDF file as a separate named PyMOL object.

    The molecule name (first line of each SDF record) is used as the object
    name after sanitization.  Returns the list of loaded object names.
    """
    loaded: List[str] = []
    if _RDKIT:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            raw = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            name = _sanitize_obj_name(raw or f"mol_{i+1:04d}")
            tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w")
            tmp.close()
            writer = Chem.SDWriter(tmp.name)
            writer.write(mol)
            writer.close()
            cmd.load(tmp.name, name)
            os.unlink(tmp.name)
            loaded.append(name)
    else:
        with open(sdf_path) as fh:
            content = fh.read()
        for i, block in enumerate(content.split("$$$$")):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            raw = lines[0].strip() if lines else ""
            name = _sanitize_obj_name(raw or f"mol_{i+1:04d}")
            tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w")
            tmp.write(block + "\n$$$$\n")
            tmp.close()
            cmd.load(tmp.name, name)
            os.unlink(tmp.name)
            loaded.append(name)
    return loaded


_GNINA_PROPS = ("minimizedAffinity", "CNNscore", "CNNaffinity", "CNN_VS", "CNNaffinity_variance")


def _parse_sdf(sdf_path: str) -> List[Dict[str, Any]]:
    if not _RDKIT:
        return []
    rows: List[Dict[str, Any]] = []
    pose_counter: Dict[str, int] = {}
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        row: Dict[str, Any] = {}
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        row["name"] = name
        pose_counter[name] = pose_counter.get(name, 0) + 1
        row["pose"] = pose_counter[name]
        for prop in _GNINA_PROPS:
            if mol.HasProp(prop):
                try:
                    row[prop] = float(mol.GetProp(prop))
                except ValueError:
                    pass
        row["mol_block"] = Chem.MolToMolBlock(mol) + "$$$$\n"
        rows.append(row)
    return rows


def _fmt(v: Any, d: int = 2) -> str:
    return f"{v:.{d}f}" if isinstance(v, float) else ("—" if v is None else str(v))


def _monofont() -> QtGui.QFont:
    f = QtGui.QFont("Monospace")
    f.setStyleHint(QtGui.QFont.Monospace)
    f.setPointSize(8)
    return f


# ─── WSL support (Windows only) ───────────────────────────────────────────────

_WSL_FILE_FLAGS = {"--receptor", "--ligand", "--autobox_ligand", "--out", "--flex", "--maps"}


def _to_wsl(path: str) -> str:
    path = os.path.abspath(path).replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        return f"/mnt/{path[0].lower()}{path[2:]}"
    return path


def _wrap_wsl(args: List[str]) -> List[str]:
    if not _IS_WINDOWS:
        return args
    out = ["wsl.exe", args[0]]
    i = 1
    while i < len(args):
        tok = args[i]
        if tok in _WSL_FILE_FLAGS and i + 1 < len(args):
            out.append(tok)
            out.append(_to_wsl(args[i + 1]))
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


# ─── Reusable widgets ─────────────────────────────────────────────────────────

class _PathEdit(QtWidgets.QWidget):
    """Label + line-edit + browse button for a file path."""

    def __init__(self, cfg_key: str, label: str,
                 file_filter: str = "All (*)", parent=None):
        super().__init__(parent)
        self._key = cfg_key
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit(_cfg.get(cfg_key, ""))
        btn = QtWidgets.QPushButton("…")
        btn.setMaximumWidth(28)
        btn.clicked.connect(lambda: self._browse(file_filter))
        self.edit.textChanged.connect(self._on_change)
        row.addWidget(self.edit)
        row.addWidget(btn)

    def _browse(self, flt):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", "", flt)
        if p:
            self.edit.setText(p)

    def _on_change(self, text):
        _cfg[self._key] = text
        _save_cfg(_cfg)

    def path(self) -> str:
        return self.edit.text().strip()


class _GninaPath(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit(_cfg.get("gnina_path", ""))
        btn = QtWidgets.QPushButton("…")
        btn.setMaximumWidth(28)
        if _IS_WINDOWS:
            self.edit.setPlaceholderText("/opt/gnina/gnina  (WSL path)")
            btn.setEnabled(False)
        else:
            btn.clicked.connect(self._browse)
        self.edit.textChanged.connect(lambda t: (_cfg.update({"gnina_path": t}), _save_cfg(_cfg)))
        row.addWidget(self.edit)
        row.addWidget(btn)

    def _browse(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select GNINA binary")
        if p:
            self.edit.setText(p)

    def path(self) -> str:
        return self.edit.text().strip()


def _log_widget() -> QtWidgets.QPlainTextEdit:
    w = QtWidgets.QPlainTextEdit()
    w.setReadOnly(True)
    w.setFont(_monofont())
    w.setMaximumBlockCount(3000)
    return w


def _progress_bar() -> QtWidgets.QProgressBar:
    pb = QtWidgets.QProgressBar()
    pb.setRange(0, 0)
    pb.setVisible(False)
    return pb


# ─── Background workers ───────────────────────────────────────────────────────

class _StreamWorker(QtCore.QThread):
    """Base: run a subprocess, stream stdout line by line."""
    log_line = QtCore.Signal(str)
    finished = QtCore.Signal()
    failed   = QtCore.Signal(str)

    def __init__(self, args: List[str], env=None):
        super().__init__()
        self._args = args
        self._env  = env

    def run(self):
        try:
            proc = subprocess.Popen(
                self._args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=self._env,
            )
            for line in proc.stdout:
                self.log_line.emit(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                self.finished.emit()
            else:
                self.failed.emit(f"Exit code {proc.returncode}")
        except FileNotFoundError:
            self.failed.emit(f"Binary not found: {self._args[0]}")
        except Exception as exc:
            self.failed.emit(str(exc))


class GninaWorker(QtCore.QThread):
    log_line = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    failed   = QtCore.Signal(str)

    def __init__(self, args: List[str], output_path: str):
        super().__init__()
        self._args   = args
        self._output = output_path

    def run(self):
        try:
            proc = subprocess.Popen(
                self._args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                self.log_line.emit(line.rstrip())
            proc.wait()
            if proc.returncode == 0 and os.path.exists(self._output):
                self.finished.emit(self._output)
            else:
                self.failed.emit(f"GNINA exited with code {proc.returncode}")
        except FileNotFoundError:
            self.failed.emit(f"GNINA binary not found: {self._args[0]}")
        except Exception as exc:
            self.failed.emit(str(exc))


# ─── Shared PDB inspector (no external deps needed) ───────────────────────────

def _inspect_pdb(path: str) -> Dict[str, Any]:
    """Parse a PDB file and return chains + HET groups without biopython."""
    chains: Dict[str, int] = {}   # chain_id → std residue count
    het: Dict[str, List[str]]  = {}  # resname → ["A:101", ...]
    seen_std:  set = set()
    seen_het:  set = set()

    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "ATOM":
                chain = line[21]
                resi  = line[22:26].strip()
                key   = (chain, resi)
                if key not in seen_std:
                    seen_std.add(key)
                    chains[chain] = chains.get(chain, 0) + 1
            elif rec == "HETATM":
                resname = line[17:20].strip()
                chain   = line[21]
                resi    = line[22:26].strip()
                if resname in ("HOH", "WAT", "DOD"):
                    continue
                key = (resname, chain, resi)
                if key not in seen_het:
                    seen_het.add(key)
                    het.setdefault(resname, []).append(f"{chain}:{resi}")

    return {"chains": chains, "het": het}


# ─── Tab 1: Protein Preparation ───────────────────────────────────────────────

class ProtprepWorker(QtCore.QThread):
    log_line = QtCore.Signal(str)
    finished = QtCore.Signal(str, str)   # prepared_pdb, ref_ligand_sdf (may be "")
    failed   = QtCore.Signal(str)

    def __init__(self, python: str, script: str, args: List[str], out_pdb: str, out_sdf: str):
        super().__init__()
        self._python  = python
        self._script  = script
        self._args    = args
        self._out_pdb = out_pdb
        self._out_sdf = out_sdf   # expected ref-ligand SDF (may not exist)

    def run(self):
        cmd_args = [self._python, self._script] + self._args
        try:
            proc = subprocess.Popen(
                cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                self.log_line.emit(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                self.failed.emit(f"protprep.py exited with code {proc.returncode}")
                return
            # prefer _minimized.pdb if it exists
            stem    = Path(self._out_pdb).stem.replace("_prepared", "")
            mini    = Path(self._out_pdb).parent / f"{stem}_minimized.pdb"
            final   = str(mini) if mini.exists() else self._out_pdb
            ref_sdf = self._out_sdf if os.path.exists(self._out_sdf) else ""
            self.finished.emit(final, ref_sdf)
        except FileNotFoundError:
            self.failed.emit(f"Python not found: {self._python}")
        except Exception as exc:
            self.failed.emit(str(exc))


class ProtprepPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmpdir  = tempfile.mkdtemp(prefix="pynacea_prot_")
        self._worker: Optional[ProtprepWorker] = None
        self._info: Dict[str, Any] = {}
        self._het_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self._chain_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self._input_pdb: str = ""
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)

        # ── Tool paths ────────────────────────────────────────────────────────
        tools_grp = QtWidgets.QGroupBox("Tools")
        tl = QtWidgets.QFormLayout(tools_grp)
        tl.setLabelAlignment(QtCore.Qt.AlignRight)
        self._py_edit = _PathEdit("openmmdl_python", "Python",
                                  "Python (python python3 *);;All (*)")
        self._py_edit.edit.setPlaceholderText("path to openmmdl conda python")
        self._sc_edit = _PathEdit("protprep_script", "Script",
                                  "Python scripts (*.py);;All (*)")
        self._sc_edit.edit.setPlaceholderText("path to protprep.py")
        tl.addRow("Python:", self._py_edit)
        tl.addRow("protprep.py:", self._sc_edit)
        root.addWidget(tools_grp)

        # ── Input ─────────────────────────────────────────────────────────────
        inp_grp = QtWidgets.QGroupBox("Input")
        il = QtWidgets.QVBoxLayout(inp_grp)

        id_row = QtWidgets.QHBoxLayout()
        self._pdbid_edit = QtWidgets.QLineEdit()
        self._pdbid_edit.setPlaceholderText("PDB ID  e.g. 4HHB")
        self._pdbid_edit.setMaximumWidth(90)
        fetch_btn = QtWidgets.QPushButton("Fetch")
        fetch_btn.clicked.connect(self._fetch_pdb)
        id_row.addWidget(QtWidgets.QLabel("PDB ID:"))
        id_row.addWidget(self._pdbid_edit)
        id_row.addWidget(fetch_btn)
        id_row.addStretch()
        il.addLayout(id_row)

        file_row = QtWidgets.QHBoxLayout()
        self._file_edit = QtWidgets.QLineEdit()
        self._file_edit.setPlaceholderText("or browse to a local PDB file")
        browse_btn = QtWidgets.QPushButton("…")
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._browse_pdb)
        inspect_btn = QtWidgets.QPushButton("Inspect")
        inspect_btn.clicked.connect(self._inspect)
        file_row.addWidget(self._file_edit)
        file_row.addWidget(browse_btn)
        file_row.addWidget(inspect_btn)
        il.addLayout(file_row)
        root.addWidget(inp_grp)

        # ── Structure info (populated after inspect) ──────────────────────────
        self._info_grp = QtWidgets.QGroupBox("Structure")
        info_l = QtWidgets.QVBoxLayout(self._info_grp)

        chain_lbl = QtWidgets.QLabel("Chains:")
        self._chain_area = QtWidgets.QWidget()
        self._chain_row  = QtWidgets.QHBoxLayout(self._chain_area)
        self._chain_row.setContentsMargins(0, 0, 0, 0)

        het_lbl = QtWidgets.QLabel("HETATM groups (check = keep in receptor):")
        self._het_area  = QtWidgets.QWidget()
        self._het_grid  = QtWidgets.QGridLayout(self._het_area)
        self._het_grid.setContentsMargins(0, 0, 0, 0)

        info_l.addWidget(chain_lbl)
        info_l.addWidget(self._chain_area)
        info_l.addWidget(het_lbl)
        info_l.addWidget(self._het_area)
        self._info_grp.setEnabled(False)
        root.addWidget(self._info_grp)

        # ── Options ───────────────────────────────────────────────────────────
        opt_grp = QtWidgets.QGroupBox("Options")
        ol = QtWidgets.QFormLayout(opt_grp)
        ol.setLabelAlignment(QtCore.Qt.AlignRight)
        self._ph_sp = QtWidgets.QDoubleSpinBox()
        self._ph_sp.setRange(0, 14); self._ph_sp.setValue(7.4)
        self._ph_sp.setSingleStep(0.1); self._ph_sp.setDecimals(1)
        self._min_chk  = QtWidgets.QCheckBox("Restrained vacuum minimization")
        self._min_chk.setChecked(True)
        self._nopqr_chk = QtWidgets.QCheckBox("Skip pdb2pqr (use PDBFixer H placement)")
        self._nopqr_chk.setChecked(False)
        ol.addRow("pH:", self._ph_sp)
        ol.addRow("", self._min_chk)
        ol.addRow("", self._nopqr_chk)
        root.addWidget(opt_grp)

        # ── Ref ligand to extract ─────────────────────────────────────────────
        ref_grp = QtWidgets.QGroupBox("Reference ligand to extract")
        rl = QtWidgets.QHBoxLayout(ref_grp)
        self._ref_combo = QtWidgets.QComboBox()
        self._ref_combo.addItem("(none)")
        self._ref_combo.setMinimumWidth(120)
        rl.addWidget(QtWidgets.QLabel("Extract:"))
        rl.addWidget(self._ref_combo)
        rl.addStretch()
        root.addWidget(ref_grp)

        # ── Run ───────────────────────────────────────────────────────────────
        self._run_btn = QtWidgets.QPushButton("Prepare Protein")
        self._run_btn.clicked.connect(self._run)
        self._run_btn.setEnabled(False)
        self._progress = _progress_bar()
        root.addWidget(self._run_btn)
        root.addWidget(self._progress)

        self._log = _log_widget()
        root.addWidget(self._log)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse_pdb(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select PDB file", "", "PDB (*.pdb *.ent);;All (*)"
        )
        if p:
            self._file_edit.setText(p)

    def _fetch_pdb(self):
        pdb_id = self._pdbid_edit.text().strip().upper()
        if not pdb_id:
            return
        out = os.path.join(self._tmpdir, f"{pdb_id}.pdb")
        self._log.appendPlainText(f"Fetching {pdb_id} from RCSB…")
        try:
            import urllib.request
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{pdb_id}.pdb", out
            )
            self._file_edit.setText(out)
            self._log.appendPlainText(f"Saved to {out}")
            self._inspect_path(out)
        except Exception as e:
            self._log.appendPlainText(f"Fetch failed: {e}")

    def _inspect(self):
        p = self._file_edit.text().strip()
        if not p or not os.path.exists(p):
            QtWidgets.QMessageBox.warning(self, "Pynacea", "Specify a valid PDB file first.")
            return
        self._inspect_path(p)

    def _inspect_path(self, path: str):
        self._input_pdb = path
        try:
            self._info = _inspect_pdb(path)
        except Exception as e:
            self._log.appendPlainText(f"Inspect error: {e}")
            return
        self._populate_info()
        self._info_grp.setEnabled(True)
        self._run_btn.setEnabled(True)
        chains = self._info.get("chains", {})
        het    = self._info.get("het", {})
        self._log.appendPlainText(
            f"Inspected {os.path.basename(path)}: "
            f"{len(chains)} chain(s), {sum(chains.values())} std residues, "
            f"{len(het)} HET group type(s)"
        )

    def _populate_info(self):
        # Chains
        for i in reversed(range(self._chain_row.count())):
            self._chain_row.itemAt(i).widget().deleteLater()
        self._chain_checks.clear()
        for ch, n in sorted(self._info.get("chains", {}).items()):
            cb = QtWidgets.QCheckBox(f"{ch} ({n})")
            cb.setChecked(True)
            self._chain_checks[ch] = cb
            self._chain_row.addWidget(cb)
        self._chain_row.addStretch()

        # HET groups
        for i in reversed(range(self._het_grid.count())):
            w = self._het_grid.itemAt(i).widget()
            if w:
                w.deleteLater()
        self._het_checks.clear()
        self._ref_combo.clear()
        self._ref_combo.addItem("(none)")
        het = self._info.get("het", {})
        for col, (resname, locs) in enumerate(sorted(het.items())):
            tip = ", ".join(locs[:6]) + ("…" if len(locs) > 6 else "")
            cb  = QtWidgets.QCheckBox(f"{resname}")
            cb.setToolTip(tip)
            cb.setChecked(False)
            self._het_checks[resname] = cb
            self._het_grid.addWidget(cb, col // 4, col % 4)
            self._ref_combo.addItem(resname)

    def _run(self):
        python = self._py_edit.path()
        script = self._sc_edit.path()
        if not python or not os.path.exists(python):
            QtWidgets.QMessageBox.warning(self, "Pynacea",
                "Configure the openmmdl Python path first.")
            return
        if not script or not os.path.exists(script):
            QtWidgets.QMessageBox.warning(self, "Pynacea",
                "Configure the protprep.py script path first.")
            return

        stem     = Path(self._input_pdb).stem
        out_pdb  = os.path.join(self._tmpdir, f"{stem}_prepared.pdb")
        out_sdf  = os.path.join(self._tmpdir, f"{stem}_prepared_ligand.sdf")

        chains_sel = [ch for ch, cb in self._chain_checks.items() if cb.isChecked()]
        keep_het   = [rn for rn, cb in self._het_checks.items() if cb.isChecked()]
        ref_lig    = self._ref_combo.currentText()
        if ref_lig != "(none)" and ref_lig not in keep_het:
            keep_het.append(ref_lig)

        args = [
            "--input",  self._input_pdb,
            "--output", out_pdb,
            "--ph",     str(self._ph_sp.value()),
        ]
        if chains_sel and len(chains_sel) < len(self._chain_checks):
            args += ["--chain"] + chains_sel
        if keep_het:
            args += ["--keep-het"] + keep_het
        if self._min_chk.isChecked():
            args.append("--minimize")
        if self._nopqr_chk.isChecked():
            args.append("--no-pdb2pqr")

        self._log.clear()
        self._log.appendPlainText("$ " + " ".join([python, script] + args))
        self._log.appendPlainText("─" * 60)
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)

        self._worker = ProtprepWorker(python, script, args, out_pdb, out_sdf)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _done(self, pdb_path: str, sdf_path: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        stem = Path(pdb_path).stem
        cmd.load(pdb_path, stem)
        cmd.spectrum("count", "rainbow", f"{stem} and elem C")
        self._log.appendPlainText(f"\n✓ Loaded '{stem}' into PyMOL")
        if sdf_path:
            ref_stem = Path(sdf_path).stem
            cmd.load(sdf_path, ref_stem)
            self._log.appendPlainText(f"✓ Loaded reference ligand '{ref_stem}'")

    def _fail(self, msg: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._log.appendPlainText(f"\n✗ {msg}")
        QtWidgets.QMessageBox.critical(self, "Protein preparation failed", msg)

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate(); self._worker.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ─── Tab 2: Pocket Detection ─────────────────────────────────────────────────

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
    "cent. of mass - alpha sphere max dist": "Cent. of mass - Alpha Sphere max dist",
    "flexibility":                           "Flexibility",
}

_FPOCKET_COLS = ["Pocket"] + list(_FPOCKET_COL_MAP.values())


class _NumericItem(QtWidgets.QTableWidgetItem):
    """QTableWidgetItem that sorts numerically."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except (ValueError, TypeError):
            return self.text() < other.text()


def _parse_fpocket_info(info_path: str) -> List[Dict[str, Any]]:
    """Parse fpocket *_info.txt into a list of pocket dicts."""
    pockets: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    with open(info_path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"^Pocket\s+(\d+)\s*:", line, re.IGNORECASE)
            if m:
                if cur is not None:
                    pockets.append(cur)
                cur = {"Pocket": int(m.group(1))}
                continue
            if cur is not None and ":" in line:
                raw_k, _, raw_v = line.partition(":")
                k_low = raw_k.strip().lower()
                col = _FPOCKET_COL_MAP.get(k_low)
                if col:
                    try:
                        cur[col] = float(raw_v.strip())
                    except ValueError:
                        pass
    if cur is not None:
        pockets.append(cur)
    return pockets


class FpocketWorker(QtCore.QThread):
    log_line = QtCore.Signal(str)
    finished = QtCore.Signal(str, str)   # out_dir, stem
    failed   = QtCore.Signal(str)

    def __init__(self, fpocket: str, pdb_path: str):
        super().__init__()
        self._fpocket  = fpocket
        self._pdb_path = pdb_path

    def run(self):
        try:
            stem    = Path(self._pdb_path).stem
            out_dir = Path(self._pdb_path).parent / f"{stem}_out"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            proc = subprocess.Popen(
                [self._fpocket, "-f", self._pdb_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                self.log_line.emit(line.rstrip())
            proc.wait()
            if proc.returncode == 0 and out_dir.exists():
                self.finished.emit(str(out_dir), stem)
            else:
                self.failed.emit(f"fpocket exited with code {proc.returncode}")
        except FileNotFoundError:
            self.failed.emit(f"fpocket binary not found: {self._fpocket}")
        except Exception as exc:
            self.failed.emit(str(exc))


class FpocketPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmpdir = tempfile.mkdtemp(prefix="pynacea_fpocket_")
        self._worker: Optional[FpocketWorker] = None
        self._out_dir = ""
        self._stem    = ""
        self._pockets: List[Dict[str, Any]] = []
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # Paths
        path_grp = QtWidgets.QGroupBox("fpocket")
        ol = QtWidgets.QFormLayout(path_grp)
        ol.setLabelAlignment(QtCore.Qt.AlignRight)
        self._fp_edit = _PathEdit("fpocket_path", "fpocket")
        ol.addRow("fpocket:", self._fp_edit)
        root.addWidget(path_grp)

        # Input protein
        prot_grp = QtWidgets.QGroupBox("Protein")
        pl = QtWidgets.QVBoxLayout(prot_grp)

        mode_row = QtWidgets.QHBoxLayout()
        self._rb_obj  = QtWidgets.QRadioButton("PyMOL object")
        self._rb_file = QtWidgets.QRadioButton("PDB file")
        self._rb_obj.setChecked(True)
        mode_row.addWidget(self._rb_obj)
        mode_row.addWidget(self._rb_file)
        mode_row.addStretch()
        pl.addLayout(mode_row)

        self._prot_stack = QtWidgets.QStackedWidget()

        # Page 0 — PyMOL object
        obj_page = QtWidgets.QWidget()
        obj_row  = QtWidgets.QHBoxLayout(obj_page)
        obj_row.setContentsMargins(0, 0, 0, 0)
        self._prot_combo = QtWidgets.QComboBox()
        self._prot_combo.setMinimumWidth(160)
        ref_btn = QtWidgets.QPushButton("Refresh")
        ref_btn.setMaximumWidth(64)
        ref_btn.clicked.connect(self._refresh_objects)
        obj_row.addWidget(self._prot_combo, 1)
        obj_row.addWidget(ref_btn)
        self._prot_stack.addWidget(obj_page)

        # Page 1 — PDB file on disk
        file_page = QtWidgets.QWidget()
        file_row  = QtWidgets.QHBoxLayout(file_page)
        file_row.setContentsMargins(0, 0, 0, 0)
        self._pdb_file_edit = QtWidgets.QLineEdit()
        self._pdb_file_edit.setPlaceholderText("path/to/protein.pdb")
        pdb_browse_btn = QtWidgets.QPushButton("…")
        pdb_browse_btn.setMaximumWidth(28)
        pdb_browse_btn.clicked.connect(self._browse_pdb_file)
        file_row.addWidget(self._pdb_file_edit, 1)
        file_row.addWidget(pdb_browse_btn)
        self._prot_stack.addWidget(file_page)

        self._rb_obj.toggled.connect(
            lambda on: self._prot_stack.setCurrentIndex(0 if on else 1))
        pl.addWidget(self._prot_stack)
        root.addWidget(prot_grp)

        # Run
        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = QtWidgets.QPushButton("Detect Pockets")
        btn_row.addWidget(self._run_btn)
        self._run_btn.clicked.connect(self._run)
        root.addLayout(btn_row)

        self._progress = _progress_bar()
        root.addWidget(self._progress)

        # Results table
        res_grp = QtWidgets.QGroupBox("Detected pockets")
        rl = QtWidgets.QVBoxLayout(res_grp)
        self._table = QtWidgets.QTableWidget(0, len(_FPOCKET_COLS))
        self._table.setHorizontalHeaderLabels(_FPOCKET_COLS)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(18)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        rl.addWidget(self._table)

        load_row = QtWidgets.QHBoxLayout()
        self._load_btn = QtWidgets.QPushButton("Load pocket spheres into PyMOL")
        self._load_btn.setEnabled(False)
        load_row.addWidget(self._load_btn)
        self._load_btn.clicked.connect(self._load_pocket)
        rl.addLayout(load_row)
        root.addWidget(res_grp)

        self._log = _log_widget()
        root.addWidget(self._log)

        self._refresh_objects()
        self._table.itemSelectionChanged.connect(
            lambda: self._load_btn.setEnabled(bool(self._table.selectedItems())))

    def _refresh_objects(self):
        prev = self._prot_combo.currentText()
        self._prot_combo.blockSignals(True)
        self._prot_combo.clear()
        for n in _pymol_objects():
            self._prot_combo.addItem(n)
        idx = self._prot_combo.findText(prev)
        if idx >= 0:
            self._prot_combo.setCurrentIndex(idx)
        self._prot_combo.blockSignals(False)

    def _browse_pdb_file(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select PDB file", "", "PDB (*.pdb *.ent);;All (*)")
        if p:
            self._pdb_file_edit.setText(p)

    def _run(self):
        fpocket = self._fp_edit.path()
        if not fpocket:
            QtWidgets.QMessageBox.warning(self, "Pynacea", "Set the fpocket path first.")
            return

        if self._rb_file.isChecked():
            pdb_path = self._pdb_file_edit.text().strip()
            if not pdb_path or not os.path.exists(pdb_path):
                QtWidgets.QMessageBox.warning(self, "Pynacea", "Select a valid PDB file.")
                return
            label = os.path.basename(pdb_path)
        else:
            prot = self._prot_combo.currentText()
            if not prot:
                QtWidgets.QMessageBox.warning(self, "Pynacea", "Select a protein object first.")
                return
            pdb_path = os.path.join(self._tmpdir, f"{prot}.pdb")
            if not _save_sel(prot, pdb_path):
                QtWidgets.QMessageBox.warning(self, "Pynacea", f"Could not save '{prot}' as PDB.")
                return
            label = prot

        self._log.clear()
        self._log.appendPlainText(f"Running fpocket on '{label}'…")
        self._run_btn.setEnabled(False)
        self._load_btn.setEnabled(False)
        self._table.setRowCount(0)
        self._pockets.clear()
        self._progress.setRange(0, 0)
        self._progress.setVisible(True)

        self._worker = FpocketWorker(fpocket, pdb_path)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _done(self, out_dir: str, stem: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._out_dir = out_dir
        self._stem = stem
        info_path = os.path.join(out_dir, f"{stem}_info.txt")
        if not os.path.exists(info_path):
            self._log.appendPlainText(f"✗ Info file not found: {info_path}")
            return
        self._pockets = _parse_fpocket_info(info_path)
        self._log.appendPlainText(f"✓ {len(self._pockets)} pocket(s) detected")
        self._fill_table()

    def _fail(self, msg: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._log.appendPlainText(f"\n✗ {msg}")
        QtWidgets.QMessageBox.critical(self, "Pocket detection failed", msg)

    def _fill_table(self):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._pockets))
        for r, p in enumerate(self._pockets):
            for c, col in enumerate(_FPOCKET_COLS):
                v = p.get(col, "")
                if col == "Pocket":
                    txt = str(int(v)) if isinstance(v, (int, float)) else str(v)
                else:
                    txt = _fmt(v) if isinstance(v, float) else ("" if v == "" else str(v))
                item = _NumericItem(txt)
                if c > 0:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)
        score_col = _FPOCKET_COLS.index("Score")
        self._table.sortByColumn(score_col, QtCore.Qt.DescendingOrder)

    def _load_pocket(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows or not self._out_dir:
            return
        r      = rows[0].row()
        pocket = self._pockets[r]
        n      = int(pocket["Pocket"])
        pockets_dir = os.path.join(self._out_dir, "pockets")
        vert = os.path.join(pockets_dir, f"pocket{n}_vert.pqr")

        if not os.path.exists(vert):
            QtWidgets.QMessageBox.warning(
                self, "Pynacea", f"Pocket {n} vertex file not found:\n{vert}")
            return

        sph_name = f"pocket{n}_spheres"
        cmd.load(vert, sph_name)
        cmd.hide("everything", sph_name)
        cmd.show("spheres", sph_name)
        cmd.set("sphere_scale", 0.3, sph_name)
        cmd.set("sphere_transparency", 0.1, sph_name)
        cmd.color(n + 1, sph_name)
        self._log.appendPlainText(f"✓ Loaded pocket {n} alpha spheres as '{sph_name}'")

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate(); self._worker.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ─── Tab 3: Ligand Preparation ────────────────────────────────────────────────

class LigprepWorker(QtCore.QThread):
    """Prepare ligands from SMILES using obabel (2-step: protonate → 3D)."""
    log_line  = QtCore.Signal(str)
    progress  = QtCore.Signal(int, int)   # done, total
    finished  = QtCore.Signal(str)        # output SDF path
    failed    = QtCore.Signal(str)

    def __init__(self, obabel: str, entries: List[tuple], out_sdf: str,
                 ph: float, tmpdir: str):
        super().__init__()
        self._obabel  = obabel
        self._entries = entries   # list of (smiles, name)
        self._out     = out_sdf
        self._ph      = ph
        self._tmp     = tmpdir

    def run(self):
        blocks = []
        total  = len(self._entries)
        for i, (smi, name) in enumerate(self._entries):
            block, err = self._prepare_one(smi, name, i)
            if block:
                blocks.append(block)
                self.log_line.emit(f"[{i+1}/{total}] ✓ {name}")
            else:
                self.log_line.emit(f"[{i+1}/{total}] ✗ {name}: {err}")
            self.progress.emit(i + 1, total)

        if not blocks:
            self.failed.emit("No ligands prepared successfully.")
            return
        with open(self._out, "w") as fh:
            fh.write("".join(blocks))
        self.finished.emit(self._out)

    def _prepare_one(self, smi: str, name: str, idx: int):
        wd      = os.path.join(self._tmp, f"lig_{idx}")
        os.makedirs(wd, exist_ok=True)
        smi_in  = os.path.join(wd, "in.smi")
        smi_pro = os.path.join(wd, "pro.smi")
        sdf_out = os.path.join(wd, "out.sdf")
        try:
            with open(smi_in, "w") as fh:
                fh.write(f"{smi} {name}\n")

            # Step 1: protonate
            r1 = subprocess.run(
                [self._obabel, smi_in, "-O", smi_pro, "-r", "-p", str(self._ph)],
                capture_output=True, text=True, timeout=30
            )
            if not os.path.exists(smi_pro) or os.path.getsize(smi_pro) == 0:
                return None, f"protonation failed: {r1.stderr[:80]}"

            # Step 2: 3D embed + minimize
            r2 = subprocess.run(
                [self._obabel, smi_pro, "-O", sdf_out,
                 "--gen3d", "medium", "--minimize", "--ff", "MMFF94s",
                 "--crit", "1e-7", "--sd"],
                capture_output=True, text=True, timeout=120
            )
            if not os.path.exists(sdf_out) or os.path.getsize(sdf_out) < 50:
                # fallback: no minimize
                subprocess.run(
                    [self._obabel, smi_pro, "-O", sdf_out, "--gen3d", "best"],
                    capture_output=True, text=True, timeout=60
                )
            if not os.path.exists(sdf_out) or os.path.getsize(sdf_out) < 50:
                return None, f"3D generation failed: {r2.stderr[:80]}"

            with open(sdf_out) as fh:
                block = fh.read()
            if "$$$$" not in block or "M  END" not in block:
                return None, "incomplete SDF"

            # Replace first line with molecule name
            lines      = block.split("\n")
            lines[0]   = name
            block      = "\n".join(lines)
            # Strip obabel-added properties (GNINA will add its own)
            block = re.sub(r"> <[^>]+>\n[^\n]*\n\n", "", block)
            if not block.strip().endswith("$$$$"):
                block = block.rstrip() + "\n$$$$\n"
            return block, None

        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception as e:
            return None, str(e)


def _parse_smiles_input(text: str) -> List[tuple]:
    """Parse multi-line SMILES text → [(smiles, name), ...]."""
    entries = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t,\s]+", line, maxsplit=1)
        smi   = parts[0]
        name  = parts[1].strip() if len(parts) > 1 else f"ligand_{i+1:04d}"
        entries.append((smi, name))
    return entries


class LigprepPanel(QtWidgets.QWidget):
    objects_loaded = QtCore.Signal(list)   # emitted after ligprep loads PyMOL objects

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmpdir = tempfile.mkdtemp(prefix="pynacea_lig_")
        self._worker: Optional[LigprepWorker] = None
        self._out_sdf = ""
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)

        # Tool path
        tool_grp = QtWidgets.QGroupBox("Tools")
        tl = QtWidgets.QFormLayout(tool_grp)
        tl.setLabelAlignment(QtCore.Qt.AlignRight)
        self._ob_edit = _PathEdit("obabel_path", "obabel", "obabel;;All (*)")
        self._ob_edit.edit.setPlaceholderText("obabel  (must be on PATH or set here)")
        tl.addRow("obabel:", self._ob_edit)
        root.addWidget(tool_grp)

        # Input mode toggle
        mode_grp = QtWidgets.QGroupBox("Input")
        ml = QtWidgets.QVBoxLayout(mode_grp)

        mode_row = QtWidgets.QHBoxLayout()
        self._rb_smiles = QtWidgets.QRadioButton("SMILES")
        self._rb_sdf    = QtWidgets.QRadioButton("SDF file")
        self._rb_smiles.setChecked(True)
        mode_row.addWidget(self._rb_smiles)
        mode_row.addWidget(self._rb_sdf)
        mode_row.addStretch()
        ml.addLayout(mode_row)

        self._stack = QtWidgets.QStackedWidget()

        # Page 0: SMILES text area
        smiles_page = QtWidgets.QWidget()
        sp_l = QtWidgets.QVBoxLayout(smiles_page)
        sp_l.setContentsMargins(0, 0, 0, 0)
        self._smiles_edit = QtWidgets.QPlainTextEdit()
        self._smiles_edit.setPlaceholderText(
            "One SMILES per line.  Name is optional:\n"
            "CC(=O)Oc1ccccc1C(=O)O  aspirin\n"
            "c1ccccc1  benzene"
        )
        self._smiles_edit.setFont(_monofont())
        self._smiles_edit.setMaximumHeight(120)
        sp_l.addWidget(self._smiles_edit)
        self._stack.addWidget(smiles_page)

        # Page 1: SDF file
        sdf_page = QtWidgets.QWidget()
        sdf_l = QtWidgets.QHBoxLayout(sdf_page)
        sdf_l.setContentsMargins(0, 0, 0, 0)
        self._sdf_edit = QtWidgets.QLineEdit()
        self._sdf_edit.setPlaceholderText("ligands.sdf")
        sdf_btn = QtWidgets.QPushButton("…")
        sdf_btn.setMaximumWidth(28)
        sdf_btn.clicked.connect(self._browse_sdf)
        sdf_l.addWidget(self._sdf_edit)
        sdf_l.addWidget(sdf_btn)
        self._stack.addWidget(sdf_page)

        self._rb_smiles.toggled.connect(lambda on: self._stack.setCurrentIndex(0 if on else 1))
        ml.addWidget(self._stack)
        root.addWidget(mode_grp)

        # Options
        opt_grp = QtWidgets.QGroupBox("Options")
        ol = QtWidgets.QFormLayout(opt_grp)
        ol.setLabelAlignment(QtCore.Qt.AlignRight)
        self._ph_sp = QtWidgets.QDoubleSpinBox()
        self._ph_sp.setRange(0, 14); self._ph_sp.setValue(7.4)
        self._ph_sp.setSingleStep(0.1); self._ph_sp.setDecimals(1)
        self._load_chk = QtWidgets.QCheckBox("Load output SDF into PyMOL when done")
        self._load_chk.setChecked(True)
        self._out_edit = QtWidgets.QLineEdit()
        self._out_edit.setPlaceholderText("output_ligands.sdf  (leave blank for auto)")
        ol.addRow("pH:", self._ph_sp)
        ol.addRow("Output SDF:", self._out_edit)
        ol.addRow("", self._load_chk)
        root.addWidget(opt_grp)

        # Run
        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn  = QtWidgets.QPushButton("Prepare Ligands")
        self._save_btn = QtWidgets.QPushButton("Save SDF…")
        self._save_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        self._save_btn.clicked.connect(self._save_output)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._save_btn)
        root.addLayout(btn_row)

        self._progress = _progress_bar()
        self._progress.setRange(0, 100)
        root.addWidget(self._progress)

        self._log = _log_widget()
        root.addWidget(self._log)

    def _browse_sdf(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select SDF file", "", "SDF (*.sdf *.sdf.gz);;All (*)"
        )
        if p:
            self._sdf_edit.setText(p)

    def _run(self):
        obabel = self._ob_edit.path()
        if not obabel:
            QtWidgets.QMessageBox.warning(self, "Pynacea", "Set the obabel path first.")
            return

        if self._rb_smiles.isChecked():
            text    = self._smiles_edit.toPlainText().strip()
            entries = _parse_smiles_input(text)
            if not entries:
                QtWidgets.QMessageBox.warning(self, "Pynacea", "Enter at least one SMILES.")
                return
        else:
            sdf_in = self._sdf_edit.text().strip()
            if not sdf_in or not os.path.exists(sdf_in):
                QtWidgets.QMessageBox.warning(self, "Pynacea", "Select a valid SDF file.")
                return
            # Re-prepare from SMILES extracted by RDKit, or pass SDF through obabel
            if _RDKIT:
                suppl   = Chem.SDMolSupplier(sdf_in, removeHs=True)
                entries = []
                for i, mol in enumerate(suppl):
                    if mol is None:
                        continue
                    smi  = Chem.MolToSmiles(mol)
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1:04d}"
                    entries.append((smi, name))
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Pynacea",
                    "SDF input mode requires RDKit to extract SMILES.\n"
                    "Install RDKit or paste SMILES directly."
                )
                return
            if not entries:
                QtWidgets.QMessageBox.warning(self, "Pynacea", "No valid molecules in SDF.")
                return

        out_path = self._out_edit.text().strip()
        if not out_path:
            out_path = os.path.join(self._tmpdir, "prepared_ligands.sdf")
        self._out_sdf = out_path

        self._log.clear()
        self._log.appendPlainText(f"Preparing {len(entries)} ligand(s) at pH {self._ph_sp.value()}")
        self._log.appendPlainText("─" * 60)
        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._progress.setRange(0, len(entries))
        self._progress.setValue(0)
        self._progress.setVisible(True)

        self._worker = LigprepWorker(
            obabel, entries, out_path, self._ph_sp.value(), self._tmpdir
        )
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.progress.connect(
            lambda done, total: self._progress.setValue(done)
        )
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _done(self, sdf_path: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._save_btn.setEnabled(True)
        n = sdf_path and sum(1 for l in open(sdf_path) if l.strip() == "$$$$")
        self._log.appendPlainText(f"\n✓ {n} ligand(s) written to {sdf_path}")
        if self._load_chk.isChecked():
            names = _load_sdf_as_objects(sdf_path)
            if names:
                self._log.appendPlainText(
                    f"✓ Loaded {len(names)} object(s) into PyMOL: " + ", ".join(names))
                self.objects_loaded.emit(names)
            else:
                self._log.appendPlainText("⚠ Could not load any ligands into PyMOL")

    def _fail(self, msg: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._log.appendPlainText(f"\n✗ {msg}")
        QtWidgets.QMessageBox.critical(self, "Ligand preparation failed", msg)

    def _save_output(self):
        if not (self._out_sdf and os.path.exists(self._out_sdf)):
            return
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save prepared ligands", "prepared_ligands.sdf",
            "SDF (*.sdf);;All (*)"
        )
        if dest:
            shutil.copy2(self._out_sdf, dest)
            self._log.appendPlainText(f"✓ Saved to {dest}")

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate(); self._worker.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ─── Tab 4: Docking ───────────────────────────────────────────────────────────

class DockingPanel(QtWidgets.QWidget):
    pose_loaded = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmpdir  = tempfile.mkdtemp(prefix="pynacea_dock_")
        self._worker: Optional[GninaWorker] = None
        self._results: List[Dict[str, Any]] = []
        self._out_sdf = ""
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.rec_box = QtWidgets.QComboBox()
        self.ref_box = QtWidgets.QComboBox()
        form.addRow("Receptor:", self.rec_box)
        form.addRow("Ref ligand (box):", self.ref_box)

        lig_src_w = QtWidgets.QWidget()
        lig_src_l = QtWidgets.QVBoxLayout(lig_src_w)
        lig_src_l.setContentsMargins(0, 0, 0, 0)
        lig_src_l.setSpacing(2)

        rb_row = QtWidgets.QHBoxLayout()
        self._rb_lig_file = QtWidgets.QRadioButton("File")
        self._rb_lig_file.setChecked(True)
        self._rb_lig_obj  = QtWidgets.QRadioButton("PyMOL object")
        rb_row.addWidget(self._rb_lig_file)
        rb_row.addWidget(self._rb_lig_obj)
        rb_row.addStretch()
        lig_src_l.addLayout(rb_row)

        self._lig_stack = QtWidgets.QStackedWidget()

        lig_file_w = QtWidgets.QWidget()
        lig_file_r = QtWidgets.QHBoxLayout(lig_file_w)
        lig_file_r.setContentsMargins(0, 0, 0, 0)
        self.lig_edit = QtWidgets.QLineEdit()
        self.lig_edit.setPlaceholderText("ligands.sdf")
        lig_btn = QtWidgets.QPushButton("…"); lig_btn.setMaximumWidth(28)
        lig_btn.clicked.connect(self._browse_lig)
        lig_file_r.addWidget(self.lig_edit); lig_file_r.addWidget(lig_btn)
        self._lig_stack.addWidget(lig_file_w)

        self.lig_obj_list = QtWidgets.QListWidget()
        self.lig_obj_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.lig_obj_list.setMinimumHeight(80)
        self.lig_obj_list.setMaximumHeight(140)
        self._lig_stack.addWidget(self.lig_obj_list)

        self._rb_lig_file.toggled.connect(
            lambda on: self._lig_stack.setCurrentIndex(0 if on else 1))
        self._rb_lig_obj.toggled.connect(
            lambda on: self.refresh() if on else None)
        lig_src_l.addWidget(self._lig_stack)
        form.addRow("Ligand:", lig_src_w)

        mode_w = QtWidgets.QWidget()
        mode_r = QtWidgets.QHBoxLayout(mode_w); mode_r.setContentsMargins(0,0,0,0)
        self.mode_sp = QtWidgets.QRadioButton("SP"); self.mode_sp.setChecked(True)
        self.mode_xp = QtWidgets.QRadioButton("XP")
        mode_r.addWidget(self.mode_sp); mode_r.addWidget(self.mode_xp)
        mode_r.addStretch()
        form.addRow("Mode:", mode_w)

        self.exhaus_sp = QtWidgets.QSpinBox()
        self.exhaus_sp.setRange(1, 64); self.exhaus_sp.setValue(8)
        self.poses_sp = QtWidgets.QSpinBox()
        self.poses_sp.setRange(1, 20); self.poses_sp.setValue(1)
        self.pad_sp = QtWidgets.QDoubleSpinBox()
        self.pad_sp.setRange(0, 20); self.pad_sp.setValue(4.0); self.pad_sp.setSuffix(" Å")
        self.cpu_sp = QtWidgets.QSpinBox()
        self.cpu_sp.setRange(1, 128); self.cpu_sp.setValue(min(8, os.cpu_count() or 4))
        self.gpu_chk = QtWidgets.QCheckBox("Use GPU"); self.gpu_chk.setChecked(True)
        form.addRow("Exhaustiveness:", self.exhaus_sp)
        form.addRow("Poses:", self.poses_sp)
        form.addRow("Box padding:", self.pad_sp)
        form.addRow("CPU threads:", self.cpu_sp)
        form.addRow("", self.gpu_chk)

        self.out_edit = QtWidgets.QLineEdit("docking_out")
        form.addRow("Output name:", self.out_edit)

        self.gnina_path = _GninaPath()
        form.addRow("GNINA:", self.gnina_path)
        root.addLayout(form)

        btn_w = QtWidgets.QWidget()
        btn_r = QtWidgets.QHBoxLayout(btn_w); btn_r.setContentsMargins(0,0,0,0)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.run_btn     = QtWidgets.QPushButton("Run Docking")
        self.load_btn    = QtWidgets.QPushButton("Load Selected Pose")
        self.load_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self.refresh)
        self.run_btn.clicked.connect(self._run)
        self.load_btn.clicked.connect(self._load_pose)
        for b in (self.refresh_btn, self.run_btn, self.load_btn):
            btn_r.addWidget(b)
        root.addWidget(btn_w)

        self._progress = _progress_bar()
        root.addWidget(self._progress)

        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Pose", "minimizedAffinity", "CNNscore", "CNNaffinity",
             "CNN_VS", "CNNaffinity_variance"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(180)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(self._load_pose)
        root.addWidget(self.table)

        self._log = _log_widget()
        root.addWidget(self._log)

        self._iv = _InteractionView()
        root.addWidget(self._iv)

        self.refresh()

    def refresh(self):
        objs = _pymol_objects()
        for box in (self.rec_box, self.ref_box):
            cur = box.currentText()
            box.clear(); box.addItems(objs)
            idx = box.findText(cur)
            if idx >= 0: box.setCurrentIndex(idx)
        exc = {self.rec_box.currentText(), self.ref_box.currentText()}
        prev_sel = {self.lig_obj_list.item(i).text()
                    for i in range(self.lig_obj_list.count())
                    if self.lig_obj_list.item(i).isSelected()}
        self.lig_obj_list.clear()
        for o in objs:
            if o in exc:
                continue
            item = QtWidgets.QListWidgetItem(o)
            self.lig_obj_list.addItem(item)
            if o in prev_sel:
                item.setSelected(True)

    def _browse_lig(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ligand SDF", "", "SDF (*.sdf *.sdf.gz);;All (*)"
        )
        if p: self.lig_edit.setText(p)

    def _run(self):
        gnina   = self.gnina_path.path()
        rec_obj = self.rec_box.currentText()
        ref_obj = self.ref_box.currentText()
        out_nm  = self.out_edit.text().strip() or "docking_out"

        if not rec_obj or not ref_obj:
            QtWidgets.QMessageBox.warning(self, "Pynacea", "Select receptor and ref ligand.")
            return

        if self._rb_lig_file.isChecked():
            lig_f = self.lig_edit.text().strip()
            if not lig_f or not os.path.exists(lig_f):
                QtWidgets.QMessageBox.warning(self, "Pynacea", "Specify a valid ligand SDF.")
                return
        else:
            sel_items = [self.lig_obj_list.item(i).text()
                         for i in range(self.lig_obj_list.count())
                         if self.lig_obj_list.item(i).isSelected()]
            if not sel_items:
                QtWidgets.QMessageBox.warning(self, "Pynacea",
                    "Select at least one PyMOL ligand object.")
                return
            lig_f = os.path.join(self._tmpdir, "lig_from_pymol.sdf")
            with open(lig_f, "w") as fh:
                for obj in sel_items:
                    tmp = os.path.join(self._tmpdir, f"_lig_{obj}.sdf")
                    if not _save_sel(obj, tmp):
                        QtWidgets.QMessageBox.warning(self, "Pynacea",
                            f"Could not save '{obj}' to SDF.")
                        return
                    with open(tmp) as th:
                        fh.write(th.read())

        rec_pdb = os.path.join(self._tmpdir, "receptor.pdb")
        ref_sdf = os.path.join(self._tmpdir, "ref.sdf")
        out_sdf = os.path.join(self._tmpdir, f"{out_nm}.sdf")

        cmd.save(rec_pdb, rec_obj, state=1)
        if not _save_sel(ref_obj, ref_sdf, state=1):
            QtWidgets.QMessageBox.warning(self, "Pynacea",
                f"Could not save reference ligand '{ref_obj}'.")
            return

        args = [gnina,
                "--receptor", rec_pdb, "--ligand", lig_f,
                "--autobox_ligand", ref_sdf,
                "--autobox_add", str(self.pad_sp.value()),
                "--out", out_sdf,
                "--num_modes", str(self.poses_sp.value()),
                "--exhaustiveness", str(self.exhaus_sp.value()),
                "--cpu", str(self.cpu_sp.value())]
        if not self.gpu_chk.isChecked():
            args.append("--no_gpu")
        if self.mode_xp.isChecked():
            args += ["--cnn_scoring", "refinement"]
        args = _wrap_wsl(args)

        self._out_sdf = out_sdf
        self._log.clear()
        self._log.appendPlainText("$ " + " ".join(args))
        self._log.appendPlainText("─" * 60)
        self.run_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.table.setRowCount(0)
        self._results = []
        self._progress.setVisible(True)

        self._worker = GninaWorker(args, out_sdf)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _done(self, sdf_path: str):
        self.run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self.load_btn.setEnabled(True)
        self._results = _parse_sdf(sdf_path)
        self._populate_table()
        n = len(self._results)
        self._log.appendPlainText(
            f"\n✓ {n} poses" if n else "\n✓ Done (install RDKit to populate table)"
        )
        self._log.appendPlainText("Double-click a row or click 'Load Selected Pose'")

    def _fail(self, msg: str):
        self.run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._log.appendPlainText(f"\n✗ {msg}")
        QtWidgets.QMessageBox.critical(self, "Docking failed", msg)

    def _populate_table(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._results))
        for r, row in enumerate(self._results):
            name_item = QtWidgets.QTableWidgetItem(row.get("name", ""))
            name_item.setData(QtCore.Qt.UserRole, r)
            self.table.setItem(r, 0, name_item)
            self.table.setItem(r, 1, _NumericItem(str(row["pose"])))
            self.table.setItem(r, 2, _NumericItem(_fmt(row.get("minimizedAffinity"))))
            self.table.setItem(r, 3, _NumericItem(_fmt(row.get("CNNscore"))))
            self.table.setItem(r, 4, _NumericItem(_fmt(row.get("CNNaffinity"))))
            self.table.setItem(r, 5, _NumericItem(_fmt(row.get("CNN_VS"))))
            self.table.setItem(r, 6, _NumericItem(_fmt(row.get("CNNaffinity_variance"))))
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def _load_pose(self):
        visual_row = self.table.currentRow()
        out_nm     = self.out_edit.text().strip() or "docking_out"
        if not self._results or visual_row < 0:
            if self._out_sdf and os.path.exists(self._out_sdf):
                cmd.load(self._out_sdf, out_nm)
                self._log.appendPlainText(f"Loaded all poses as '{out_nm}'")
                self.pose_loaded.emit(out_nm)
            return
        orig_idx = self.table.item(visual_row, 0).data(QtCore.Qt.UserRole)
        r        = self._results[orig_idx]
        name = f"{out_nm}_p{r['pose']}"
        tmp  = os.path.join(self._tmpdir, f"pose_{r['pose']}.sdf")
        if "mol_block" in r:
            with open(tmp, "w") as fh:
                fh.write(r["mol_block"])
            cmd.load(tmp, name)
        else:
            cmd.load(self._out_sdf, name)
        self._log.appendPlainText(f"Loaded pose {r['pose']} as '{name}'")
        try:
            pv = _import_poseviewer()
            pv.ci_setup(protein=self.rec_box.currentText(), ligands=name, mode="objects")
            self._iv.bind(pv)
        except Exception:
            pass
        self.pose_loaded.emit(name)

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate(); self._worker.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ─── Shared: PoseViewer import ───────────────────────────────────────────────

def _import_poseviewer():
    """Import PoseViewer from the same directory as this file. Cached after first call."""
    name = "PoseViewer"
    if name in sys.modules:
        return sys.modules[name]
    here = Path(__file__).parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import importlib
    return importlib.import_module(name)


class _InteractionView(QtWidgets.QWidget):
    """Reusable interaction-toggle panel, shared by DockingPanel and DesignPanel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pv = None
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(5); root.setContentsMargins(0, 0, 0, 0)

        # Reference ligand
        g_ref = QtWidgets.QGroupBox("Reference ligand")
        l_ref = QtWidgets.QHBoxLayout(g_ref)
        l_ref.addWidget(QtWidgets.QLabel("Object:"))
        self._ref_combo = QtWidgets.QComboBox(); self._ref_combo.addItem("(none)")
        self._ref_combo.setMinimumWidth(100)
        l_ref.addWidget(self._ref_combo, 1)
        self._cb_show_ref  = QtWidgets.QCheckBox("Show ref");  self._cb_show_ref.setChecked(True)
        self._cb_show_pose = QtWidgets.QCheckBox("Show pose"); self._cb_show_pose.setChecked(True)
        l_ref.addWidget(self._cb_show_ref); l_ref.addWidget(self._cb_show_pose)
        self._ref_combo.currentTextChanged.connect(self._on_ref_changed)
        self._cb_show_ref.stateChanged.connect(self._on_show_ref)
        self._cb_show_pose.stateChanged.connect(self._on_show_pose)
        root.addWidget(g_ref)

        def _cb(lay, text, hex_color, checked=True):
            row = QtWidgets.QHBoxLayout()
            sw = QtWidgets.QLabel(); sw.setFixedSize(14, 14)
            sw.setStyleSheet(f"background-color: {hex_color}; border: none;")
            row.addWidget(sw)
            cb = QtWidgets.QCheckBox(text); cb.setChecked(checked)
            row.addWidget(cb); row.addStretch()
            lay.addLayout(row)
            return cb

        def _section(title, enabled=True, expanded=False, show_enable=True):
            g = QtWidgets.QGroupBox()
            gl = QtWidgets.QVBoxLayout(g)
            gl.setContentsMargins(6, 4, 6, 6); gl.setSpacing(2)
            hl = QtWidgets.QHBoxLayout()
            if show_enable:
                en = QtWidgets.QCheckBox(title); en.setChecked(enabled)
                _f = en.font(); _f.setBold(True); en.setFont(_f)
                hl.addWidget(en)
            else:
                lbl = QtWidgets.QLabel(title)
                _f = lbl.font(); _f.setBold(True); lbl.setFont(_f)
                hl.addWidget(lbl); en = None
            hl.addStretch()
            btn_col = QtWidgets.QToolButton()
            btn_col.setCheckable(True); btn_col.setChecked(expanded)
            btn_col.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
            btn_col.setAutoRaise(True)
            hl.addWidget(btn_col)
            gl.addLayout(hl)
            body = QtWidgets.QWidget()
            bl = QtWidgets.QVBoxLayout(body)
            bl.setContentsMargins(0, 2, 0, 0); bl.setSpacing(2)
            body.setVisible(expanded)
            gl.addWidget(body)
            def _toggle(checked):
                body.setVisible(checked)
                btn_col.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
            btn_col.toggled.connect(_toggle)
            return g, en, body, bl

        g1, g1_en, _g1b, l1 = _section("Non-covalent bonds", enabled=True)
        cb_hb = _cb(l1, "Hydrogen bonds",  "#ffd900")
        cb_xb = _cb(l1, "Halogen bonds",   "#9933e6")
        cb_sb = _cb(l1, "Salt bridges",    "#e633e6")
        cb_ah = _cb(l1, "Aromatic H-Bond", "#4dd97f")
        root.addWidget(g1)

        g2, g2_en, _g2b, l2 = _section("Pi interactions", enabled=True)
        cb_pp = _cb(l2, "Pi-pi stacking", "#4dc0ff")
        cb_pc = _cb(l2, "Pi-cation",      "#33cc33")
        root.addWidget(g2)

        g3, g3_en, _g3b, l3 = _section("Contacts / Clashes", enabled=False)
        cb_cg  = _cb(l3, "Good", "#33cc33")
        cb_cb_ = _cb(l3, "Bad",  "#ff9900")
        cb_cu  = _cb(l3, "Ugly", "#ff2626")
        root.addWidget(g3)

        g_disp, g_disp_en, _gdb, l_disp = _section("Display", enabled=True, show_enable=True)
        self._cb_lb    = QtWidgets.QCheckBox("Show distance labels"); self._cb_lb.setChecked(True)
        self._cb_surf  = QtWidgets.QCheckBox("Show surface");         self._cb_surf.setChecked(True)
        self._cb_rlbl  = QtWidgets.QCheckBox("Show residue labels");  self._cb_rlbl.setChecked(True)
        self._cb_zoom  = QtWidgets.QCheckBox("Auto-zoom to pose");    self._cb_zoom.setChecked(True)
        self._cb_lig_h = QtWidgets.QCheckBox("Show nonpolar H on ligands"); self._cb_lig_h.setChecked(False)
        for _w in (self._cb_lb, self._cb_surf, self._cb_rlbl, self._cb_zoom, self._cb_lig_h):
            l_disp.addWidget(_w)
        root.addWidget(g_disp)
        root.addStretch()

        self._g1_en = g1_en; self._g2_en = g2_en; self._g3_en = g3_en
        self._g_disp_en = g_disp_en
        self._cb_hb = cb_hb; self._cb_xb = cb_xb; self._cb_sb = cb_sb; self._cb_ah = cb_ah
        self._cb_pp = cb_pp; self._cb_pc = cb_pc
        self._cb_cg = cb_cg; self._cb_cb_ = cb_cb_; self._cb_cu = cb_cu

        def _defer(cb, attr, g_en):
            cb.stateChanged.connect(lambda _: self._tog(attr, g_en))
        _defer(cb_hb, "show_hbonds",      g1_en); _defer(cb_xb, "show_halogen",   g1_en)
        _defer(cb_sb, "show_salt",        g1_en); _defer(cb_ah, "show_arom_hb",   g1_en)
        _defer(cb_pp, "show_pipi",        g2_en); _defer(cb_pc, "show_pi_cation", g2_en)
        _defer(cb_cg, "show_clash_good",  g3_en); _defer(cb_cb_, "show_clash_bad", g3_en)
        _defer(cb_cu, "show_clash_ugly",  g3_en)

        def _grp_defer(g_en, pairs):
            g_en.stateChanged.connect(lambda _: self._group_tog(g_en, pairs))
        _grp_defer(g1_en, [(cb_hb,"show_hbonds"),(cb_xb,"show_halogen"),
                            (cb_sb,"show_salt"),  (cb_ah,"show_arom_hb")])
        _grp_defer(g2_en, [(cb_pp,"show_pipi"),(cb_pc,"show_pi_cation")])
        _grp_defer(g3_en, [(cb_cg,"show_clash_good"),(cb_cb_,"show_clash_bad"),
                            (cb_cu,"show_clash_ugly")])

        self._cb_zoom.stateChanged.connect( lambda _: self._set_attr("auto_zoom", self._cb_zoom.isChecked()))
        self._cb_lb.stateChanged.connect(   lambda _: self._set_attr_disp("show_labels", g_disp_en, self._cb_lb))
        self._cb_lig_h.stateChanged.connect(lambda _: self._set_attr_disp("show_lig_h",  g_disp_en, self._cb_lig_h))
        self._cb_surf.stateChanged.connect( lambda _: self._do_toggle_surf())
        self._cb_rlbl.stateChanged.connect( lambda _: self._do_toggle_rlbl())
        g_disp_en.stateChanged.connect(     lambda _: self._do_disp_group_tog())

    # ── Bind / unbind ──────────────────────────────────────────────────────────

    def bind(self, pv_module):
        """Attach to a loaded PoseViewer module after ci_setup has been called."""
        self._pv = pv_module
        self._ref_combo.blockSignals(True)
        self._ref_combo.clear(); self._ref_combo.addItem("(none)")
        self._ref_combo.blockSignals(False)
        if pv_module is not None:
            self._populate_ref_combo()

    # ── Reference ligand ──────────────────────────────────────────────────────

    def _populate_ref_combo(self):
        pv = self._pv
        if pv is None: return
        prev = self._ref_combo.currentText()
        self._ref_combo.blockSignals(True)
        self._ref_combo.clear(); self._ref_combo.addItem("(none)")
        for n in cmd.get_names("objects"):
            if n.startswith("_ci_") or n in (
                    pv._OBJ_PTS, pv._OBJ_REF_PTS, pv._OBJ_SURF):
                continue
            try:
                if (cmd.count_atoms(f"{n} and organic") > 0 and
                        cmd.count_atoms(f"{n} and ({pv._stepper.protein_sel or 'polymer.protein'})") == 0):
                    self._ref_combo.addItem(n)
            except Exception:
                pass
        target = pv._stepper.ref_ligand or prev
        idx = self._ref_combo.findText(target) if target else -1
        self._ref_combo.setCurrentIndex(max(idx, 0))
        self._ref_combo.blockSignals(False)

    def _on_ref_changed(self, text):
        pv = self._pv
        if pv is None: return
        prev = pv._stepper.ref_ligand
        pv._stepper.ref_ligand = None if text == "(none)" else text
        if pv._stepper.ref_ligand:
            pv._color_ref_ligand(pv._stepper.ref_ligand)
            pv._stepper.show_ref = True
            self._cb_show_ref.blockSignals(True); self._cb_show_ref.setChecked(True)
            self._cb_show_ref.blockSignals(False)
        else:
            if prev:
                try: cmd.disable(prev)
                except Exception: pass
            if pv._OBJ_REF_PTS in pv._created_objects:
                try:
                    cmd.delete(pv._OBJ_REF_PTS)
                    pv._created_objects.discard(pv._OBJ_REF_PTS)
                except Exception: pass
            pv._stepper.show_ref = False
            self._cb_show_ref.blockSignals(True); self._cb_show_ref.setChecked(False)
            self._cb_show_ref.blockSignals(False)
        pv.ci_update()

    def _on_show_ref(self, _):
        pv = self._pv
        if pv: pv._stepper.show_ref = self._cb_show_ref.isChecked(); pv.ci_update()

    def _on_show_pose(self, _):
        pv = self._pv
        if pv: pv._stepper.show_pose = self._cb_show_pose.isChecked(); pv.ci_update()

    # ── Interaction toggles ───────────────────────────────────────────────────

    def _set_attr(self, attr, val):
        pv = self._pv
        if pv is not None:
            setattr(pv._stepper, attr, val)
            pv.ci_update()

    def _set_attr_disp(self, attr, g_en, cb):
        pv = self._pv
        if pv is not None:
            setattr(pv._stepper, attr, g_en.isChecked() and cb.isChecked())
            pv.ci_update()

    def _tog(self, attr, g_en):
        pv = self._pv
        if pv is None: return
        cb = getattr(self, f"_cb_{attr.replace('show_','')}", None)
        val = g_en.isChecked() and (cb.isChecked() if cb else True)
        setattr(pv._stepper, attr, val)
        pv.ci_update()

    def _group_tog(self, g_en, pairs):
        pv = self._pv
        if pv is None: return
        checked = g_en.isChecked()
        for cb, attr in pairs:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            setattr(pv._stepper, attr, checked)
        pv.ci_update()

    # ── Display toggles ───────────────────────────────────────────────────────

    def _do_toggle_surf(self):
        pv = self._pv
        if pv is None: return
        if pv._OBJ_SURF in pv._created_objects:
            if self._g_disp_en.isChecked() and self._cb_surf.isChecked():
                cmd.show("surface", pv._OBJ_SURF)
            else:
                cmd.hide("surface", pv._OBJ_SURF)

    def _do_toggle_rlbl(self):
        pv = self._pv
        if pv is None: return
        if pv._shell_sel is not None:
            sel = f"({pv._shell_sel}) and name CA"
            if self._g_disp_en.isChecked() and self._cb_rlbl.isChecked():
                cmd.show("labels", sel)
            else:
                cmd.hide("labels", sel)

    def _do_disp_group_tog(self):
        pv = self._pv
        if pv is None: return
        checked = self._g_disp_en.isChecked()
        pv._stepper.show_labels = checked and self._cb_lb.isChecked()
        pv._stepper.show_lig_h  = checked and self._cb_lig_h.isChecked()
        self._do_toggle_surf(); self._do_toggle_rlbl()
        pv.ci_update()


class PoseViewerPanel:
    """Removed — interaction visualization is now embedded in the Docking tab."""

# ─── Tab 5: Interactive Design ────────────────────────────────────────────────

# JSME 2-D molecular editor, loaded from CDN inside a QWebEngineView.
_JSME_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; width: 100%; height: 100%; background: #fff; }
  #status { color: #888; padding: 8px; font-family: sans-serif; font-size: 12px; }
</style>
</head>
<body>
<p id="status">Loading JSME sketcher…</p>
<div id="jsme_container"></div>
<script>
function jsmeOnLoad() {
    try {
        window._jsme = new JSApplet.JSME(
            "jsme_container", "100%", "420px",
            {"options": "query,depict,rSMARTS"});
        document.getElementById("status").style.display = "none";
    } catch(e) {
        document.getElementById("status").textContent =
            "JSME could not initialise (" + e + "). Check internet access.";
    }
}
</script>
<script src="https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"></script>
</body>
</html>"""


def _smiles_from_sdf(sdf_path: str, obabel: str) -> str:
    """Return the canonical SMILES for the first molecule in an SDF file."""
    tmp = sdf_path + "._smi"
    try:
        subprocess.run(_wrap_wsl([obabel, sdf_path, "-O", tmp]),
                       capture_output=True, timeout=15)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            with open(tmp) as fh:
                line = fh.readline().strip()
            return line.split()[0] if line else ""
    except Exception:
        pass
    finally:
        try: os.unlink(tmp)
        except Exception: pass
    return ""


class _StructureEditor(QtWidgets.QDialog):
    """SMILES text field + optional embedded JSME 2-D sketcher."""

    def __init__(self, smiles: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Structure")
        self.setMinimumSize(640, 540)
        self._smiles    = smiles
        self._has_jsme  = False
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)

        # SMILES text row (always present)
        smi_row = QtWidgets.QHBoxLayout()
        smi_row.addWidget(QtWidgets.QLabel("SMILES:"))
        self._smi_edit = QtWidgets.QLineEdit(self._smiles)
        self._smi_edit.setFont(_monofont())
        smi_row.addWidget(self._smi_edit, 1)
        root.addLayout(smi_row)

        # Try to embed JSME via QWebEngineView (probe multiple Qt bindings)
        wev_mod = self._try_get_webengine()
        if wev_mod is not None:
            try:
                self._view = wev_mod.QWebEngineView()
                self._view.setMinimumHeight(400)
                self._view.loadFinished.connect(self._on_load)
                self._view.page().setHtml(_JSME_HTML)
                root.addWidget(self._view, 1)
                self._has_jsme = True

                sync_row = QtWidgets.QHBoxLayout()
                push_btn = QtWidgets.QPushButton("Send SMILES → sketcher")
                pull_btn = QtWidgets.QPushButton("Get SMILES ← sketcher")
                push_btn.clicked.connect(self._push_smiles)
                pull_btn.clicked.connect(self._pull_smiles)
                sync_row.addWidget(push_btn); sync_row.addWidget(pull_btn)
                sync_row.addStretch()
                root.addLayout(sync_row)
            except Exception:
                pass

        if not self._has_jsme:
            if _RDKIT:
                self._preview_lbl = QtWidgets.QLabel()
                self._preview_lbl.setAlignment(QtCore.Qt.AlignCenter)
                self._preview_lbl.setMinimumHeight(300)
                self._preview_lbl.setStyleSheet(
                    "border: 1px solid #aaa; background: white;")
                root.addWidget(self._preview_lbl, 1)
                self._smi_edit.textChanged.connect(self._update_preview)
                self._update_preview(self._smiles)
            else:
                root.addWidget(QtWidgets.QLabel(
                    "ℹ No 2D sketcher — edit SMILES in the field above."))

        # OK / Cancel
        btn_row = QtWidgets.QHBoxLayout()
        apply_btn  = QtWidgets.QPushButton("Apply")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._on_apply)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(apply_btn); btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

    # ── WebEngine probe ───────────────────────────────────────────────────────

    @staticmethod
    def _try_get_webengine():
        for pkg in ("pymol.Qt", "PyQt5", "PySide2", "PyQt6", "PySide6"):
            try:
                mod = __import__(f"{pkg}.QtWebEngineWidgets",
                                 fromlist=["QWebEngineView"])
                if hasattr(mod, "QWebEngineView"):
                    return mod
            except Exception:
                continue
        return None

    # ── RDKit live preview ────────────────────────────────────────────────────

    def _update_preview(self, smiles: str = ""):
        if not hasattr(self, "_preview_lbl"):
            return
        import io
        from rdkit import Chem
        from rdkit.Chem import Draw
        smi = smiles if isinstance(smiles, str) else self._smi_edit.text().strip()
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            self._preview_lbl.setText("(invalid SMILES)" if smi else "")
            return
        try:
            img = Draw.MolToImage(mol, size=(400, 300))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            qimg = QtGui.QImage()
            qimg.loadFromData(buf.getvalue())
            pix = QtGui.QPixmap.fromImage(qimg)
            w = self._preview_lbl.width() or 400
            h = self._preview_lbl.height() or 300
            self._preview_lbl.setPixmap(pix.scaled(
                w, h,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))
        except Exception as exc:
            self._preview_lbl.setText(f"(preview error: {exc})")

    # ── JSME bridge ───────────────────────────────────────────────────────────

    def _on_load(self, ok):
        if ok and self._smiles:
            QtCore.QTimer.singleShot(900, self._push_smiles)

    def _push_smiles(self):
        smi = self._smi_edit.text().strip()
        if smi and self._has_jsme:
            js = f"if(window._jsme) window._jsme.readGenericMolecularInput({json.dumps(smi)});"
            self._view.page().runJavaScript(js)

    def _pull_smiles(self):
        if not self._has_jsme: return
        self._view.page().runJavaScript(
            "window._jsme ? window._jsme.smiles() : ''",
            lambda s: self._smi_edit.setText((s or "").strip()))

    def _on_apply(self):
        if self._has_jsme:
            self._view.page().runJavaScript(
                "window._jsme ? window._jsme.smiles() : ''",
                self._finish)
        else:
            self._finish("")

    def _finish(self, jsme_smiles: str):
        smi = (jsme_smiles or "").strip() or self._smi_edit.text().strip()
        self._smiles = smi
        self._smi_edit.setText(smi)
        self.accept()

    def smiles(self) -> str:
        return self._smiles


class _HistEntry:
    __slots__ = ("iteration", "smiles", "scores", "sdf_path")
    def __init__(self, it, smi, sc, path):
        self.iteration = it; self.smiles = smi
        self.scores = sc;    self.sdf_path = path


class DesignPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmpdir      = tempfile.mkdtemp(prefix="pynacea_design_")
        self._worker: Optional[GninaWorker] = None
        self._prep_worker: Optional[LigprepWorker] = None
        self._history: List[_HistEntry] = []
        self._iter = 0
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.rec_box = QtWidgets.QComboBox()
        self.ref_box = QtWidgets.QComboBox()
        self.lig_box = QtWidgets.QComboBox()
        form.addRow("Receptor:", self.rec_box)
        form.addRow("Ref ligand (box):", self.ref_box)

        lig_w = QtWidgets.QWidget()
        lig_r = QtWidgets.QHBoxLayout(lig_w); lig_r.setContentsMargins(0, 0, 0, 0)
        lig_r.addWidget(self.lig_box, 1)
        lig_load_btn = QtWidgets.QPushButton("Load SDF…"); lig_load_btn.setMaximumWidth(80)
        lig_load_btn.clicked.connect(self._load_lig_sdf)
        lig_r.addWidget(lig_load_btn)
        form.addRow("Design ligand:", lig_w)

        self.pad_sp = QtWidgets.QDoubleSpinBox()
        self.pad_sp.setRange(0, 20); self.pad_sp.setValue(4.0); self.pad_sp.setSuffix(" Å")
        form.addRow("Box padding:", self.pad_sp)

        mode_w = QtWidgets.QWidget()
        mode_r = QtWidgets.QHBoxLayout(mode_w); mode_r.setContentsMargins(0,0,0,0)
        self.mode_min   = QtWidgets.QRadioButton("Minimize (fast)")
        self.mode_min.setChecked(True)
        self.mode_local = QtWidgets.QRadioButton("Local opt (thorough)")
        mode_r.addWidget(self.mode_min); mode_r.addWidget(self.mode_local)
        mode_r.addStretch()
        form.addRow("Mode:", mode_w)

        self.exhaus_lbl = QtWidgets.QLabel("Exhaustiveness:")
        self.exhaus_sp  = QtWidgets.QSpinBox()
        self.exhaus_sp.setRange(1, 16); self.exhaus_sp.setValue(4)
        form.addRow(self.exhaus_lbl, self.exhaus_sp)
        self.exhaus_lbl.setVisible(False); self.exhaus_sp.setVisible(False)
        self.mode_min.toggled.connect(
            lambda on: (self.exhaus_lbl.setVisible(not on),
                        self.exhaus_sp.setVisible(not on))
        )

        self._prep_ph_sp = QtWidgets.QDoubleSpinBox()
        self._prep_ph_sp.setRange(0, 14); self._prep_ph_sp.setValue(7.4)
        self._prep_ph_sp.setSingleStep(0.1); self._prep_ph_sp.setDecimals(1)
        form.addRow("Prep pH:", self._prep_ph_sp)

        self.gnina_path = _GninaPath()
        form.addRow("GNINA:", self.gnina_path)
        root.addLayout(form)

        btn_w = QtWidgets.QWidget()
        btn_r = QtWidgets.QHBoxLayout(btn_w); btn_r.setContentsMargins(0,0,0,0)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.edit_btn    = QtWidgets.QPushButton("Edit Structure…")
        self.score_btn   = QtWidgets.QPushButton("Minimize && Score")
        self.restore_btn = QtWidgets.QPushButton("Restore Selected")
        self.restore_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self.refresh)
        self.edit_btn.clicked.connect(self._edit_structure)
        self.score_btn.clicked.connect(self._run)
        self.restore_btn.clicked.connect(self._restore)
        for b in (self.refresh_btn, self.edit_btn, self.score_btn, self.restore_btn):
            btn_r.addWidget(b)
        root.addWidget(btn_w)

        self._progress = _progress_bar()
        root.addWidget(self._progress)

        score_grp = QtWidgets.QGroupBox("Current Scores")
        sc_r = QtWidgets.QHBoxLayout(score_grp)
        self.lbl_cnn_aff = self._score_lbl("CNNaffinity")
        self.lbl_cnn_sc  = self._score_lbl("CNNscore")
        self.lbl_vina    = self._score_lbl("Vina")
        for lbl in (self.lbl_cnn_aff, self.lbl_cnn_sc, self.lbl_vina):
            sc_r.addWidget(lbl)
        root.addWidget(score_grp)

        hist_grp = QtWidgets.QGroupBox("Score History")
        hist_l = QtWidgets.QVBoxLayout(hist_grp)
        self.hist = QtWidgets.QTableWidget(0, 5)
        self.hist.setHorizontalHeaderLabels(["Iter","CNNaffinity","CNNscore","Vina","SMILES"])
        self.hist.horizontalHeader().setStretchLastSection(True)
        self.hist.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.hist.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.hist.setAlternatingRowColors(True)
        self.hist.setMaximumHeight(160)
        self.hist.itemSelectionChanged.connect(
            lambda: self.restore_btn.setEnabled(bool(self.hist.selectedItems()))
        )
        hist_l.addWidget(self.hist)
        root.addWidget(hist_grp)

        self._log = _log_widget()
        self._log.setMaximumHeight(120)
        root.addWidget(self._log)

        self._iv = _InteractionView()
        root.addWidget(self._iv)

        self.refresh()

    def _score_lbl(self, title: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(f"{title}\n—")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        lbl.setMinimumWidth(110); lbl.setMinimumHeight(44)
        return lbl

    def _load_lig_sdf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Design Ligand", "", "SDF (*.sdf *.SDF);;All (*)")
        if not path:
            return
        name = _sanitize_obj_name(Path(path).stem)
        cmd.load(path, name)
        self.refresh(preselect=name)

    def _edit_structure(self):
        lig_obj = self.lig_box.currentText()
        if not lig_obj:
            QtWidgets.QMessageBox.warning(self, "Pynacea",
                "Select a design ligand first.")
            return
        obabel  = _cfg.get("obabel_path", "obabel")
        tmp_sdf = os.path.join(self._tmpdir, "edit_in.sdf")
        _save_sel(lig_obj, tmp_sdf, state=-1)
        smiles  = _smiles_from_sdf(tmp_sdf, obabel) if os.path.exists(tmp_sdf) else ""

        dlg = _StructureEditor(smiles, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_smiles = dlg.smiles()
        if not new_smiles:
            QtWidgets.QMessageBox.warning(self, "Pynacea", "No SMILES returned.")
            return

        existing = [o for o in _pymol_objects() if o.startswith(lig_obj + "_edit")]
        new_name = f"{lig_obj}_edit{len(existing) + 1}"
        out_sdf  = os.path.join(self._tmpdir, f"{new_name}.sdf")

        self._log.appendPlainText(f"\nPreparing '{new_name}' from edited SMILES…")
        self.score_btn.setEnabled(False)
        self.edit_btn.setEnabled(False)
        self._prep_worker = LigprepWorker(
            obabel=obabel,
            entries=[(new_smiles, new_name)],
            out_sdf=out_sdf,
            ph=self._prep_ph_sp.value(),
            tmpdir=self._tmpdir,
        )
        self._prep_worker.log_line.connect(self._log.appendPlainText)
        self._prep_worker.finished.connect(lambda p: self._edit_done(p, new_name))
        self._prep_worker.failed.connect(self._edit_fail)
        self._prep_worker.start()

    def _edit_done(self, sdf_path: str, name: str):
        self.score_btn.setEnabled(True)
        self.edit_btn.setEnabled(True)
        cmd.load(sdf_path, name)
        self.refresh(preselect=name)
        self._log.appendPlainText(f"✓ '{name}' ready — click Minimize & Score")

    def _edit_fail(self, msg: str):
        self.score_btn.setEnabled(True)
        self.edit_btn.setEnabled(True)
        self._log.appendPlainText(f"✗ Ligand prep failed: {msg}")

    def refresh(self, preselect: str = ""):
        objs = _pymol_objects()
        for box in (self.rec_box, self.ref_box, self.lig_box):
            cur = box.currentText()
            box.clear(); box.addItems(objs)
            idx = box.findText(cur)
            if idx >= 0: box.setCurrentIndex(idx)
        if preselect:
            idx = self.lig_box.findText(preselect)
            if idx >= 0: self.lig_box.setCurrentIndex(idx)

    def _run(self):
        gnina   = self.gnina_path.path()
        rec_obj = self.rec_box.currentText()
        ref_obj = self.ref_box.currentText()
        lig_obj = self.lig_box.currentText()
        if not rec_obj or not ref_obj or not lig_obj:
            QtWidgets.QMessageBox.warning(
                self, "Pynacea", "Select receptor, ref ligand, and design ligand."
            )
            return

        self._iter += 1
        rec_pdb = os.path.join(self._tmpdir, "receptor.pdb")
        ref_sdf = os.path.join(self._tmpdir, "ref.sdf")
        lig_sdf = os.path.join(self._tmpdir, f"in_{self._iter}.sdf")
        out_sdf = os.path.join(self._tmpdir, f"min_{self._iter}.sdf")

        cmd.save(rec_pdb, rec_obj, state=1)
        if not _save_sel(ref_obj, ref_sdf, state=1):
            QtWidgets.QMessageBox.warning(self, "Pynacea",
                f"Could not save ref ligand '{ref_obj}'.")
            return
        if not _save_sel(lig_obj, lig_sdf, state=-1):
            QtWidgets.QMessageBox.warning(self, "Pynacea",
                f"Could not save design ligand '{lig_obj}'.")
            return

        # Reprotonation + in-place FF cleanup via obabel (fast, keeps 3D coords)
        obabel  = _cfg.get("obabel_path", "obabel")
        lig_prep = os.path.join(self._tmpdir, f"prep_{self._iter}.sdf")
        ph       = self._prep_ph_sp.value()
        try:
            r = subprocess.run(
                _wrap_wsl([obabel, lig_sdf, "-O", lig_prep,
                           "-p", str(ph), "--minimize", "--ff", "MMFF94s",
                           "--crit", "1e-6", "--sd"]),
                capture_output=True, text=True, timeout=60
            )
            gnina_lig = lig_prep if (os.path.exists(lig_prep) and
                                     os.path.getsize(lig_prep) > 50) else lig_sdf
        except Exception:
            gnina_lig = lig_sdf
        if gnina_lig == lig_sdf:
            self._log.appendPlainText("⚠ obabel prep failed — using raw PyMOL export")
        else:
            self._log.appendPlainText(f"  obabel prep pH {ph:.1f} ✓")

        args = [gnina,
                "--receptor", rec_pdb, "--ligand", gnina_lig,
                "--autobox_ligand", ref_sdf,
                "--autobox_add", str(self.pad_sp.value()),
                "--out", out_sdf,
                "--num_modes", "1",
                "--cpu", str(min(4, os.cpu_count() or 2))]
        if self.mode_local.isChecked():
            args += ["--local_only", "--exhaustiveness", str(self.exhaus_sp.value())]
        else:
            args.append("--minimize")
        args = _wrap_wsl(args)

        self._log.appendPlainText(f"\n[iter {self._iter}] $ " + " ".join(args))
        self.score_btn.setEnabled(False)
        self._progress.setVisible(True)

        self._worker = GninaWorker(args, out_sdf)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.finished.connect(lambda p: self._done(p, lig_obj))
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _done(self, sdf_path: str, lig_obj: str):
        self.score_btn.setEnabled(True)
        self._progress.setVisible(False)
        poses = _parse_sdf(sdf_path)
        if not poses:
            self._log.appendPlainText("⚠ Could not parse scores (RDKit unavailable?)")
            cmd.load(sdf_path, lig_obj)
            return
        p      = poses[0]
        smiles = p.get("smiles", "")
        scores = {k: p[k] for k in ("CNNaffinity", "CNNscore", "Vina") if k in p}
        self.lbl_cnn_aff.setText(f"CNNaffinity\n{_fmt(scores.get('CNNaffinity'))}")
        self.lbl_cnn_sc.setText( f"CNNscore\n{_fmt(scores.get('CNNscore'))}")
        self.lbl_vina.setText(   f"Vina\n{_fmt(scores.get('Vina'))}")
        cmd.delete(lig_obj)
        cmd.load(sdf_path, lig_obj)
        self._log.appendPlainText(
            f"[iter {self._iter}]  CNNaff={_fmt(scores.get('CNNaffinity'))}  "
            f"CNNsc={_fmt(scores.get('CNNscore'))}  Vina={_fmt(scores.get('Vina'))}"
        )
        try:
            pv = _import_poseviewer()
            pv.ci_setup(protein=self.rec_box.currentText(),
                        ligands=lig_obj, mode="objects")
            self._iv.bind(pv)
        except Exception:
            pass
        entry = _HistEntry(self._iter, smiles, scores, sdf_path)
        self._history.append(entry)
        r = self.hist.rowCount()
        self.hist.insertRow(r)
        self.hist.setItem(r, 0, QtWidgets.QTableWidgetItem(str(entry.iteration)))
        self.hist.setItem(r, 1, QtWidgets.QTableWidgetItem(_fmt(scores.get("CNNaffinity"))))
        self.hist.setItem(r, 2, QtWidgets.QTableWidgetItem(_fmt(scores.get("CNNscore"))))
        self.hist.setItem(r, 3, QtWidgets.QTableWidgetItem(_fmt(scores.get("Vina"))))
        self.hist.setItem(r, 4, QtWidgets.QTableWidgetItem(smiles))
        self.hist.resizeColumnsToContents()
        self.hist.scrollToBottom()

    def _fail(self, msg: str):
        self.score_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._log.appendPlainText(f"✗ {msg}")
        QtWidgets.QMessageBox.critical(self, "Minimize failed", msg)

    def _restore(self):
        row = self.hist.currentRow()
        if row < 0 or row >= len(self._history):
            return
        e       = self._history[row]
        lig_obj = self.lig_box.currentText()
        if not lig_obj:
            return
        cmd.delete(lig_obj)
        cmd.load(e.sdf_path, lig_obj)
        self.lbl_cnn_aff.setText(f"CNNaffinity\n{_fmt(e.scores.get('CNNaffinity'))}")
        self.lbl_cnn_sc.setText( f"CNNscore\n{_fmt(e.scores.get('CNNscore'))}")
        self.lbl_vina.setText(   f"Vina\n{_fmt(e.scores.get('Vina'))}")
        self._log.appendPlainText(f"Restored iter {e.iteration}")

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate(); self._worker.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ─── Main dialog ──────────────────────────────────────────────────────────────

class SBDDDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Pynacea  v{PLUGIN_VERSION}")
        self.setMinimumWidth(560)
        self.setMinimumHeight(720)
        self.setSizeGripEnabled(True)

        self._protprep = ProtprepPanel()
        self._fpocket  = FpocketPanel()
        self._ligprep  = LigprepPanel()
        self._docking  = DockingPanel()
        self._design   = DesignPanel()

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._protprep, "1 · Protein Prep")
        self._tabs.addTab(self._fpocket,  "2 · Pockets")
        self._tabs.addTab(self._ligprep,  "3 · Ligand Prep")
        self._tabs.addTab(self._docking,  "4 · Docking")
        self._tabs.addTab(self._design,   "5 · Design")

        # After ligprep loads objects, refresh the docking ligand list
        self._ligprep.objects_loaded.connect(lambda _: self._docking.refresh())
        # Loading a docked pose refreshes the Design ligand list
        self._docking.pose_loaded.connect(lambda name: self._design.refresh(preselect=name))

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._tabs)

    def closeEvent(self, event):
        for panel in (self._protprep, self._fpocket, self._ligprep,
                      self._docking, self._design):
            panel.cleanup()
        super().closeEvent(event)


# ─── Plugin entry points ──────────────────────────────────────────────────────

_dialog: Optional[SBDDDialog] = None


def run_plugin_gui():
    global _dialog
    if _dialog is not None:
        try:
            _dialog.close()
        except RuntimeError:
            pass
        _dialog = None
    _dialog = SBDDDialog()
    _dialog.show()
    _dialog.raise_()
    _dialog.activateWindow()


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("Pynacea", run_plugin_gui)


cmd.extend("pynacea", lambda: run_plugin_gui())
