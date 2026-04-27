"""
SBDD Suite — PyMOL Plugin
=========================
Structure-Based Drug Design workflow plugin for PyMOL.  Four tabs follow
the standard pipeline:

  1. Protein Prep   — clean, fix, protonate, minimize receptor (protprep.py)
  2. Ligand Prep    — protonate + 3-D embed ligands from SMILES or SDF (obabel)
  3. Docking        — GNINA docking: receptor + ref ligand → ranked pose table
  4. Design         — load a pose, edit in PyMOL builder, minimize & rescore

Requirements
------------
  protprep.py  — SBDD protein preparation script (openmmdl conda env)
  obabel       — OpenBabel CLI, must be on PATH or configured below
  GNINA        — https://github.com/gnina/gnina

Installation
------------
  Plugin > Plugin Manager > Install New Plugin > choose this file
  — or —
  run /path/to/sbdd_suite.py   then   sbdd_suite

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

_CFG_FILE = Path.home() / ".config" / "sbdd_suite.json"

_DEFAULTS: Dict[str, str] = {
    "gnina_path":      os.environ.get("GNINA_PATH", "/opt/gnina/gnina"),
    "protprep_script": os.environ.get("PROTPREP_SCRIPT", ""),
    "openmmdl_python": os.environ.get("OPENMMDL_PYTHON", ""),
    "obabel_path":     os.environ.get("OBABEL_PATH", shutil.which("obabel") or "obabel"),
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


def _parse_sdf(sdf_path: str) -> List[Dict[str, Any]]:
    if not _RDKIT:
        return []
    rows: List[Dict[str, Any]] = []
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        row: Dict[str, Any] = {"pose": i + 1}
        for prop in ("CNNaffinity", "CNNscore", "minimizedAffinity"):
            if mol.HasProp(prop):
                try:
                    row[prop] = float(mol.GetProp(prop))
                except ValueError:
                    pass
        if "minimizedAffinity" in row:
            row["Vina"] = row["minimizedAffinity"]
        try:
            row["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            row["smiles"] = ""
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
        self._tmpdir  = tempfile.mkdtemp(prefix="sbdd_prot_")
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
            QtWidgets.QMessageBox.warning(self, "SBDD Suite", "Specify a valid PDB file first.")
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
            QtWidgets.QMessageBox.warning(self, "SBDD Suite",
                "Configure the openmmdl Python path first.")
            return
        if not script or not os.path.exists(script):
            QtWidgets.QMessageBox.warning(self, "SBDD Suite",
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


# ─── Tab 2: Ligand Preparation ────────────────────────────────────────────────

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmpdir = tempfile.mkdtemp(prefix="sbdd_lig_")
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
        self._open_btn = QtWidgets.QPushButton("Open Output SDF…")
        self._open_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        self._open_btn.clicked.connect(self._open_output)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._open_btn)
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
            QtWidgets.QMessageBox.warning(self, "SBDD Suite", "Set the obabel path first.")
            return

        if self._rb_smiles.isChecked():
            text    = self._smiles_edit.toPlainText().strip()
            entries = _parse_smiles_input(text)
            if not entries:
                QtWidgets.QMessageBox.warning(self, "SBDD Suite", "Enter at least one SMILES.")
                return
        else:
            sdf_in = self._sdf_edit.text().strip()
            if not sdf_in or not os.path.exists(sdf_in):
                QtWidgets.QMessageBox.warning(self, "SBDD Suite", "Select a valid SDF file.")
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
                    self, "SBDD Suite",
                    "SDF input mode requires RDKit to extract SMILES.\n"
                    "Install RDKit or paste SMILES directly."
                )
                return
            if not entries:
                QtWidgets.QMessageBox.warning(self, "SBDD Suite", "No valid molecules in SDF.")
                return

        out_path = self._out_edit.text().strip()
        if not out_path:
            out_path = os.path.join(self._tmpdir, "prepared_ligands.sdf")
        self._out_sdf = out_path

        self._log.clear()
        self._log.appendPlainText(f"Preparing {len(entries)} ligand(s) at pH {self._ph_sp.value()}")
        self._log.appendPlainText("─" * 60)
        self._run_btn.setEnabled(False)
        self._open_btn.setEnabled(False)
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
        self._open_btn.setEnabled(True)
        n = sdf_path and sum(1 for l in open(sdf_path) if l.strip() == "$$$$")
        self._log.appendPlainText(f"\n✓ {n} ligand(s) written to {sdf_path}")
        if self._load_chk.isChecked():
            stem = Path(sdf_path).stem
            cmd.load(sdf_path, stem)
            self._log.appendPlainText(f"✓ Loaded '{stem}' into PyMOL")

    def _fail(self, msg: str):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._log.appendPlainText(f"\n✗ {msg}")
        QtWidgets.QMessageBox.critical(self, "Ligand preparation failed", msg)

    def _open_output(self):
        if self._out_sdf and os.path.exists(self._out_sdf):
            QtWidgets.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(self._out_sdf)
            )

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate(); self._worker.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ─── Tab 3: Docking ───────────────────────────────────────────────────────────

class DockingPanel(QtWidgets.QWidget):
    pose_loaded = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmpdir  = tempfile.mkdtemp(prefix="sbdd_dock_")
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

        lig_w = QtWidgets.QWidget()
        lig_r = QtWidgets.QHBoxLayout(lig_w); lig_r.setContentsMargins(0,0,0,0)
        self.lig_edit = QtWidgets.QLineEdit()
        self.lig_edit.setPlaceholderText("ligands.sdf")
        lig_btn = QtWidgets.QPushButton("…"); lig_btn.setMaximumWidth(28)
        lig_btn.clicked.connect(self._browse_lig)
        lig_r.addWidget(self.lig_edit); lig_r.addWidget(lig_btn)
        form.addRow("Ligand file:", lig_w)

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
        self.poses_sp.setRange(1, 20); self.poses_sp.setValue(9)
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

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Pose","CNNaffinity","CNNscore","Vina","SMILES"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(180)
        self.table.doubleClicked.connect(self._load_pose)
        root.addWidget(self.table)

        self._log = _log_widget()
        root.addWidget(self._log)

        self.refresh()

    def refresh(self):
        objs = _pymol_objects()
        for box in (self.rec_box, self.ref_box):
            cur = box.currentText()
            box.clear(); box.addItems(objs)
            idx = box.findText(cur)
            if idx >= 0: box.setCurrentIndex(idx)

    def _browse_lig(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ligand SDF", "", "SDF (*.sdf *.sdf.gz);;All (*)"
        )
        if p: self.lig_edit.setText(p)

    def _run(self):
        gnina   = self.gnina_path.path()
        rec_obj = self.rec_box.currentText()
        ref_obj = self.ref_box.currentText()
        lig_f   = self.lig_edit.text().strip()
        out_nm  = self.out_edit.text().strip() or "docking_out"

        if not rec_obj or not ref_obj:
            QtWidgets.QMessageBox.warning(self, "SBDD Suite", "Select receptor and ref ligand.")
            return
        if not lig_f or not os.path.exists(lig_f):
            QtWidgets.QMessageBox.warning(self, "SBDD Suite", "Specify a valid ligand SDF.")
            return

        rec_pdb = os.path.join(self._tmpdir, "receptor.pdb")
        ref_sdf = os.path.join(self._tmpdir, "ref.sdf")
        out_sdf = os.path.join(self._tmpdir, f"{out_nm}.sdf")

        cmd.save(rec_pdb, rec_obj, state=1)
        if not _save_sel(ref_obj, ref_sdf, state=1):
            QtWidgets.QMessageBox.warning(self, "SBDD Suite",
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
        self.table.setRowCount(len(self._results))
        for r, row in enumerate(self._results):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(row["pose"])))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(_fmt(row.get("CNNaffinity"))))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(_fmt(row.get("CNNscore"))))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(_fmt(row.get("Vina"))))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(row.get("smiles", "")))
        self.table.resizeColumnsToContents()

    def _load_pose(self):
        row_idx = self.table.currentRow()
        out_nm  = self.out_edit.text().strip() or "docking_out"
        if not self._results or row_idx < 0:
            if self._out_sdf and os.path.exists(self._out_sdf):
                cmd.load(self._out_sdf, out_nm)
                self._log.appendPlainText(f"Loaded all poses as '{out_nm}'")
                self.pose_loaded.emit(out_nm)
            return
        r    = self._results[row_idx]
        name = f"{out_nm}_p{r['pose']}"
        tmp  = os.path.join(self._tmpdir, f"pose_{r['pose']}.sdf")
        if "mol_block" in r:
            with open(tmp, "w") as fh:
                fh.write(r["mol_block"])
            cmd.load(tmp, name)
        else:
            cmd.load(self._out_sdf, name)
        self._log.appendPlainText(f"Loaded pose {r['pose']} as '{name}'")
        self.pose_loaded.emit(name)

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate(); self._worker.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ─── Tab 4: Interactive Design ────────────────────────────────────────────────

class _HistEntry:
    __slots__ = ("iteration", "smiles", "scores", "sdf_path")
    def __init__(self, it, smi, sc, path):
        self.iteration = it; self.smiles = smi
        self.scores = sc;    self.sdf_path = path


class DesignPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tmpdir = tempfile.mkdtemp(prefix="sbdd_design_")
        self._worker: Optional[GninaWorker] = None
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
        form.addRow("Design ligand:", self.lig_box)

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

        self.gnina_path = _GninaPath()
        form.addRow("GNINA:", self.gnina_path)
        root.addLayout(form)

        btn_w = QtWidgets.QWidget()
        btn_r = QtWidgets.QHBoxLayout(btn_w); btn_r.setContentsMargins(0,0,0,0)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.score_btn   = QtWidgets.QPushButton("Minimize && Score")
        self.restore_btn = QtWidgets.QPushButton("Restore Selected")
        self.restore_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self.refresh)
        self.score_btn.clicked.connect(self._run)
        self.restore_btn.clicked.connect(self._restore)
        for b in (self.refresh_btn, self.score_btn, self.restore_btn):
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

        self.refresh()

    def _score_lbl(self, title: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(f"{title}\n—")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        lbl.setMinimumWidth(110); lbl.setMinimumHeight(44)
        return lbl

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
                self, "SBDD Suite", "Select receptor, ref ligand, and design ligand."
            )
            return

        self._iter += 1
        rec_pdb = os.path.join(self._tmpdir, "receptor.pdb")
        ref_sdf = os.path.join(self._tmpdir, "ref.sdf")
        lig_sdf = os.path.join(self._tmpdir, f"in_{self._iter}.sdf")
        out_sdf = os.path.join(self._tmpdir, f"min_{self._iter}.sdf")

        cmd.save(rec_pdb, rec_obj, state=1)
        if not _save_sel(ref_obj, ref_sdf, state=1):
            QtWidgets.QMessageBox.warning(self, "SBDD Suite",
                f"Could not save ref ligand '{ref_obj}'.")
            return
        if not _save_sel(lig_obj, lig_sdf, state=-1):
            QtWidgets.QMessageBox.warning(self, "SBDD Suite",
                f"Could not save design ligand '{lig_obj}'.")
            return

        args = [gnina,
                "--receptor", rec_pdb, "--ligand", lig_sdf,
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
        self.setWindowTitle(f"SBDD Suite  v{PLUGIN_VERSION}")
        self.setMinimumWidth(560)
        self.setMinimumHeight(720)
        self.setSizeGripEnabled(True)

        self._protprep = ProtprepPanel()
        self._ligprep  = LigprepPanel()
        self._docking  = DockingPanel()
        self._design   = DesignPanel()

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._protprep, "1 · Protein Prep")
        self._tabs.addTab(self._ligprep,  "2 · Ligand Prep")
        self._tabs.addTab(self._docking,  "3 · Docking")
        self._tabs.addTab(self._design,   "4 · Design")

        # Loading a docked pose auto-selects it in the Design tab
        self._docking.pose_loaded.connect(self._on_pose_loaded)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._tabs)

    def _on_pose_loaded(self, name: str):
        self._design.refresh(preselect=name)
        self._tabs.setCurrentWidget(self._design)

    def closeEvent(self, event):
        for panel in (self._protprep, self._ligprep, self._docking, self._design):
            panel.cleanup()
        super().closeEvent(event)


# ─── Plugin entry points ──────────────────────────────────────────────────────

_dialog: Optional[SBDDDialog] = None


def run_plugin_gui():
    global _dialog
    if _dialog is None or not _dialog.isVisible():
        _dialog = SBDDDialog()
    _dialog.show()
    _dialog.raise_()
    _dialog.activateWindow()


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("SBDD Suite", run_plugin_gui)


cmd.extend("sbdd_suite", lambda: run_plugin_gui())
