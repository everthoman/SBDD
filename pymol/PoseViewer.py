"""
PoseViewer - PyMOL Plugin  v1.9.8
==============================
Maestro-inspired protein-ligand interaction viewer for PyMOL. Automatically
detects and visualizes all major non-covalent interactions, with ligand
stepping for docking pose review.

H-bonds use PyMOL's built-in polar contact detection (cmd.distance mode=2)
which correctly handles donor/acceptor chemistry. All other interaction
types are detected geometrically.

Interaction categories:
  Non-covalent bonds:  H-bonds, halogen bonds, salt bridges, aromatic H-bonds
  Pi interactions:     Pi-pi stacking (face-to-face & edge-to-face), pi-cation
  Contacts/Clashes:    Good, bad, ugly  (off by default)

Installation:
  1. Plugin > Plugin Manager > Install New Plugin > choose this file, or
  2. run /path/to/PoseViewer.py   then   ci_gui

Authors: Evert J. Homan, PhD; Claude (Anthropic)
Date:    2026-09-10
Version: 1.9.2
License: MIT
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from typing import List, Dict, Optional, Set

try:
    import numpy as np
except ImportError:
    np = None

from pymol import cmd, CmdException

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COLORS = {
    "ci_hbond":       (1.00, 0.85, 0.00),
    "ci_halogen":     (0.60, 0.20, 0.90),
    "ci_salt":        (0.90, 0.20, 0.90),
    "ci_arom_hb":     (0.30, 0.85, 0.50),
    "ci_pipi":        (0.30, 0.75, 1.00),
    "ci_pi_cat":      (0.20, 0.80, 0.20),
    "ci_clash_good":  (0.20, 0.80, 0.20),
    "ci_clash_bad":   (1.00, 0.60, 0.00),
    "ci_clash_ugly":  (1.00, 0.15, 0.15),
    "ci_water":       (0.30, 0.80, 0.95),
    # Surface charge shades: saturated for formal charge, pale for partial.
    "ci_surf_neg":     (0.90, 0.20, 0.20),   # carboxylate O
    "ci_surf_neg_wk":  (0.96, 0.62, 0.60),   # amide / hydroxyl / thiol
    "ci_surf_pos":     (0.24, 0.36, 0.92),   # guanidinium / ammonium / HIP
    "ci_surf_pos_wk":  (0.62, 0.70, 0.96),   # amide N / indole / neutral His
}

def _register_colors():
    """Register custom colors with PyMOL. Called at runtime to ensure PyMOL is ready."""
    for name, rgb in COLORS.items():
        cmd.set_color(name, list(rgb))

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Fe": 1.80, "Zn": 1.39, "Mg": 1.73, "Ca": 1.74, "Mn": 1.80,
    "Se": 1.90, "Si": 2.10, "B": 1.92,
}

HBOND_ELEMENTS = {"N", "O", "S", "F"}
# H-bonds are found by PyMOL's cmd.distance(mode=2), then filtered on the real
# D-H...A angle (see _hbond_dha_angle).  PyMOL's own h_bond_max_angle is measured
# at the donor heavy atom, so its 63 deg default lets through contacts whose
# proton points ~100 deg away from the acceptor.  0 disables the filter.
HBOND_MIN_DHA_ANGLE = 130.0

HALOGEN_DONORS = {"Cl", "Br", "I"}
HALOGEN_ACCEPTORS = {"O", "N", "S"}
HALOGEN_DIST_MAX = 3.5

AROM_HBOND_DIST_MAX = 3.5
AROM_HBOND_ACCEPTORS = {"O", "N", "S"}

SALT_BRIDGE_DIST_MAX = 4.0

PIPI_FTF_DIST_MAX = 4.8
PIPI_FTF_ANGLE_MAX = 40.0
PIPI_ETF_DIST_MAX = 5.5
PIPI_ETF_ANGLE_MIN = 45.0
PIPI_ETF_ANGLE_MAX = 90.0

PI_CATION_DIST_MAX = 6.0

CLASH_GOOD_FRAC = 1.30        # comfortable VDW contact (up to 130% of VDW sum)
CLASH_BAD_FRAC = 0.89         # mild steric overlap
CLASH_UGLY_FRAC = 0.75        # severe steric overlap

# Widest separation at which any clash can still register, derived from the
# table above so it stays correct if a larger radius is ever added.
CLASH_DIST_MAX = 2.0 * max(VDW_RADII.values()) * CLASH_GOOD_FRAC

SHELL_DIST = 5.0
# The pocket surface is a piece of the real protein molecular surface: computed
# on a generous residue shell (so the solvent-excluded surface is correct near
# the pocket) and then carved back to just the wall facing the ligand.  Surfacing
# only the atoms within SHELL_DIST instead produced a closed blob around a bag of
# clipped side chains that swallowed the pocket residues whole.
SURF_CARVE_DIST = 7.5                # default surface reach (Å); GUI-tunable per session
SURF_SHELL_PAD  = 3.0                # extra reach for the SES scratch geometry
ZOOM_BUFFER = 2.0             # padding around the binding site when auto-zooming

DASH_RADIUS = 0.06
DASH_GAP = 0.35
DASH_LENGTH = 0.20
LABEL_SIZE = 14

# ---------------------------------------------------------------------------
# Object tracking
# ---------------------------------------------------------------------------

_created_objects: Set[str] = set()
_shell_sel: Optional[str] = None   # selection used for lines/labels (no copy object)
_shell_key: Optional[tuple] = None # inputs the current shell was built from

_OBJ_PTS        = "_ci_pts"
_OBJ_SNAP       = "_ci_snap"     # single-state pose copies used by `within`
_OBJ_SHELL      = "_ci_shell"    # materialised shell selection
_OBJ_SHELL_ATOMS = "_ci_shell_atoms"
_OBJ_CARVE      = "_ci_carve"    # single-state ligand copy the surface carves against
_OBJ_REF_PTS    = "_ci_ref_pts"
_OBJ_SURF       = "_ci_surf"

_INTERACTION_NAMES = {
    "hbonds": "hbonds",
    "halogen": "halogen_bonds",
    "salt": "salt_bridges",
    "arom_hb": "arom_hbonds",
    "pipi": "pi_pi",
    "pi_cation": "pi_cation",
    "clash_good": "clash_good",
    "clash_bad": "clash_bad",
    "clash_ugly": "clash_ugly",
    "water":      "water_bridges",
}

_AUTOSPLIT_PREFIX = "obj"

def _track(name):
    _created_objects.add(name)

def _clear_shell():
    global _shell_sel, _shell_key
    _shell_key = None
    if _shell_sel is not None:
        try:
            cmd.hide("lines",  _shell_sel)
            cmd.hide("labels", f"({_shell_sel}) and name CA")
        except Exception:
            pass
        try: cmd.delete(_OBJ_SHELL)
        except Exception: pass
        _shell_sel = None
    for _o in (_OBJ_SURF, _OBJ_CARVE):
        if _o in _created_objects:
            try: cmd.delete(_o)
            except Exception: pass
            _created_objects.discard(_o)

def _clear_all():
    _clear_shell()
    for name in list(_created_objects):
        try: cmd.delete(name)
        except Exception: pass
    _created_objects.clear()
    _stepper._cleanup_cmp()

def _cleanup_autosplit():
    """Delete any auto-split ligand objects from a previous ci_setup call."""
    for name in list(_created_objects):
        if name.startswith(_AUTOSPLIT_PREFIX):
            try: cmd.delete(name)
            except Exception: pass
            _created_objects.discard(name)

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

# These run tens of thousands of times per pose, always on 3-vectors, where
# numpy's per-call overhead dominates: np.linalg.norm(np.array(a) - np.array(b))
# measures ~2.8 us against ~0.14 us for the scalar form below.  numpy is still
# used where it pays for itself (the SVD in _ring_planarity_rmsd).

def _norm(v):
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return [v[0] / mag, v[1] / mag, v[2] / mag] if mag > 1e-9 else list(v)

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def _dist(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def _dist2(a, b):
    """Squared distance — lets the pairwise loop reject most pairs without sqrt."""
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz

def _centroid(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n,
            sum(p[1] for p in pts) / n,
            sum(p[2] for p in pts) / n)

def _angle_normals(n1, n2):
    c = abs(_dot(_norm(n1), _norm(n2)))
    return math.degrees(math.acos(min(1.0, max(0.0, c))))

# ---------------------------------------------------------------------------
# Ring detection
# ---------------------------------------------------------------------------

def _sym(atom):
    """Element symbol, normalised.  PDB writes 'CL'/'BR', SDF writes 'Cl'/'Br'."""
    return atom.symbol.strip().capitalize()


def _adjacency(model):
    """Bond adjacency of a chempy model as {index: {neighbour indices}}."""
    adj = defaultdict(set)
    for bond in model.bond:
        i, j = bond.index
        adj[i].add(j); adj[j].add(i)
    return adj


def _nonpolar_h_indices(model):
    """Return indices of H atoms bonded only to C (nonpolar H, excluded from clash detection)."""
    adj = _adjacency(model)
    nonpolar = set()
    for i, a in enumerate(model.atom):
        if _sym(a) != "H":
            continue
        if not any(_sym(model.atom[nb]) in {"N", "O", "S", "F"} for nb in adj[i]):
            nonpolar.add(i)
    return nonpolar


def _ring_planarity_rmsd(coords):
    """Compute RMSD of ring atoms from their best-fit plane.
    Aromatic rings: < 0.1 A.  Cyclohexyl chair: ~ 0.5 A."""
    if np is not None:
        pts = np.array(coords)
        cen = pts.mean(axis=0)
        pts_c = pts - cen
        # SVD to find best-fit plane normal
        _, _, vh = np.linalg.svd(pts_c)
        normal = vh[2]  # smallest singular value = plane normal
        dists = pts_c @ normal
        return float(np.sqrt(np.mean(dists ** 2)))
    else:
        # Pure-python fallback: use ring normal from Newell's method
        n = len(coords)
        cen = [sum(c[i] for c in coords) / n for i in range(3)]
        nm = [0.0, 0.0, 0.0]
        for i in range(n):
            j = (i + 1) % n
            ci, cj = coords[i], coords[j]
            nm[0] += (ci[1] - cj[1]) * (ci[2] + cj[2])
            nm[1] += (ci[2] - cj[2]) * (ci[0] + cj[0])
            nm[2] += (ci[0] - cj[0]) * (ci[1] + cj[1])
        nm = _norm(nm)
        # Distance of each point from plane through centroid
        ss = 0.0
        for c in coords:
            d = sum((c[i] - cen[i]) * nm[i] for i in range(3))
            ss += d * d
        return math.sqrt(ss / n)

def _find_aromatic_rings(model):
    n = len(model.atom)
    adj = _adjacency(model)

    ring_elems = {"C", "N", "O", "S"}
    rings: List[List[int]] = []
    seen: Set[frozenset] = set()

    for start in range(n):
        if _sym(model.atom[start]) not in ring_elems:
            continue
        _dfs_rings(adj, start, start, [start], set(), rings, seen, model, 6)

    out = []
    for ring in rings:
        if len(ring) not in (5, 6):
            continue
        ok = True
        for idx in ring:
            a = model.atom[idx]
            if _sym(a) not in ring_elems:
                ok = False; break
            # sp2 carbon: exactly 3 total bonds (2 ring + 1 substituent/H)
            # sp3 carbon: 4 total bonds → not aromatic
            if _sym(a) == "C":
                total_nb = len(adj[idx])
                if total_nb > 3:
                    ok = False; break
        if not ok:
            continue
        # Planarity check: aromatic rings are flat (RMSD < 0.15 A)
        coords = [model.atom[i].coord for i in ring]
        if _ring_planarity_rmsd(coords) > 0.15:
            continue
        out.append(ring)
    return out, adj


def _dfs_rings(adj, start, cur, path, vis, rings, seen, model, mx):
    if len(path) > mx:
        return
    for nb in adj[cur]:
        if nb == start and len(path) >= 5:
            fs = frozenset(path)
            if fs not in seen:
                seen.add(fs); rings.append(list(path))
        elif nb not in vis and nb > start:
            vis.add(nb); path.append(nb)
            _dfs_rings(adj, start, nb, path, vis, rings, seen, model, mx)
            path.pop(); vis.discard(nb)


def _ring_normal(coords):
    n = len(coords)
    nm = [0.0, 0.0, 0.0]
    for i in range(n):
        j = (i + 1) % n
        ci, cj = coords[i], coords[j]
        nm[0] += (ci[1] - cj[1]) * (ci[2] + cj[2])
        nm[1] += (ci[2] - cj[2]) * (ci[0] + cj[0])
        nm[2] += (ci[0] - cj[0]) * (ci[1] + cj[1])
    return _norm(nm)


def _get_rings_info(model):
    rings, adj = _find_aromatic_rings(model)
    out = []
    for r in rings:
        coords = [model.atom[i].coord for i in r]
        out.append((r, _centroid(coords), _ring_normal(coords)))
    return out, adj


def _get_aromatic_ch_atoms(model, rings, adj):
    ring_atoms = set()
    for r in rings:
        ring_atoms.update(r)
    ch_atoms = []
    for idx in ring_atoms:
        a = model.atom[idx]
        if _sym(a) != "C":
            continue
        for nb in adj[idx]:
            if _sym(model.atom[nb]) == "H":
                ch_atoms.append((a, tuple(a.coord), tuple(model.atom[nb].coord)))
                break
    return ch_atoms


# ---------------------------------------------------------------------------
# Ligand formal charges
# ---------------------------------------------------------------------------

def _cyclic_atoms(heavy, indices):
    """Atoms surviving iterative removal of terminal atoms — the cyclic core.

    A cheap stand-in for full ring perception: every ring atom survives, and so
    do linkers running between two rings.  Used only to keep ring nitrogens out
    of the "basic amine" bucket, where a linker N is an amide or aniline anyway.
    """
    deg = {i: len(heavy[i] & indices) for i in indices}
    stack = [i for i in indices if deg[i] <= 1]
    pruned = set()
    while stack:
        i = stack.pop()
        if i in pruned:
            continue
        pruned.add(i)
        for j in heavy[i] & indices:
            if j not in pruned:
                deg[j] -= 1
                if deg[j] <= 1:
                    stack.append(j)
    return indices - pruned


def _ligand_charges(model, aromatic_atoms=frozenset()):
    """(positive, negative) atom-index sets for a ligand model.

    When the file supplies formal charges — SDF, mol2 — they are used verbatim.
    PDB has no charge column, so a ligand read out of a complex arrives entirely
    neutral, and every salt bridge plus the ligand side of pi-cation silently
    stops firing.  In that case infer the groups ionised at physiological pH,
    conservatively:

      negative   carboxylate / phosphate / sulfonate: a terminal O on a C, P or
                 S that carries at least two such terminal O
      positive   quaternary N; guanidinium / amidinium (N on a carbon bearing
                 two or more N); aliphatic amines outside the ring system that
                 are neither amides nor anilines

    Deliberately silent on ring nitrogens (imidazole, pyridine) and on anilines:
    their basicity depends on the ring, and guessing wrong invents interactions
    that are not there.
    """
    n = len(model.atom)
    charges = [int(getattr(a, "formal_charge", 0) or 0) for a in model.atom]
    if any(charges):
        return ({i for i in range(n) if charges[i] > 0},
                {i for i in range(n) if charges[i] < 0})

    adj = _adjacency(model)
    el = [_sym(a) for a in model.atom]
    heavy = {i: {j for j in adj[i] if el[j] != "H"} for i in range(n)}
    heavy_idx = {i for i in range(n) if el[i] != "H"}
    core = _cyclic_atoms(heavy, heavy_idx)

    pos: Set[int] = set()
    neg: Set[int] = set()
    for i in range(n):
        e = el[i]
        hv = heavy[i]
        if e == "O":
            if len(hv) != 1:
                continue
            (x,) = tuple(hv)
            if el[x] not in ("C", "P", "S"):
                continue
            if sum(1 for o in heavy[x]
                   if el[o] == "O" and len(heavy[o]) == 1) >= 2:
                neg.add(i)
        elif e == "N":
            if len(adj[i]) >= 4:
                pos.add(i)          # quaternary / protonated primary amine
                continue
            if any(el[x] in ("N", "O", "S") for x in hv):
                continue            # nitro, sulfonamide, hydrazine, N-oxide
            amide = guanidine = False
            for x in hv:
                if el[x] != "C":
                    continue
                # A carbon double-bonded to a terminal O or S is an amide,
                # thioamide, urea or thiourea centre.  Testing this *before*
                # the two-nitrogen rule matters: a urea (N-CO-N) has the same
                # N-C-N skeleton as an amidine but is not basic at all —
                # biotin's ureido nitrogens were being called cations.
                if any(el[o] in ("O", "S") and len(heavy[o]) == 1
                       for o in heavy[x]):
                    amide = True
                elif sum(1 for nn in heavy[x] if el[nn] == "N") >= 2:
                    guanidine = True
            if guanidine and i not in aromatic_atoms:
                pos.add(i)
            elif (not amide and hv and i not in core and i not in aromatic_atoms
                  and all(el[x] == "C" and x not in aromatic_atoms for x in hv)):
                pos.add(i)
    return pos, neg


# ---------------------------------------------------------------------------
# Interaction result (for non-hbond types detected geometrically)
# ---------------------------------------------------------------------------

class InteractionResult:
    def __init__(self):
        # H-bonds are whatever PyMOL's polar-contact detection actually drew:
        # visualize() reads the pairs back off the distance object and fills
        # these in, so the summary can never disagree with the picture.
        self.hbond_count: int = 0
        self.hbonds: List[dict] = []
        self.hbonds_rejected: int = 0   # dropped by the D-H...A angle filter
        # (x, y, z) rounded to 2 dp -> (label, element), so visualize() can both
        # name the endpoints PyMOL chose and measure the geometry at them.
        self.atom_info: Dict[tuple, tuple] = {}
        self.halogen: List[dict] = []
        self.salt_bridges: List[dict] = []
        self.arom_hbonds: List[dict] = []
        self.pipi: List[dict] = []
        self.pi_cation: List[dict] = []
        self.clash_good: List[dict] = []
        self.clash_bad: List[dict] = []
        self.clash_ugly: List[dict] = []
        self.water_bridges: List[dict] = []


# ---------------------------------------------------------------------------
# Detection (non-hbond interactions only; H-bonds use PyMOL mode=2)
# ---------------------------------------------------------------------------

def detect_interactions(
    lig_sel, prot_sel, state=-1,
    do_halogen=True, do_salt=True, do_arom_hb=True,
    do_pipi=True, do_pi_cation=True,
    do_clash_good=False, do_clash_bad=False, do_clash_ugly=False,
    do_water=True,
) -> InteractionResult:
    """Detect every enabled interaction type between one pose and the receptor."""
    # Every `within` below must run against a single-state stand-in for the pose,
    # or PyMOL walks all 1600 states of a docking run for each one.
    snap_sel, snaps = _snapshot_ligands(lig_sel, state)
    try:
        return _detect_interactions(
            lig_sel, snap_sel, prot_sel, state,
            do_halogen, do_salt, do_arom_hb, do_pipi, do_pi_cation,
            do_clash_good, do_clash_bad, do_clash_ugly, do_water)
    finally:
        _delete_snapshots(snaps)


def _detect_interactions(
    lig_sel, snap_sel, prot_sel, state,
    do_halogen, do_salt, do_arom_hb, do_pipi, do_pi_cation,
    do_clash_good, do_clash_bad, do_clash_ugly, do_water,
) -> InteractionResult:

    result = InteractionResult()
    search = max(HALOGEN_DIST_MAX, SALT_BRIDGE_DIST_MAX,
                 PI_CATION_DIST_MAX, PIPI_ETF_DIST_MAX) + 2.0

    _TMP = "_ci_tmp"
    try:
        lig_model = cmd.get_model(lig_sel, state=state)
        # cmd.select evaluates proximity at the current global state (ligand pose),
        # then cmd.get_model extracts protein coords at state=1 (protein is single-state).
        cmd.select(_TMP, f"({prot_sel}) within {search} of ({snap_sel})",
                   state=1)
        prot_model = cmd.get_model(_TMP, state=1)
    finally:
        try: cmd.delete(_TMP)
        except Exception: pass
    if not lig_model.atom or not prot_model.atom:
        return result

    # (index, atom, coord, element).  The element is normalised once here rather
    # than inside the pairwise loop, which runs tens of thousands of times.
    lig_atoms = [(i, a, tuple(a.coord), _sym(a))
                 for i, a in enumerate(lig_model.atom)]
    prot_atoms = [(i, a, tuple(a.coord), _sym(a))
                  for i, a in enumerate(prot_model.atom)]

    do_any_clash = do_clash_good or do_clash_bad or do_clash_ugly
    if do_any_clash:
        lig_nonpolar_h = _nonpolar_h_indices(lig_model)
        prot_nonpolar_h = _nonpolar_h_indices(prot_model)
    else:
        lig_nonpolar_h = prot_nonpolar_h = set()

    def _il(a): return f"{a.resn} {a.name}"
    def _ip(a): return f"{a.chain}/{a.resn}{a.resi}.{a.name}"

    # Endpoint -> label map, so visualize() can name the atoms in the H-bonds
    # PyMOL drew for itself without going back to the PyMOL API to identify them.
    for _i, a, c, e in lig_atoms:
        result.atom_info[(round(c[0], 2), round(c[1], 2), round(c[2], 2))] = (_il(a), e)
    for _i, a, c, e in prot_atoms:
        result.atom_info[(round(c[0], 2), round(c[1], 2), round(c[2], 2))] = (_ip(a), e)

    # --- Ligand rings and charges (the pairwise loop needs the charges) ------
    # Done once, up front, from the model already in hand.
    need_rings = do_pipi or do_arom_hb or do_pi_cation
    lig_rings_info: list = []
    lig_adj = None
    if need_rings:
        try:
            lig_rings_info, lig_adj = _get_rings_info(lig_model)
        except Exception:
            pass
    lig_aromatic = {i for r, _, _ in lig_rings_info for i in r}

    lig_pos: Set[int] = set()
    lig_neg: Set[int] = set()
    if do_salt or do_pi_cation:
        try:
            lig_pos, lig_neg = _ligand_charges(lig_model, lig_aromatic)
        except Exception:
            pass

    # --- Pairwise ---
    # A single squared cutoff gates every per-pair test, so the large majority of
    # pairs cost three subtractions and one comparison rather than a sqrt and a
    # chain of element lookups.
    pair_cut = max(HALOGEN_DIST_MAX, SALT_BRIDGE_DIST_MAX,
                   CLASH_DIST_MAX if do_any_clash else 0.0)
    pair_cut2 = pair_cut * pair_cut

    for li, la, lc, le in lig_atoms:
        lx, ly, lz = lc
        l_halogen = le in HALOGEN_DONORS
        l_hal_acc = le in HALOGEN_ACCEPTORS
        l_pos = li in lig_pos
        l_neg = li in lig_neg
        l_clashable = do_any_clash and li not in lig_nonpolar_h
        l_vdw = VDW_RADII.get(le, 1.70)
        for pi, pa, pc, pe in prot_atoms:
            dx = lx - pc[0]; dy = ly - pc[1]; dz = lz - pc[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > pair_cut2:
                continue
            d = math.sqrt(d2)

            if do_halogen and d <= HALOGEN_DIST_MAX:
                if ((l_halogen and pe in HALOGEN_ACCEPTORS) or
                        (pe in HALOGEN_DONORS and l_hal_acc)):
                    result.halogen.append({
                        "p1": lc, "p2": pc, "dist": d,
                        "info1": _il(la), "info2": _ip(pa)})

            if do_salt and (l_pos or l_neg) and d <= SALT_BRIDGE_DIST_MAX:
                ppos = (pa.resn in ("ARG", "LYS", "HIS", "HID", "HIE", "HIP")
                        and pa.name in ("NH1", "NH2", "NE", "NZ", "ND1", "NE2"))
                pneg = ((pa.resn == "ASP" and pa.name in ("OD1", "OD2"))
                        or (pa.resn == "GLU" and pa.name in ("OE1", "OE2")))
                if (l_pos and pneg) or (l_neg and ppos):
                    result.salt_bridges.append({
                        "p1": lc, "p2": pc, "dist": d,
                        "info1": _il(la), "info2": _ip(pa)})

            if l_clashable and pi not in prot_nonpolar_h:
                vdw = l_vdw + VDW_RADII.get(pe, 1.70)
                if d <= vdw * CLASH_GOOD_FRAC:
                    frac = d / vdw if vdw > 0 else 1.0
                    if frac < CLASH_UGLY_FRAC:
                        if do_clash_ugly:
                            result.clash_ugly.append({
                                "p1": lc, "p2": pc, "dist": d, "vdw": vdw,
                                "quality": "ugly",
                                "info1": _il(la), "info2": _ip(pa)})
                    elif frac < CLASH_BAD_FRAC:
                        if do_clash_bad:
                            result.clash_bad.append({
                                "p1": lc, "p2": pc, "dist": d, "vdw": vdw,
                                "quality": "bad",
                                "info1": _il(la), "info2": _ip(pa)})
                    elif do_clash_good and le != "H" and pe != "H":
                        result.clash_good.append({
                            "p1": lc, "p2": pc, "dist": d, "vdw": vdw,
                            "quality": "good",
                            "info1": _il(la), "info2": _ip(pa)})

    # --- Protein ring detection ---
    prot_rings_info: list = []
    _prm = None
    _prm_adj = None
    if need_rings:
        try:
            prs = (f"({prot_sel} within {PI_CATION_DIST_MAX+3} of ({snap_sel}))"
                   f" and (resn PHE+TYR+TRP+HIS+HIE+HID+HIP)")
            cmd.select(_TMP, prs, state=1)
            _prm = cmd.get_model(_TMP, state=1)
            prot_rings_info, _prm_adj = _get_rings_info(_prm)
        except Exception:
            pass
        finally:
            try: cmd.delete(_TMP)
            except Exception: pass

    # Pi-pi
    if do_pipi and lig_rings_info and prot_rings_info:
        for _, lrc, lrn in lig_rings_info:
            for pri, prc, prn in prot_rings_info:
                d = _dist(lrc, prc)
                ang = _angle_normals(lrn, prn)
                a0 = _prm.atom[pri[0]]
                info = f"{a0.chain}/{a0.resn}{a0.resi}"
                if d <= PIPI_FTF_DIST_MAX and ang <= PIPI_FTF_ANGLE_MAX:
                    result.pipi.append({
                        "p1": list(lrc), "p2": list(prc), "dist": d,
                        "angle": ang, "type": "face-to-face",
                        "info1": "lig ring", "info2": info})
                elif (d <= PIPI_ETF_DIST_MAX and
                      PIPI_ETF_ANGLE_MIN <= ang <= PIPI_ETF_ANGLE_MAX):
                    result.pipi.append({
                        "p1": list(lrc), "p2": list(prc), "dist": d,
                        "angle": ang, "type": "edge-to-face",
                        "info1": "lig ring", "info2": info})

    # Aromatic H-bonds: aromatic C-H ... acceptor (O/N/S)
    if do_arom_hb:
        if lig_adj and lig_rings_info:
            lig_ring_indices = [r for r, _, _ in lig_rings_info]
            lig_ch = _get_aromatic_ch_atoms(lig_model, lig_ring_indices, lig_adj)
            for ca, cc, hc in lig_ch:
                for _pi, pa, pc, pe in prot_atoms:
                    if pe not in AROM_HBOND_ACCEPTORS:
                        continue
                    d = _dist(cc, pc)
                    if d <= AROM_HBOND_DIST_MAX:
                        vh_c = [cc[i] - hc[i] for i in range(3)]
                        vh_a = [pc[i] - hc[i] for i in range(3)]
                        dot = _dot(_norm(vh_c), _norm(vh_a))
                        angle = math.degrees(math.acos(
                            min(1.0, max(-1.0, dot))))
                        if angle > 120.0:
                            result.arom_hbonds.append({
                                "p1": cc, "p2": pc, "dist": d,
                                "info1": f"{ca.resn} {ca.name}",
                                "info2": _ip(pa)})

        if _prm and _prm_adj and prot_rings_info:
            prot_ring_indices = [r for r, _, _ in prot_rings_info]
            prot_ch = _get_aromatic_ch_atoms(_prm, prot_ring_indices, _prm_adj)
            for ca, cc, hc in prot_ch:
                for _li, la, lc, le in lig_atoms:
                    if le not in AROM_HBOND_ACCEPTORS:
                        continue
                    d = _dist(cc, lc)
                    if d <= AROM_HBOND_DIST_MAX:
                        vh_c = [cc[i] - hc[i] for i in range(3)]
                        vh_a = [lc[i] - hc[i] for i in range(3)]
                        dot = _dot(_norm(vh_c), _norm(vh_a))
                        angle = math.degrees(math.acos(
                            min(1.0, max(-1.0, dot))))
                        if angle > 120.0:
                            result.arom_hbonds.append({
                                "p1": cc, "p2": lc, "dist": d,
                                "info1": f"{ca.chain}/{ca.resn}{ca.resi}.{ca.name}",
                                "info2": _il(la)})

    # Pi-cation
    if do_pi_cation:
        pcs = (f"({prot_sel} within {PI_CATION_DIST_MAX+1} of ({snap_sel}))"
               f" and ((resn ARG and name CZ) or (resn LYS and name NZ))")
        try:
            cmd.select(_TMP, pcs, state=1)
            pcm = cmd.get_model(_TMP, state=1)
            for pa in pcm.atom:
                pac = tuple(pa.coord)
                for _, lrc, _ in lig_rings_info:
                    d = _dist(pac, lrc)
                    if d <= PI_CATION_DIST_MAX:
                        result.pi_cation.append({
                            "p1": list(lrc), "p2": pac, "dist": d,
                            "info1": "lig ring", "info2": _ip(pa)})
        except Exception:
            pass
        finally:
            try: cmd.delete(_TMP)
            except Exception: pass
        if _prm:
            for _li, la, lc, _le in lig_atoms:
                if _li in lig_pos:
                    for pri, prc, _ in prot_rings_info:
                        d = _dist(lc, prc)
                        if d <= PI_CATION_DIST_MAX:
                            a0 = _prm.atom[pri[0]]
                            result.pi_cation.append({
                                "p1": lc, "p2": list(prc), "dist": d,
                                "info1": _il(la),
                                "info2": f"{a0.chain}/{a0.resn}{a0.resi} ring"})

    def _dedup(items):
        """Remove interactions with identical spatial endpoints (handles altloc duplicates)."""
        seen = set()
        out = []
        for it in items:
            key = (tuple(round(x, 1) for x in it["p1"]),
                   tuple(round(x, 1) for x in it["p2"]))
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out

    result.halogen      = _dedup(result.halogen)
    result.salt_bridges = _dedup(result.salt_bridges)
    result.arom_hbonds  = _dedup(result.arom_hbonds)
    result.pipi         = _dedup(result.pipi)
    result.pi_cation    = _dedup(result.pi_cation)
    result.clash_good   = _dedup(result.clash_good)
    result.clash_bad    = _dedup(result.clash_bad)
    result.clash_ugly   = _dedup(result.clash_ugly)

    # --- Water-mediated H-bonds ---
    if do_water:
        _TMP_W = "_ci_tmp_w"
        try:
            cmd.select(_TMP_W,
                       f"(resn HOH+WAT+H2O+SOL) within 3.5 of ({snap_sel})",
                       state=1)
            wat_model = cmd.get_model(_TMP_W, state=1)
            for wa in wat_model.atom:
                if _sym(wa) != "O":
                    continue
                wc = tuple(wa.coord)
                lig_match = min(
                    ((la, lc, _dist(wc, lc)) for _, la, lc, le in lig_atoms
                     if le in HBOND_ELEMENTS),
                    key=lambda x: x[2], default=None)
                if lig_match is None or lig_match[2] > 3.5:
                    continue
                prot_match = min(
                    ((pa, pc, _dist(wc, pc)) for _, pa, pc, pe in prot_atoms
                     if pe in HBOND_ELEMENTS),
                    key=lambda x: x[2], default=None)
                if prot_match is None or prot_match[2] > 3.5:
                    continue
                la, lc, dl = lig_match
                pa, pc, dp = prot_match
                result.water_bridges.append({
                    "p1": lc, "p_wat": wc, "p2": pc,
                    "dist": dl + dp, "d_lig": dl, "d_prot": dp,
                    "info1": _il(la), "info2": _ip(pa),
                    "info_wat": f"HOH {wa.chain}/{wa.resi}",
                })
        except Exception:
            pass
        finally:
            try: cmd.delete(_TMP_W)
            except Exception: pass

        def _dedup_wb(items):
            seen = set()
            out = []
            for it in items:
                key = (tuple(round(x, 1) for x in it["p1"]),
                       tuple(round(x, 1) for x in it["p_wat"]),
                       tuple(round(x, 1) for x in it["p2"]))
                if key not in seen:
                    seen.add(key); out.append(it)
            return out

        result.water_bridges = _dedup_wb(result.water_bridges)

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _clear_contacts():
    # _OBJ_SURF and _OBJ_CARVE are the pocket shell, not interaction geometry —
    # _OBJ_CARVE is what the surface carves against, so dropping it here leaves
    # surface_carve_selection dangling and the surface fails to render.
    keep = {_OBJ_SURF, _OBJ_CARVE}
    keep.update(n for n in _created_objects if n.startswith(_AUTOSPLIT_PREFIX))
    for name in list(_created_objects):
        if name not in keep:
            try: cmd.delete(name)
            except Exception: pass
            _created_objects.discard(name)


def _style(name, color, radius=DASH_RADIUS, gap=DASH_GAP, length=DASH_LENGTH):
    cmd.set("dash_radius", radius, name)
    cmd.set("dash_gap", gap, name)
    cmd.set("dash_length", length, name)
    cmd.color(color, name)
    cmd.set("label_color", "white", name)
    cmd.set("label_size", LABEL_SIZE, name)


def _add_pair_points(pts, pairs):
    """Create the anchor pseudoatoms for every pair in one PyMOL call.

    cmd.pseudoatom rebuilds the whole object on each call, so adding the points
    one at a time costs about twice what a single load_model of the finished set
    does (0.62 s vs 0.27 s for 600 pairs).  Symbol "PS" keeps the points out of
    the organic/polymer selectors the rest of the plugin relies on.
    """
    from chempy import Atom
    from chempy.models import Indexed

    model = Indexed()
    for i, (_name, p1, p2) in enumerate(pairs):
        for atom_name, pos in (("L", p1), ("P", p2)):
            a = Atom()
            a.name = atom_name
            a.resn = "PSD"
            a.resi = str(i)
            a.chain = "X"
            a.symbol = "PS"
            a.hetatm = 1
            a.coord = [float(pos[0]), float(pos[1]), float(pos[2])]
            model.atom.append(a)
    try: cmd.delete(pts)
    except Exception: pass
    cmd.load_model(model, pts, 1)


def _distance_object_pairs(name, state=0):
    """Endpoint coordinate pairs of a PyMOL distance object, for one state.

    cmd.distance returns the *average* distance — not a count — and offers no
    way to ask which pairs it drew, so read them back out of the session data.
    The object carries one point set per state (None for states it never
    computed); state <= 0 means the current global state.  Returns [] if the
    session layout is not the one we expect, so a PyMOL version change degrades
    to "no listing" rather than to wrong listings.
    """
    try:
        dsets = cmd.get_session(name, partial=1)["names"][0][5][2]
    except Exception:
        return []
    if not dsets:
        return []
    if not state or state <= 0:
        try: state = cmd.get_state()
        except Exception: state = 1
    dset = None
    if 0 <= state - 1 < len(dsets) and dsets[state - 1]:
        dset = dsets[state - 1]
    else:
        present = [d for d in dsets if d]
        if len(present) == 1:      # single-state object shown at any state
            dset = present[0]
    if not dset:
        return []
    try:
        flat = dset[1] or []
        npts = min(int(dset[0]), len(flat) // 3)
        return [(tuple(flat[i * 3:i * 3 + 3]), tuple(flat[i * 3 + 3:i * 3 + 6]))
                for i in range(0, npts - 1, 2)]
    except Exception:
        return []


def _coord_key(p):
    return (round(p[0], 2), round(p[1], 2), round(p[2], 2))


def _hbond_dha_angle(p1, p2, info):
    """True D-H...A angle in degrees for a drawn polar contact, or None.

    PyMOL draws these proton-first when h_bond_from_proton is on, so one endpoint
    is usually the hydrogen itself; the donor is then the nearest heavy atom to
    it.  Returns None when there is no proton to measure from — a receptor loaded
    without hydrogens, or h_bond_from_proton switched off — so those contacts are
    left alone rather than silently dropped.
    """
    k1, k2 = _coord_key(p1), _coord_key(p2)
    e1 = info.get(k1, ("", ""))[1]
    e2 = info.get(k2, ("", ""))[1]
    if e1 == "H":
        h, acc = p1, p2
    elif e2 == "H":
        h, acc = p2, p1
    else:
        return None
    donor, best = None, 1.35
    for coord, (_label, elem) in info.items():
        if elem == "H":
            continue
        d = _dist(coord, h)
        if d < best:
            best, donor = d, coord
    if donor is None:
        return None
    v1 = _norm([donor[i] - h[i] for i in range(3)])
    v2 = _norm([acc[i] - h[i] for i in range(3)])
    return math.degrees(math.acos(min(1.0, max(-1.0, _dot(v1, v2)))))


def visualize(lig_sel, prot_sel, result: InteractionResult,
              show_hbonds=True, show_labels=True, state=-1,
              name_prefix="", clear=True, hbond_min_angle=HBOND_MIN_DHA_ANGLE):
    """Visualize all interactions.

    H-bonds are created via PyMOL's built-in polar contact detection
    (cmd.distance mode=2), then read back off the resulting object so that
    result.hbonds lists exactly the contacts that were drawn.

    state:       pose state.  Passed to cmd.distance so only that state's polar
                 contacts are computed — the default computes every state of the
                 object, which on a 200-pose SDF costs 11.9 ms per call against
                 0.3 ms for one state.
    hbond_min_angle: drop polar contacts whose D-H...A angle is below this (deg).
                 0 keeps everything PyMOL reported.
    name_prefix: prepended to all PyMOL object names (used for reference
                 ligand so its objects are distinct from pose objects).
    clear:       call _clear_contacts() before drawing (set False when
                 adding reference interactions after pose interactions).
    """
    _register_colors()
    if clear:
        _clear_contacts()

    # Thinner dashes for reference ligand interactions (visual distinction)
    r_scale = 0.65 if name_prefix else 1.0
    hb_state = state if state and state > 0 else 0

    # --- H-bonds: PyMOL finds them, we vet the geometry, then we draw them ---
    # The native mode=2 object is used only to *find* candidates and is deleted
    # again; the survivors are redrawn through the same pseudoatom machinery as
    # every other type.  Filtering without redrawing would put dashes on screen
    # that the summary does not list, which is the divergence v1.6 removed.
    hb_name = name_prefix + _INTERACTION_NAMES["hbonds"]
    result.hbonds = []
    result.hbond_count = 0
    result.hbonds_rejected = 0
    hb_pairs: List[tuple] = []
    if show_hbonds:
        try:
            cmd.distance(hb_name, lig_sel, prot_sel, mode=2, state=hb_state)
            raw = _distance_object_pairs(hb_name, hb_state)
        except Exception:
            raw = []
        try: cmd.delete(hb_name)
        except Exception: pass

        info = result.atom_info
        # Collapse duplicate endpoints, the same way _dedup() does for every
        # other type.  Altlocs produce them, and so does the source copy that
        # _auto_split_ligands leaves in the session: a second ligand at
        # identical coordinates makes PyMOL report each contact twice.
        # Unordered, since the two dashes can arrive donor- and acceptor-first.
        seen: Set[tuple] = set()
        for p1, p2 in raw:
            k1, k2 = _coord_key(p1), _coord_key(p2)
            key = (k1, k2) if k1 <= k2 else (k2, k1)
            if key in seen:
                continue
            seen.add(key)
            angle = _hbond_dha_angle(p1, p2, info)
            if (hbond_min_angle and angle is not None
                    and angle < hbond_min_angle):
                result.hbonds_rejected += 1
                continue
            hb_pairs.append((p1, p2))
            result.hbonds.append({
                "p1": p1, "p2": p2, "dist": _dist(p1, p2), "angle": angle,
                "info1": info.get(k1, ("", ""))[0],
                "info2": info.get(k2, ("", ""))[0]})
        result.hbond_count = len(result.hbonds)

    # --- All other types via pseudoatom pairs ---
    pts = _OBJ_REF_PTS if name_prefix else _OBJ_PTS
    N = {k: name_prefix + v for k, v in _INTERACTION_NAMES.items()}

    # Collect every pair first, then build the anchor object in a single call.
    pending: List[tuple] = []
    styles: List[tuple] = []

    def _draw(items, obj_name, color, **kw):
        if not items:
            return
        _track(obj_name)
        for it in items:
            pending.append((obj_name, it["p1"], it["p2"]))
        styles.append((obj_name, color, kw))

    if hb_pairs:
        _track(hb_name)
        for _p1, _p2 in hb_pairs:
            pending.append((hb_name, _p1, _p2))
        styles.append((hb_name, "ci_hbond", dict(radius=DASH_RADIUS * r_scale)))

    _draw(result.halogen,      N["halogen"],  "ci_halogen",
          radius=DASH_RADIUS * r_scale)
    _draw(result.salt_bridges, N["salt"],     "ci_salt",
          radius=DASH_RADIUS * r_scale)
    _draw(result.arom_hbonds,  N["arom_hb"],  "ci_arom_hb",
          radius=DASH_RADIUS * r_scale)

    _draw(result.pipi,         N["pipi"],     "ci_pipi",
          gap=0.30, length=0.25, radius=0.05 * r_scale)
    _draw(result.pi_cation,    N["pi_cation"],"ci_pi_cat",
          gap=0.30, length=0.25, radius=0.05 * r_scale)

    _draw(result.clash_good,   N["clash_good"],  "ci_clash_good",
          gap=0.15, length=0.10, radius=DASH_RADIUS * r_scale)
    _draw(result.clash_bad,    N["clash_bad"],   "ci_clash_bad",
          gap=0.15, length=0.10, radius=DASH_RADIUS * r_scale)
    _draw(result.clash_ugly,   N["clash_ugly"],  "ci_clash_ugly",
          gap=0.15, length=0.10, radius=DASH_RADIUS * r_scale)

    # Water bridges: two dashes per bridge (lig->water, water->prot)
    wb_name = name_prefix + _INTERACTION_NAMES["water"]
    if result.water_bridges:
        _track(wb_name)
        for b in result.water_bridges:
            pending.append((wb_name, b["p1"],    b["p_wat"]))
            pending.append((wb_name, b["p_wat"], b["p2"]))
        styles.append((wb_name, "ci_water",
                       dict(radius=DASH_RADIUS * 0.8 * r_scale,
                            gap=0.20, length=0.15)))

    if pending:
        _track(pts)
        _add_pair_points(pts, pending)
        for i, (obj_name, _p1, _p2) in enumerate(pending):
            cmd.distance(obj_name,
                         f"{pts} and resi {i} and name L",
                         f"{pts} and resi {i} and name P")
        for obj_name, color, kw in styles:
            _style(obj_name, color, **kw)
        cmd.hide("everything", pts)

    # Contacts/clashes: never show distance labels (too cluttered)
    for q in ("clash_good", "clash_bad", "clash_ugly"):
        if N[q] in _created_objects:
            cmd.hide("labels", N[q])

    if not show_labels:
        for n in N.values():
            if n in _created_objects:
                cmd.hide("labels", n)


# ---------------------------------------------------------------------------
# Scene preparation
# ---------------------------------------------------------------------------

_ELEM_COLORS = [
    ("N",  "tv_blue"), ("O",  "tv_red"),   ("S",  "tv_yellow"),
    ("P",  "orange"),  ("F",  "palegreen"),
    ("Cl", "green"),   ("Br", "firebrick"), ("I",  "purple"),
]

def _apply_elem_colors(sel):
    """Apply standard element colors to heteroatoms in sel."""
    for elem, color in _ELEM_COLORS:
        try:
            cmd.color(color, f"({sel}) and elem {elem}")
        except Exception:
            pass

def _color_rainbow_elem(sel):
    """Color carbon atoms rainbow by residue number; heteroatoms by element; H white."""
    cmd.spectrum("count", "rainbow", f"({sel}) and elem C", byres=1)
    _apply_elem_colors(sel)
    cmd.color("white", f"({sel}) and elem H")


def _prepare_scene(protein_sel, ligand_sels):
    """Color protein rainbow (C atoms) + element colors, cartoon, hide non-polar H."""
    try:
        # PoseViewer frames the view itself (_zoom_to_site).  With PyMOL's own
        # auto_zoom left on, every distance object and pseudoatom that visualize()
        # creates while stepping triggers a zoom-to-fit, so the camera lurches
        # toward each pose — the "zooms on the ligand every step" complaint.
        cmd.set("auto_zoom", 0)
        cmd.hide("surface")   # remove any pre-existing surfaces before adding ours
        cmd.show("cartoon", protein_sel)
        _color_rainbow_elem(protein_sel)

        # Hide non-polar H on protein (keep polar H on N/O/S visible)
        cmd.hide("everything",
                 f"({protein_sel}) and elem H and "
                 f"not (neighbor (elem N+O+S))")

        # Show ligands as sticks, white carbons + element colors for heteroatoms
        lig_u = _lig_union(ligand_sels)
        cmd.show("sticks", lig_u)
        cmd.hide("everything",
                 f"({lig_u}) and elem H and "
                 f"not (neighbor (elem N+O+S))")
        cmd.color("white", f"({lig_u}) and elem C")
        cmd.color("white", f"({lig_u}) and elem H")
        _apply_elem_colors(lig_u)
    except Exception as e:
        print(f"PoseViewer: scene setup warning: {e}")


def _apply_lig_h(lig_sel, show):
    """Show all H on ligand as sticks, or restore default (hide non-polar H only)."""
    try:
        if show:
            cmd.show("sticks", f"({lig_sel}) and elem H")
        else:
            cmd.hide("everything",
                     f"({lig_sel}) and elem H and not (neighbor (elem N+O+S))")
    except Exception:
        pass


def _color_ref_ligand(name):
    """Color reference ligand with magenta carbons + element colors."""
    try:
        cmd.color("magenta", f"({name}) and elem C")
        _apply_elem_colors(name)
    except Exception:
        pass


def _lig_union(ligand_sels):
    """Build a PyMOL union selection string from ligand(s)."""
    if isinstance(ligand_sels, str):
        return ligand_sels
    return " or ".join(f"({l})" for l in ligand_sels)


def _snapshot_ligands(ligand_sels, state=0):
    """Single-state copies of the ligand(s), for use inside `within` selections.

    PyMOL evaluates a distance operator against *every* state of a multi-state
    object, and the cost is superlinear in the state count: selecting the
    residues within 5 A of a pose object measured 0.001 s at one state, 0.15 s
    at 100 and 6.9 s at 400.  At 1600 poses that is what makes PyMOL look hung.

    Pinning the selection to one state is not the fix — the two sides have
    different state counts, and PyMOL does not clamp, so a single-state receptor
    matches nothing at state 7 and the shell comes back empty.  Copying the
    current state of each ligand into its own single-state object makes every
    side single-state, which is correct *and* independent of how many poses are
    loaded.

    The snapshot is only half the fix.  cmd.select's default state=0 means "every
    state in the session", so it loops 1..max_states even when both operands are
    single-state — 0.13 s per call in a session whose largest object has 800
    states.  Every selection here therefore also passes state=1, which is only
    safe *because* of the snapshot: PyMOL does not clamp, so asking for state 7
    of a single-state receptor selects nothing at all.

    Returns (selection, temp_names); the caller must pass temp_names to
    _delete_snapshots().  Falls back to the plain union if copying fails, so a
    surprise here costs speed rather than function.
    """
    names = [ligand_sels] if isinstance(ligand_sels, str) else list(ligand_sels)
    temps, parts = [], []
    for i, name in enumerate(names):
        tmp = f"{_OBJ_SNAP}{i}"
        try:
            cmd.delete(tmp)
            # A single-state object — a reference ligand — is always at state 1,
            # whatever the pose slider currently reads.
            src = state if (state and state > 0) else -1
            try:
                if cmd.count_states(name) <= 1:
                    src = 1
            except Exception:
                pass
            cmd.create(tmp, name, src, 1)
            if cmd.count_atoms(tmp, state=1) == 0:
                cmd.delete(tmp)
                continue
            cmd.disable(tmp)
            temps.append(tmp)
            parts.append(f"({tmp})")
        except Exception:
            try: cmd.delete(tmp)
            except Exception: pass
    if not parts:
        return _lig_union(ligand_sels), []
    return " or ".join(parts), temps


def _delete_snapshots(temps):
    for tmp in temps:
        try: cmd.delete(tmp)
        except Exception: pass


# ---------------------------------------------------------------------------
# Residue shell
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Surface charge colouring — two styles, chosen by _stepper.surf_charge_style:
#   "tiers"  flat colour on a side chain's charged/polar functional atoms only,
#            two intensities per sign (formal charge saturated, partial pale)
#   "ramp"   smooth red->white->blue by ff14SB partial charge (below)
# ---------------------------------------------------------------------------

_HIS_RESN = "HIS+HID+HIE+HIP+HSD+HSE+HSP"
# A histidine counts as protonated (+1) only when both imidazole NH are present:
# explicitly named HIP/HSP, or an explicit-H model carrying both HD1 and HE2.
_HIS_POS  = (f"((resn HIP+HSP) or ((resn {_HIS_RESN}) "
             f"and (byres (name HD1)) and (byres (name HE2))))")

_SURF_NEG_STRONG = "(resn ASP and name OD1+OD2) or (resn GLU and name OE1+OE2)"
_SURF_NEG_WEAK   = ("(resn ASN and name OD1) or (resn GLN and name OE1) or "
                    "(resn SER and name OG) or (resn THR and name OG1) or "
                    "(resn TYR and name OH) or (resn CYS and name SG)")
_SURF_POS_STRONG = ("(resn ARG and name NH1+NH2+NE) or (resn LYS and name NZ) or "
                    f"(({_HIS_POS}) and name ND1+NE2)")
_SURF_POS_WEAK   = ("(resn ASN and name ND2) or (resn GLN and name NE2) or "
                    "(resn TRP and name NE1) or "
                    f"((resn {_HIS_RESN}) and not ({_HIS_POS}) and name ND1+NE2)")

# The ramp is the tier model made continuous: the same functional atoms carry a
# signed magnitude, the surface interpolates between them.  Raw force-field atom
# charges are no good here — they put the negative label on a guanidinium /
# ammonium nitrogen (the + lives on the hydrogens), so a cation lined up red on a
# structure with no explicit H.  For a proper integrated potential, use APBS.
_RAMP_LEVELS = ((-1.0, _SURF_NEG_STRONG),
                (-0.4, f"({_SURF_NEG_WEAK}) or (name O and polymer.protein)"),
                (+1.0, _SURF_POS_STRONG),
                (+0.4, f"({_SURF_POS_WEAK}) or (name H and polymer.protein)"))


def _color_surface_by_type(surf_obj):
    """Charge-code the pocket surface, dispatching on _stepper.surf_charge_style."""
    _register_colors()
    if getattr(_stepper, "surf_charge_style", "ramp") == "tiers":
        _color_surface_tiers(surf_obj)
    else:
        _color_surface_ramp(surf_obj)


def _color_surface_tiers(surf_obj):
    """Deep red carboxylate O, pale red amide/hydroxyl/thiol; deep blue
    guanidinium/ammonium/HIP N, pale blue amide/indole/neutral His N.  Backbone
    and everything else stay grey.
    """
    try: cmd.unset("surface_color", surf_obj)
    except Exception: pass
    cmd.color("grey80", surf_obj)
    # Weak tiers first, then the strong tiers paint over any overlap.
    for col, sel in (("ci_surf_neg_wk", _SURF_NEG_WEAK),
                     ("ci_surf_pos_wk", _SURF_POS_WEAK),
                     ("ci_surf_neg",    _SURF_NEG_STRONG),
                     ("ci_surf_pos",    _SURF_POS_STRONG)):
        try: cmd.color(col, f"({surf_obj}) and ({sel})")
        except Exception: pass


def _color_surface_ramp(surf_obj):
    """Smooth red (negative) -> white -> blue (positive) across the pocket wall.

    Same functional atoms as the tiers, given a signed magnitude; the surface
    interpolates.  Backbone carbonyl O and amide H are included here (they are
    left grey in the tiers) so the ramp shows the peptide dipole too.
    """
    try: cmd.unset("surface_color", surf_obj)
    except Exception: pass
    cmd.alter(surf_obj, "partial_charge = 0.0")
    for val, sel in _RAMP_LEVELS:
        try: cmd.alter(f"({surf_obj}) and ({sel})", f"partial_charge = {val}")
        except Exception: pass
    try:
        cmd.spectrum("partial_charge", "red_white_blue", surf_obj,
                     minimum=-1.0, maximum=1.0)
        cmd.recolor(surf_obj)
    except Exception:
        cmd.color("grey80", surf_obj)


def _create_shell(protein_sel, ligand_sels, dist=SHELL_DIST, state=0):
    """Show lines + labels for residues near the ligand, with a surface on top.

    Rebuilding means re-running cmd.create and a full surface recalculation, so
    skip it when nothing that feeds the shell has changed — every checkbox
    toggle routes through _show_current() and would otherwise pay for it.
    Surface colour and visibility are handled by their own toggles and so are
    deliberately not part of the key.
    """
    global _shell_sel, _shell_key
    lig_union = _lig_union(ligand_sels)
    surf_dist = _stepper.surf_dist

    key = (protein_sel, lig_union, dist, state, surf_dist)
    if key == _shell_key and _shell_sel is not None and _OBJ_SURF in _created_objects:
        return

    _clear_shell()

    # Single-state stand-in for the pose, then materialise the result as a named
    # selection.  Both matter: `within` against a multi-state object is
    # superlinear in the state count (see _snapshot_ligands), and keeping the
    # shell as an expression string meant every later use of it — the residue
    # label toggle, _clear_shell — paid for that `within` all over again.
    snap_sel, snaps = _snapshot_ligands(ligand_sels, state)
    try:
        cmd.select(_OBJ_SHELL,
                   f"byres (({protein_sel}) within {dist} of ({snap_sel}))",
                   enable=0, state=1)
        cmd.show("lines", _OBJ_SHELL)
        cmd.hide("lines",
                 f"({_OBJ_SHELL}) and elem H and not (neighbor (elem N+O+S))")
        cmd.label(f"({_OBJ_SHELL}) and name CA", '"%s %s" % (resn, resi)')
        _shell_sel = _OBJ_SHELL
    except Exception as e:
        print(f"PoseViewer: shell setup warning: {e}")
        _delete_snapshots(snaps)
        return

    # Pocket surface: a carved patch of the real protein molecular surface, not
    # the surface of a bag of clipped atoms.  Surfacing just the atoms within
    # SHELL_DIST closed the mesh over into a blob that swallowed the pocket
    # residues; instead build the SES on a wide residue shell (so it is correct
    # at the pocket wall), keep a single-state ligand copy to carve against, and
    # trim to the wall within surf_dist of the ligand (the negative normal-cutoff
    # keeps the patch continuous rather than dropping to fragments).  surf_dist is
    # _stepper.surf_dist, tunable from the Display group.
    try:
        cmd.create(_OBJ_CARVE, snap_sel, 1, 1)
        _track(_OBJ_CARVE)
        cmd.disable(_OBJ_CARVE)
        cmd.select(_OBJ_SHELL_ATOMS,
                   f"byres (({protein_sel}) within {surf_dist + SURF_SHELL_PAD} "
                   f"of ({snap_sel}))",
                   enable=0, state=1)
        cmd.create(_OBJ_SURF, _OBJ_SHELL_ATOMS, 1, 1)
        cmd.delete(_OBJ_SHELL_ATOMS)
        _track(_OBJ_SURF)
        cmd.hide("everything", _OBJ_SURF)
        cmd.show("surface", _OBJ_SURF)
        cmd.set("surface_carve_selection", _OBJ_CARVE, _OBJ_SURF)
        cmd.set("surface_carve_cutoff", surf_dist, _OBJ_SURF)
        cmd.set("surface_carve_normal_cutoff", -0.5, _OBJ_SURF)
        if _stepper.color_surf_by_type:
            _color_surface_by_type(_OBJ_SURF)
        else:
            cmd.set("surface_color", "grey80", _OBJ_SURF)
        cmd.set("transparency", 0.4, _OBJ_SURF)
    except Exception:
        pass
    finally:
        _delete_snapshots(snaps)

    _shell_key = key



# ---------------------------------------------------------------------------
# Ligand stepper
# ---------------------------------------------------------------------------

class LigandStepper:
    _COMPARE_PALETTE = ["cyan", "tv_orange", "forest", "hotpink", "violet", "salmon"]

    def __init__(self):
        self.protein_sel = ""
        self.ligand_objects: List[str] = []
        self.current_index = 0
        self.mode = "objects"
        self.state_object = ""
        self.last_result: Optional[InteractionResult] = None

        self.show_hbonds = True
        self.show_halogen = True
        self.show_salt = True
        self.show_arom_hb = True
        self.show_pipi = True
        self.show_pi_cation = True
        self.show_clash_good = False
        self.show_clash_bad = False
        self.show_clash_ugly = False
        self.show_labels = True
        self.hbond_min_angle: float = HBOND_MIN_DHA_ANGLE
        self.auto_zoom = True
        # Frame the pocket rather than the ligand.  Set False to go back to
        # zooming the ligand alone.
        self.zoom_to_shell: bool = True
        self.zoom_buffer: float = ZOOM_BUFFER
        # How far the pocket surface reaches around the ligand (Display group).
        self.surf_dist: float = SURF_CARVE_DIST
        self.sdf_records: list = []   # populated by ci_load_scores / GUI browse
        self.all_properties: list = []  # one dict per pose, built at setup time
        self.poses: list = []          # [(obj_name, state_1based), ...]
        self.ref_ligand: Optional[str] = None
        self.show_ref: bool = True
        self.show_pose: bool = True
        self.show_lig_h: bool = False
        self.show_surface: bool = True
        self._cmp_objs:      List[str]      = []
        self._cmp_indices:   List[int]      = []
        self._in_compare:    bool           = False
        self._obj_colors:    Dict[str, str] = {}
        self.show_cmp_hbonds: bool          = False
        self.show_water: bool               = True
        self.color_surf_by_type: bool       = True
        self.surf_charge_style: str          = "ramp"   # "ramp" or "tiers"
        # Keyed by (object, state) like `computed` below, so a bookmark keeps
        # pointing at its pose when the list is rebuilt (objects added, deleted
        # or renumbered) instead of sliding onto whatever now sits at that index.
        self._bookmarks: Set[tuple]         = set()
        # Metrics computed by _MetricsJob, keyed by (object, state) so they
        # survive the pose list being rebuilt (objects added/removed/renumbered).
        self.computed: Dict[tuple, dict]    = {}
        self._table_dirty: bool             = False
        self._sdf_mismatch: Optional[tuple] = None

    def setup_objects(self, prot, ligs, ref_lig=None):
        self.protein_sel = prot
        self.ligand_objects = ligs
        self.current_index = 0
        self.mode = "objects"
        self.ref_ligand = ref_lig
        self.poses = [(n, st) for n in ligs
                      for st in range(1, max(1, cmd.count_states(n)) + 1)]
        all_ligs = list(ligs) + ([ref_lig] if ref_lig else [])
        _prepare_scene(prot, all_ligs)
        if ref_lig:
            _color_ref_ligand(ref_lig)
        _create_shell(prot, all_ligs)
        if ligs:
            self._show_current()
        self._prefetch_all_properties()
        self._build_obj_colors()

    def setup_states(self, prot, obj):
        self.protein_sel = prot
        self.state_object = obj
        self.current_index = 0
        self.mode = "states"
        extras = []
        for n in cmd.get_names("objects"):
            if n == obj or n in _created_objects:
                continue
            try:
                if (cmd.count_atoms(f"{n} and organic", state=1) > 0 and
                        cmd.count_atoms(f"{n} and polymer.protein", state=1) == 0):
                    extras.append(n)
            except Exception:
                pass
        self.ref_ligand = extras[0] if extras else None
        n_states = cmd.count_states(obj)
        self.poses = [(obj, st) for st in range(1, n_states + 1)]
        all_ligs = [obj] + extras if extras else obj
        _prepare_scene(prot, all_ligs)
        if self.ref_ligand:
            _color_ref_ligand(self.ref_ligand)
        # Before _show_current, because the auto-zoom frames the shell: build it
        # after and the very first pose would be framed on the ligand alone.
        # The state is passed explicitly, so it no longer matters what the state
        # slider happened to read when Setup was pressed.
        _create_shell(prot, all_ligs, state=self.poses[0][1] if self.poses else 1)
        self._show_current()
        self._prefetch_all_properties()
        self._build_obj_colors()

    def _count(self):
        return len(self.poses)

    def pose_key(self, index=None):
        """(object, state) identity of a pose index, or None when out of range."""
        if index is None:
            index = self.current_index
        if 0 <= index < len(self.poses):
            return self.poses[index]
        return None

    def is_bookmarked(self, index=None):
        key = self.pose_key(index)
        return key is not None and key in self._bookmarks

    def toggle_bookmark(self, index=None):
        """Add/remove a bookmark; returns True when the pose is now bookmarked."""
        key = self.pose_key(index)
        if key is None:
            return False
        if key in self._bookmarks:
            self._bookmarks.discard(key)
            return False
        self._bookmarks.add(key)
        return True

    def table_columns(self):
        """Data column keys for the pose table, in display order."""
        seen: dict = {}
        for props in self.all_properties:
            for k in props:
                seen.setdefault(k, None)
        cols = list(seen)
        return (["_name"] if "_name" in cols else []) + [
            c for c in cols if c != "_name" and "rank" not in c.lower()]

    def _label(self):
        if not self.poses:
            return "none"
        obj, st = self.poses[self.current_index]
        base = f"{obj} state {st}" if cmd.count_states(obj) > 1 else obj
        i = self.current_index
        name = (self.all_properties[i].get("_name")
                if 0 <= i < len(self.all_properties) else None)
        return f"{base}  [{name}]" if name and name != obj else base

    def next(self):
        c = self._count()
        if c:
            self.current_index = (self.current_index + 1) % c
            self._show_current()

    def prev(self):
        c = self._count()
        if c:
            self.current_index = (self.current_index - 1) % c
            self._show_current()

    def goto(self, i):
        if 0 <= i < self._count():
            self.current_index = i
            self._show_current()

    def refresh(self):
        """Re-detect interactions for the current PyMOL state without changing it.

        In states mode (single object) syncs current_index from cmd.get_state()
        so that manual slider movement is reflected in the panel.
        """
        if self._in_compare:
            return
        if not self.poses:
            return
        if self.mode == "states":
            st = cmd.get_state()   # 1-based; follows the slider
            for i, (o, s) in enumerate(self.poses):
                if s == st:
                    self.current_index = i
                    break
            obj, st2 = self.poses[self.current_index]
            if self.auto_zoom:
                self._zoom_to_site(obj, st2)
            self._update(obj, state=st2)
        else:
            self._show_current()

    def rebuild_shell(self):
        """Rebuild the residue shell + pocket surface for the current pose.

        For use when something behind the shell changes mid-session (the
        surface reach).  Objects mode reuses _show_current, which rebuilds the
        shell anyway; states mode has to build it explicitly because there it
        is otherwise static after setup.
        """
        global _shell_key
        if not self.poses:
            return
        _shell_key = None
        if self.mode != "states":
            self._show_current()
            return
        _, st = self.poses[self.current_index]
        ligs = [self.state_object] + ([self.ref_ligand] if self.ref_ligand else [])
        _create_shell(self.protein_sel, ligs, state=st)
        if not self.show_surface and _OBJ_SURF in _created_objects:
            try: cmd.hide("surface", _OBJ_SURF)
            except Exception: pass

    def _show_current(self):
        self._cleanup_cmp()
        if not self.poses:
            return
        obj, st = self.poses[self.current_index]
        obj_names = set(cmd.get_names("objects"))
        for n in {o for o, _ in self.poses}:
            if n in obj_names:
                cmd.disable(n)
        if obj in obj_names and self.show_pose:
            cmd.enable(obj)
        cmd.set("state", st)
        _apply_lig_h(obj, self.show_lig_h)
        if self.ref_ligand:
            _apply_lig_h(self.ref_ligand, self.show_lig_h)
            try:
                if self.show_ref:
                    cmd.enable(self.ref_ligand)
                else:
                    cmd.disable(self.ref_ligand)
            except Exception:
                pass
        # In objects mode each ligand may sit in a different pocket, so rebuild
        # the shell/surface around just the current ligand (+ ref if visible).
        # Done before the zoom, which targets the shell and so needs it to
        # describe the pose about to be shown rather than the previous one.
        if self.mode == "objects":
            visible = []
            if self.show_pose:
                visible.append(obj)
            if self.ref_ligand and self.show_ref:
                visible.append(self.ref_ligand)
            if visible:
                _create_shell(self.protein_sel, visible, state=st)
                if not self.show_surface and _OBJ_SURF in _created_objects:
                    try: cmd.hide("surface", _OBJ_SURF)
                    except Exception: pass
        if self.auto_zoom:
            self._zoom_to_site(obj, st)
        self._update(obj, state=st)

    def _zoom_to_site(self, obj, st):
        """Frame the binding site rather than the ligand on its own.

        Zooming the ligand puts the camera right on top of it and the pocket
        falls outside the view, which is disorienting when stepping through
        poses.  The residue shell is the useful frame, and in states mode it is
        fixed, so the view stops jumping from pose to pose.  Falls back to the
        ligand when there is no shell — nothing within SHELL_DIST, or shell
        setup failed — or when zoom_to_shell is off.
        """
        target, state = None, st
        if self.zoom_to_shell and _shell_sel is not None:
            target, state = _shell_sel, 1
        elif self.show_pose:
            target = obj
        elif self.ref_ligand and self.show_ref:
            target, state = self.ref_ligand, 1
        if target is None:
            return
        try:
            cmd.zoom(target, buffer=self.zoom_buffer, animate=1, state=state)
        except Exception:
            pass

    def _build_obj_colors(self):
        seen = dict.fromkeys(obj for obj, _ in self.poses)
        self._obj_colors = {n: self._COMPARE_PALETTE[i % len(self._COMPARE_PALETTE)]
                            for i, n in enumerate(seen)}

    def _cleanup_cmp(self):
        for name in self._cmp_objs:
            try: cmd.delete(name)
            except Exception: pass
            _created_objects.discard(name)
        self._cmp_objs.clear()
        self._in_compare = False

    def show_comparison(self, indices: List[int]):
        """Show two poses simultaneously as colored single-state copies.

        Each pose is extracted via cmd.create so the global state slider does not
        interfere.  Interactions are hidden by default; H-bonds shown if
        show_cmp_hbonds is True, colored per source object.
        """
        self._cleanup_cmp()
        _clear_contacts()
        _register_colors()

        for obj in {o for o, _ in self.poses}:
            try: cmd.disable(obj)
            except Exception: pass

        # Two poses from the same object get slot-index colors (e.g. states mode);
        # poses from different objects use their per-object palette entry.
        n = len(indices)
        same_obj = (n >= 2 and self.poses[indices[0]][0] == self.poses[indices[1]][0])

        def _slot_color(slot: int, pose_idx: int) -> str:
            if same_obj:
                return self._COMPARE_PALETTE[slot % len(self._COMPARE_PALETTE)]
            obj, _ = self.poses[pose_idx]
            return self._obj_colors.get(obj, self._COMPARE_PALETTE[slot % len(self._COMPARE_PALETTE)])

        lig_sels: List[str] = []
        for i, pose_idx in enumerate(indices[:2]):
            obj, st = self.poses[pose_idx]
            tmp = f"_cmp_{i}"
            try:
                cmd.create(tmp, obj, st, 1)
                _track(tmp)
                self._cmp_objs.append(tmp)
                color = _slot_color(i, pose_idx)
                cmd.show("sticks", tmp)
                cmd.color(color,   f"({tmp}) and elem C")
                cmd.color("white", f"({tmp}) and elem H")
                _apply_elem_colors(tmp)
                cmd.hide("everything",
                         f"({tmp}) and elem H and not (neighbor (elem N+O+S))")
                if self.show_lig_h:
                    _apply_lig_h(tmp, True)
                lig_sels.append(tmp)
            except Exception as e:
                print(f"PoseViewer: compare copy failed for pose {pose_idx}: {e}")

        if not lig_sels:
            return

        self._in_compare  = True
        self._cmp_indices = list(indices[:2])
        self.current_index = indices[0]

        if self.ref_ligand:
            try:
                cmd.enable(self.ref_ligand) if self.show_ref else cmd.disable(self.ref_ligand)
            except Exception:
                pass
            if self.show_ref:
                _color_ref_ligand(self.ref_ligand)

        shell_ligs = lig_sels + ([self.ref_ligand] if self.ref_ligand and self.show_ref else [])
        _create_shell(self.protein_sel, shell_ligs)

        if self.show_cmp_hbonds:
            for i, (tmp, pose_idx) in enumerate(zip(lig_sels, indices[:2])):
                color = _slot_color(i, pose_idx)
                hb_name = f"_cmp_hb_{i}"
                try:
                    n_hb = cmd.distance(hb_name, tmp, self.protein_sel, mode=2)
                    if n_hb and n_hb > 0:
                        _track(hb_name)
                        cmd.set("dash_radius", DASH_RADIUS, hb_name)
                        cmd.set("dash_gap",    DASH_GAP,    hb_name)
                        cmd.set("dash_length", DASH_LENGTH, hb_name)
                        cmd.color(color, hb_name)
                        cmd.set("label_color", "white",     hb_name)
                        cmd.set("label_size",  LABEL_SIZE,  hb_name)
                        if not self.show_labels:
                            cmd.hide("labels", hb_name)
                    else:
                        try: cmd.delete(hb_name)
                        except Exception: pass
                except Exception:
                    pass
            if self.ref_ligand and self.show_ref:
                hb_ref = "_cmp_hb_ref"
                try:
                    n_hb = cmd.distance(hb_ref, self.ref_ligand, self.protein_sel, mode=2)
                    if n_hb and n_hb > 0:
                        _track(hb_ref)
                        cmd.set("dash_radius", DASH_RADIUS * 0.65, hb_ref)
                        cmd.set("dash_gap",    DASH_GAP,            hb_ref)
                        cmd.set("dash_length", DASH_LENGTH,         hb_ref)
                        cmd.color("ci_hbond", hb_ref)
                        cmd.set("label_color", "white",    hb_ref)
                        cmd.set("label_size",  LABEL_SIZE, hb_ref)
                    else:
                        try: cmd.delete(hb_ref)
                        except Exception: pass
                except Exception:
                    pass

        if self.auto_zoom and lig_sels:
            union = " or ".join(f"({s})" for s in lig_sels)
            try: cmd.zoom(union, buffer=3.0, animate=1)
            except Exception: pass

    def compare_label(self) -> str:
        parts = []
        for idx in self._cmp_indices:
            if 0 <= idx < len(self.poses):
                obj, st = self.poses[idx]
                parts.append(f"{obj}(s{st})" if cmd.count_states(obj) > 1 else obj)
        return "  vs  ".join(parts) if parts else "Compare"

    def _update(self, lig, state=-1):
        import traceback
        try:
            r = detect_interactions(
                lig, self.protein_sel, state=state,
                do_halogen=self.show_halogen,
                do_salt=self.show_salt, do_arom_hb=self.show_arom_hb,
                do_pipi=self.show_pipi, do_pi_cation=self.show_pi_cation,
                do_clash_good=self.show_clash_good,
                do_clash_bad=self.show_clash_bad,
                do_clash_ugly=self.show_clash_ugly,
                do_water=self.show_water)
            self.last_result = r
            if self.show_pose:
                visualize(lig, self.protein_sel, r,
                          show_hbonds=self.show_hbonds,
                          show_labels=self.show_labels, state=state,
                          hbond_min_angle=self.hbond_min_angle)
            else:
                _clear_contacts()
            if self.ref_ligand and self.show_ref:
                try:
                    r_ref = detect_interactions(
                        self.ref_ligand, self.protein_sel, state=1,
                        do_halogen=self.show_halogen,
                        do_salt=self.show_salt, do_arom_hb=self.show_arom_hb,
                        do_pipi=self.show_pipi, do_pi_cation=self.show_pi_cation,
                        do_clash_good=self.show_clash_good,
                        do_clash_bad=self.show_clash_bad,
                        do_clash_ugly=self.show_clash_ugly,
                        do_water=self.show_water)
                    visualize(self.ref_ligand, self.protein_sel, r_ref,
                              show_hbonds=self.show_hbonds,
                              show_labels=self.show_labels, state=1,
                              name_prefix="ref_", clear=not self.show_pose,
                              hbond_min_angle=self.hbond_min_angle)
                except Exception:
                    pass
        except Exception as e:
            msg = f"PoseViewer error: {e}\n{traceback.format_exc()}"
            print(msg)
            if _gui_window is not None:
                try:
                    from pymol.Qt import QtWidgets
                    QtWidgets.QMessageBox.warning(_gui_window, "PoseViewer", str(e))
                except Exception:
                    pass

    def _prefetch_all_properties(self):
        """Populate all_properties: one dict per pose, used by the table view."""
        self._build_all_properties()
        self._apply_computed()

    def _apply_computed(self):
        """Overlay metrics computed in-session onto the SDF/object properties.

        Copies each touched dict first: in states mode all_properties holds the
        sdf_records objects themselves, which must not be mutated.
        """
        if not self.computed or not self.all_properties:
            return
        merged = []
        for key, props in zip(self.poses, self.all_properties):
            vals = self.computed.get(key)
            if vals:
                props = dict(props)
                props.update(vals)
            merged.append(props)
        self.all_properties = merged

    def merge_metrics(self, results: dict):
        """Store freshly computed metrics and push them into the table data."""
        for key, vals in results.items():
            self.computed.setdefault(key, {}).update(vals)
        self._prefetch_all_properties()

    def invalidate_metrics(self, fields):
        """Drop computed values for `fields` (e.g. after the reference changes)."""
        dropped = False
        for vals in self.computed.values():
            for f in fields:
                if vals.pop(f, None) is not None:
                    dropped = True
        if dropped:
            self._prefetch_all_properties()
        return dropped

    def _build_all_properties(self):
        """Build all_properties from the SDF records / PyMOL object properties."""
        if not self.sdf_records:
            result = []
            for obj, st in self.poses:
                props = _get_pose_properties(obj, st)
                if not props.get("_name"):
                    props["_name"] = _pose_title(obj, st) or obj
                result.append(props)
            self.all_properties = result
            return

        if self.mode == "states":
            # Single-object states mode: records are 1-to-1 with poses in SDF
            # order.  Pad/truncate to the pose count so the table can never grow
            # rows that address no pose, or leave poses without a row.
            n = len(self.poses)
            recs = list(self.sdf_records)
            if len(recs) != n and self._sdf_mismatch != (len(recs), n):
                self._sdf_mismatch = (len(recs), n)
                print(f"PoseViewer: {len(recs)} SDF record(s) for {n} state(s) "
                      f"— table aligned to the states.")
            self.all_properties = (recs + [{}] * n)[:n]
            return

        # Objects mode with SDF loaded: match records to poses by _name.
        # Group SDF records by molecule title, preserving their in-file order.
        # That order corresponds to state indices (1st occurrence = state 1, etc.)
        # because docking software emits poses per compound in score order.
        # PyMOL converts hyphens (and other non-word chars) to underscores when
        # it creates object names from SDF titles, so normalise before matching.
        import re as _re
        from collections import defaultdict

        def _norm(s: str) -> str:
            return _re.sub(r'[^\w]', '_', s)

        by_name: dict = defaultdict(list)
        for rec in self.sdf_records:
            by_name[_norm(rec.get("_name", ""))].append(rec)

        obj_names = {o for o, _ in self.poses}
        if obj_names & set(by_name.keys()):
            # at least some _names match object names — use name-based alignment;
            # unmatched objects (e.g. a ref ligand included as a pose) get a
            # sparse entry instead of stealing a sequential SDF record
            counters: dict = defaultdict(int)
            self.all_properties = []
            for obj, _st in self.poses:
                recs = by_name.get(obj, [])
                i = counters[obj]
                self.all_properties.append(recs[i] if i < len(recs) else {"_name": obj})
                counters[obj] += 1
        else:
            # Names don't match — fall back to sequential (may mis-align)
            n = len(self.poses)
            self.all_properties = (list(self.sdf_records) + [{}] * n)[:n]

    def summary(self):
        r = self.last_result
        if r is None:
            return "No interactions detected."
        lines = [f"=== {self._label()} "
                 f"({self.current_index + 1}/{self._count()}) ==="]

        def _s(title, items, extra_fn=None):
            if not items: return
            lines.append(f"\n{title} ({len(items)}):")
            for it in items:
                ex = extra_fn(it) if extra_fn else ""
                i1 = it.get("info1", "")
                i2 = it.get("info2", "")
                lines.append(f"  {i1} -- {i2}  {it['dist']:.2f} A{ex}")

        _s("H-bonds", r.hbonds,
           lambda x: f"  {x['angle']:.0f} deg" if x.get("angle") is not None else "")
        if r.hbonds_rejected:
            lines.append(f"  ({r.hbonds_rejected} polar contact(s) below "
                         f"{self.hbond_min_angle:.0f} deg D-H...A not shown)")
        _s("Halogen bonds", r.halogen)
        _s("Salt bridges", r.salt_bridges)
        _s("Aromatic H-bonds", r.arom_hbonds)
        _s("Pi-pi", r.pipi,
           lambda x: f"  {x['angle']:.0f} deg [{x['type']}]")
        _s("Pi-cation", r.pi_cation)
        _s("Clash good", r.clash_good)
        _s("Clash bad", r.clash_bad)
        _s("Clash ugly", r.clash_ugly)

        if r.water_bridges:
            lines.append(f"\nWater bridges ({len(r.water_bridges)}):")
            for b in r.water_bridges:
                lines.append(f"  {b['info1']} -- {b['info_wat']} -- {b['info2']}"
                             f"  {b['d_lig']:.2f}+{b['d_prot']:.2f} A")

        total = sum(len(getattr(r, a)) for a in (
            "hbonds","halogen","salt_bridges","arom_hbonds",
            "pipi","pi_cation","clash_good","clash_bad","clash_ugly",
            "water_bridges"))
        if total == 0:
            lines.append("  No interactions found.")
        return "\n".join(lines)


_stepper = LigandStepper()


# ---------------------------------------------------------------------------
# Pose property helpers
# ---------------------------------------------------------------------------

def _parse_sdf_records(path: str) -> list:
    """Parse an SDF file and return a list of property dicts (one per record).

    Reads ``> <name>\\nvalue`` SD data tags.  Tries to coerce values to int
    or float; leaves them as str otherwise.  Also stores the molecule title
    (first line of each record) under the key ``_name``.
    """
    import re
    records = []
    try:
        with open(path) as fh:
            content = fh.read()
        for i, block in enumerate(content.split("$$$$")):
            if not block.strip():
                continue
            if i:
                # Drop only the newline left by the separator: strip() would eat
                # an empty title line too, promoting the counts line to the title.
                if block.startswith("\r\n"):
                    block = block[2:]
                elif block.startswith("\n"):
                    block = block[1:]
            props = {}
            lines = block.splitlines()
            if lines and lines[0].strip():
                props["_name"] = lines[0].strip()
            # The header line may carry extra fields after the tag name —
            # "> <Ligand_ID>  (1) " as written by RDKit/GNINA — so skip to EOL
            # rather than requiring the newline right after '>'.
            for m in re.finditer(r">\s*<([^>]+)>[^\n]*\n([^\n]*)", block):
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
        print(f"PoseViewer: could not parse SDF '{path}': {e}")
    return records


def _get_pose_properties(lig_name: str, state: int) -> dict:
    """Return a dict of SD/object properties via PyMOL's incentive-only API.

    Works only in Incentive PyMOL (get_property_list / get_property).
    Open-source PyMOL returns {} silently.
    """
    props = {}
    try:
        names = cmd.get_property_list(lig_name, state)
        if names:
            for n in names:
                v = cmd.get_property(n, lig_name, state)
                if v is not None:
                    props[n] = v
    except Exception:
        pass
    return props


def _pose_title(lig_name: str, state: int) -> str:
    """Per-state molecule title of a loaded object, or "" if none.

    PyMOL loads a multi-record SDF as one object named after the file, but it
    keeps each record's title line (MOL block line 1) as that state's title.
    cmd.get_title works in open-source PyMOL, so this is the per-pose name even
    when no scores SDF has been loaded.  Docking output sometimes repeats one
    title across every pose of a compound, or leaves it blank — hence "" when
    the title is empty or just echoes the object name.
    """
    try:
        title = (cmd.get_title(lig_name, state) or "").strip()
    except Exception:
        return ""
    return "" if title == lig_name else title


# ---------------------------------------------------------------------------
# Pose metrics
# ---------------------------------------------------------------------------
# Everything below computes pose quality/similarity metrics from what is loaded
# in the PyMOL session, so they are available even when the docking program did
# not write them into the SDF.  Field names match those written by the GNINA
# webapp (MCS_RMSD, Shape_Sim, Ref_Sim, PLIF_Sim, PB_Flags) so computed and
# SDF-supplied values are interchangeable in the Pose Data table.
#
# MCS_RMSD / Shape_Sim / Ref_Sim need only RDKit and run in a background thread.
# PLIF_Sim / PB_Flags need prolif / posebusters, which PyMOL's own interpreter
# rarely has, so they run in a subprocess under an interpreter that does (see
# _external_python_candidates).

METRIC_FIELDS = {
    "mcs_rmsd":    "MCS_RMSD",
    "shape_sim":   "Shape_Sim",
    "ref_sim":     "Ref_Sim",
    "plif_sim":    "PLIF_Sim",
    "posebusters": "PB_Flags",
}

METRIC_LABELS = {
    "mcs_rmsd":    "MCS RMSD vs reference (Å)",
    "shape_sim":   "Shape similarity vs reference (3D)",
    "ref_sim":     "2D similarity vs reference (ECFP4)",
    "plif_sim":    "PLIF similarity vs reference",
    "posebusters": "PoseBusters flags (failed checks)",
}

METRIC_ORDER = ("mcs_rmsd", "shape_sim", "ref_sim", "plif_sim", "posebusters")

# Metrics compared against the reference ligand — dropped when the reference changes.
REF_METRICS = ("mcs_rmsd", "shape_sim", "ref_sim", "plif_sim")

# Metrics that need the receptor.
RECEPTOR_METRICS = ("plif_sim", "posebusters")

# Metrics that run in a subprocess, and the modules that subprocess must import.
EXTERNAL_METRICS = ("plif_sim", "posebusters")
_EXTERNAL_REQUIRES = {
    "plif_sim":    ("prolif", "MDAnalysis"),
    "posebusters": ("posebusters",),
}


# --- RDKit (imported on demand: ~1 s, not worth paying at plugin load) -------

_RDKIT: dict = {}


def _load_rdkit() -> bool:
    """Import RDKit and cache the pieces the metrics need. False if unavailable."""
    if _RDKIT:
        return bool(_RDKIT.get("ok"))
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFMCS, rdShapeHelpers, rdFingerprintGenerator
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog("rdApp.*")
        _RDKIT.update(ok=True, Chem=Chem, DataStructs=DataStructs, rdFMCS=rdFMCS,
                      rdShapeHelpers=rdShapeHelpers,
                      rdFingerprintGenerator=rdFingerprintGenerator,
                      rdMolStandardize=rdMolStandardize)
    except Exception as e:
        _RDKIT.update(ok=False, error=f"{type(e).__name__}: {e}")
    return bool(_RDKIT.get("ok"))


def _sanitize_keep_hs(mol):
    """Sanitize a mol parsed with sanitize=False, in place. Returns mol.

    Full SanitizeMol can fail on docking output (unusual atom types, charges);
    fall back to FastFindRings so ring info is populated and fingerprints /
    MCS still work.

    PyMOL exports delocalized groups outside rings — a carboxylate written as
    C(:O):[O-] — using MDL bond type 4 ("aromatic").  RDKit's aromaticity model
    only processes ring systems, so those bonds survive sanitization as bare
    BondType.AROMATIC and hash differently from the same group's Kekule form.
    Resolve each such set into an explicit single/double pair (double bond to
    the least-negatively-charged neighbour) and re-sanitize.
    """
    Chem = _RDKIT["Chem"]
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.FastFindRings(mol)

    stray = [b for b in mol.GetBonds()
             if b.GetBondType() == Chem.BondType.AROMATIC and not b.IsInRing()]
    if stray:
        stray_idx = {b.GetIdx() for b in stray}
        resolved = set()
        # An atom may be the DOUBLE end of at most one stray bond; tracking that
        # across atoms stops a 3-atom conjugated chain from getting a cumulated
        # double bond that SanitizeMol would then reject.
        has_double = set()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            bonds = [b for b in atom.GetBonds()
                     if b.GetIdx() in stray_idx and b.GetIdx() not in resolved]
            if not bonds:
                continue
            chosen_double = None
            if idx not in has_double:
                eligible = [b for b in bonds
                            if b.GetOtherAtom(atom).GetIdx() not in has_double]
                pool = eligible or bonds
                pool.sort(key=lambda b: b.GetOtherAtom(atom).GetFormalCharge(),
                          reverse=True)
                chosen_double = pool[0]
            for bond in bonds:
                bond.SetIsAromatic(False)
                if bond is chosen_double:
                    bond.SetBondType(Chem.BondType.DOUBLE)
                    has_double.add(idx)
                    has_double.add(bond.GetOtherAtom(atom).GetIdx())
                else:
                    bond.SetBondType(Chem.BondType.SINGLE)
                resolved.add(bond.GetIdx())
        for atom in mol.GetAtoms():
            atom.SetIsAromatic(False)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            Chem.FastFindRings(mol)
    return mol


def _sanitize_heavy(mol):
    """_sanitize_keep_hs plus explicit-H removal, for heavy-atom-only comparisons.

    Poses often carry explicit Hs while the reference does not; that asymmetry
    corrupts every heavy-atom comparison (Morgan FP, shape overlap, MCS), so
    strip them here rather than relying on the parser's removeHs (which is
    silently skipped when sanitize=False).
    """
    Chem = _RDKIT["Chem"]
    mol = _sanitize_keep_hs(mol)
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
        # RemoveHs(sanitize=False) drops ring info, same as a fresh unsanitized
        # parse would — repopulate it so shape/MCS/FP calls don't hit
        # "RingInfo not initialized".
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            Chem.FastFindRings(mol)
    except Exception:
        pass
    return mol


# Uncharger / TautomerEnumerator / fingerprint generator construction has
# non-trivial setup cost, so instances are reused — but kept thread-local, since
# RDKit's MolStandardize classes are not thread-safe and a metrics job runs off
# the main thread.
_RD_CACHE = threading.local()


def _standardize_2d(mol):
    """Neutralize charges and collapse to a canonical tautomer for Ref_Sim, so a
    different protomer/tautomer of the same compound scores 1.0 rather than being
    penalized as a distinct structure.  Order matters: uncharge first, then
    canonicalize tautomers on the neutral form."""
    rdMolStandardize = _RDKIT["rdMolStandardize"]
    uncharger = getattr(_RD_CACHE, "uncharger", None)
    if uncharger is None:
        uncharger = _RD_CACHE.uncharger = rdMolStandardize.Uncharger()
    tautomer = getattr(_RD_CACHE, "tautomer", None)
    if tautomer is None:
        tautomer = _RD_CACHE.tautomer = rdMolStandardize.TautomerEnumerator()
    try:
        mol = uncharger.uncharge(mol)
    except Exception:
        pass
    try:
        mol = tautomer.Canonicalize(mol)
    except Exception:
        pass
    return mol


def _mol_from_block(block: str, keep_hs: bool = False):
    """Parse a molblock into a sanitized RDKit mol, or None."""
    Chem = _RDKIT["Chem"]
    try:
        mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return _sanitize_keep_hs(mol) if keep_hs else _sanitize_heavy(mol)
    except Exception:
        return None


def _conf_coords(mol):
    """Return [(x, y, z), ...] for mol's conformer, or None if it has none."""
    if not mol.GetNumConformers():
        return None
    conf = mol.GetConformer()
    return [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]


# --- In-process metrics -----------------------------------------------------

def _calc_mcs_rmsd(ref_mol, pose_mol):
    """Symmetry-aware heavy-atom RMSD over the maximum common substructure.

    No superposition: poses and reference already share the receptor frame, so
    aligning first would report conformer difference rather than pose deviation.
    Returns a float, or None when no usable MCS was found.
    """
    rdFMCS, Chem = _RDKIT["rdFMCS"], _RDKIT["Chem"]
    res = rdFMCS.FindMCS(
        [ref_mol, pose_mol],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        timeout=2,
    )
    if res.numAtoms < 3:
        return None
    mcs_mol = Chem.MolFromSmarts(res.smartsString)
    if mcs_mol is None:
        return None
    ref_matches = ref_mol.GetSubstructMatches(mcs_mol, uniquify=False, maxMatches=16)
    pose_matches = pose_mol.GetSubstructMatches(mcs_mol, uniquify=False, maxMatches=16)
    if not ref_matches or not pose_matches:
        return None
    ref_xyz = _conf_coords(ref_mol)
    pose_xyz = _conf_coords(pose_mol)
    if ref_xyz is None or pose_xyz is None:
        return None
    best = None
    for ref_match in ref_matches:
        rc = [ref_xyz[i] for i in ref_match]
        for pose_match in pose_matches:
            total = 0.0
            for (ax, ay, az), pi in zip(rc, pose_match):
                bx, by, bz = pose_xyz[pi]
                total += (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
            rmsd = math.sqrt(total / len(ref_match))
            if best is None or rmsd < best:
                best = rmsd
    return best


def _calc_shape_sim(ref_mol, pose_mol):
    """Shape Tanimoto (1 = identical) between pose and reference as placed.

    No alignment: docking already puts both in the binding site, so this measures
    overlap with the reference rather than best-case shape agreement.
    """
    if not ref_mol.GetNumConformers() or not pose_mol.GetNumConformers():
        return None
    dist = _RDKIT["rdShapeHelpers"].ShapeTanimotoDist(ref_mol, pose_mol)
    return 1.0 - dist


def _morgan_generator():
    """ECFP4 generator (radius 2, 2048 bits), reused per thread."""
    gen = getattr(_RD_CACHE, "morgan", None)
    if gen is None:
        gen = _RD_CACHE.morgan = _RDKIT["rdFingerprintGenerator"].GetMorganGenerator(
            radius=2, fpSize=2048)
    return gen


def _calc_ref_sim(ref_fp, pose_mol):
    """Morgan ECFP4 Tanimoto to the reference, on standardized 2D structures."""
    fp = _morgan_generator().GetFingerprint(_standardize_2d(pose_mol))
    return _RDKIT["DataStructs"].TanimotoSimilarity(ref_fp, fp)


# --- External worker (prolif / posebusters) ---------------------------------

_WORKER_SRC = r'''#!/usr/bin/env python3
"""PoseViewer external metric worker.

Runs under an interpreter that has prolif / posebusters installed.  Reads a JSON
job file, writes JSON results, and prints "PROGRESS <done> <total>" lines so the
caller can drive a progress bar.  Input SDFs are written by PoseViewer with
already-sanitized molecules, so no bond/valence fixing is needed here.
"""
import json
import os
import sys
import tempfile
import warnings


def _first_mol(path):
    from rdkit import Chem
    for mol in Chem.SDMolSupplier(path, removeHs=False, sanitize=True):
        if mol is not None and mol.GetNumAtoms() > 0:
            return mol
    return None


def _split_blocks(path):
    """Split an SDF into molblocks.

    Drops only the single newline that follows the "$$$$" separator: a molecule
    whose title line is empty would otherwise lose it to strip(), shifting the
    counts line up and making the block unparseable.
    """
    with open(path) as fh:
        text = fh.read()
    blocks = []
    for i, raw in enumerate(text.split("$$$$")):
        if not raw.strip():
            continue
        if i:  # not the first record: drop the newline left by the separator
            if raw.startswith("\r\n"):
                raw = raw[2:]
            elif raw.startswith("\n"):
                raw = raw[1:]
        blocks.append(raw)
    return blocks


def run_plif(job):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import MDAnalysis as mda
    import prolif
    from rdkit import Chem, DataStructs

    n = job["n_poses"]
    print("PROGRESS 0 %d" % n, flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(job["receptor"])
        prot_ag = u.select_atoms("protein")
        if prot_ag.n_atoms == 0:
            raise ValueError("no protein atoms in receptor")
        try:
            protein = prolif.Molecule.from_mda(prot_ag)
        except Exception:
            # Some PDB connectivity trips RDKit's strict valence check when
            # NoImplicit=True; retry with implicit Hs allowed.
            protein = prolif.Molecule.from_mda(prot_ag, NoImplicit=False)

    ref = _first_mol(job["reference"])
    if ref is None:
        raise ValueError("could not read reference ligand")
    ref_lig = prolif.Molecule.from_rdkit(ref)

    ligs, valid = [], []
    supplier = Chem.SDMolSupplier(job["poses"], removeHs=False, sanitize=True)
    for i, mol in enumerate(supplier):
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        try:
            ligs.append(prolif.Molecule.from_rdkit(mol))
            valid.append(i)
        except Exception:
            pass

    values = [None] * n
    if ligs:
        # One Fingerprint instance for reference + all poses, so every bitvector
        # shares the same residue-interaction columns and they stay comparable.
        fp = prolif.Fingerprint()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Explicit over relying on n_jobs=None's default: that only
                # started meaning "use every core" as of prolif 2.1.0 (older
                # versions ran serial unless told otherwise) -- see PoseBusters'
                # analogous max_workers=None above.
                fp.run_from_iterable([ref_lig] + ligs, protein, progress=False,
                                      n_jobs=os.cpu_count() or None)
            except TypeError:
                fp.run_from_iterable([ref_lig] + ligs, protein, progress=False)
        bvs = fp.to_bitvectors()
        for i, bv in zip(valid, bvs[1:]):
            values[i] = round(DataStructs.TanimotoSimilarity(bvs[0], bv), 4)
    print("PROGRESS %d %d" % (n, n), flush=True)
    return values, ""


def run_posebusters(job):
    from posebusters import PoseBusters

    n = job["n_poses"]
    blocks = _split_blocks(job["poses"])
    try:
        pb = PoseBusters(config="dock", max_workers=None)
    except TypeError:
        pb = PoseBusters(config="dock")

    infra_cols = {"mol_cond_loaded", "mol_true_loaded"}
    values = [None] * n
    fail_totals = {}

    def bust(indices):
        fd, tmp = tempfile.mkstemp(suffix=".sdf", dir=job["tmpdir"])
        try:
            with os.fdopen(fd, "w") as fh:
                for i in indices:
                    fh.write(blocks[i].rstrip("\n") + "\n$$$$\n")
            return pb.bust(tmp, mol_cond=job["receptor"], full_report=False)
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def process(indices):
        df = bust(indices)
        if df is None:
            # Isolate the offending pose by bisection so one bad molecule does
            # not invalidate the whole batch.
            if len(indices) == 1:
                return
            mid = len(indices) // 2
            process(indices[:mid])
            process(indices[mid:])
            return
        cols = [c for c in df.columns if df[c].dtype == bool and c not in infra_cols]
        for pos, (_, row) in zip(indices, df.iterrows()):
            vals = row[cols].dropna()
            values[pos] = int((vals == False).sum())  # noqa: E712
        for col, k in (df[cols] == False).sum().items():  # noqa: E712
            fail_totals[col] = fail_totals.get(col, 0) + int(k)

    # PoseBusters' own _run_parallel_over_poses opens a fresh ProcessPoolExecutor
    # per bust() call and tears it down when it returns -- it does NOT reuse a
    # pool across calls. Batching in chunks of 20 (as before) paid that pool
    # startup/shutdown cost once per chunk instead of once per run, which for
    # hundreds of poses dwarfed the actual busting time and looked like the job
    # kept restarting every few seconds. PoseBusters already parallelizes a
    # single bust() call internally (config chunk_size=100, across max_workers
    # processes), so batch at that same size here -- far fewer pool spin-ups
    # than chunks of 20, while still giving the progress bar periodic updates
    # on a large run. process()'s bisection still isolates a bad molecule
    # on failure within a batch.
    batch = 100
    done = 0
    for start in range(0, n, batch):
        idx = list(range(start, min(start + batch, n)))
        process(idx)
        done += len(idx)
        print("PROGRESS %d %d" % (done, n), flush=True)

    failing = [(k, v) for k, v in fail_totals.items() if v]
    failing.sort(key=lambda kv: -kv[1])
    summary = "; ".join("%s: %d" % (k, v) for k, v in failing[:6])
    return values, summary


def main():
    job = json.load(open(sys.argv[1]))
    try:
        if job["metric"] == "plif_sim":
            values, summary = run_plif(job)
        elif job["metric"] == "posebusters":
            values, summary = run_posebusters(job)
        else:
            raise ValueError("unknown metric %r" % job["metric"])
        out = {"values": values, "summary": summary}
    except Exception as e:
        out = {"error": "%s: %s" % (type(e).__name__, e)}
    with open(job["out"], "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
'''

_EXTERNAL_PYTHON: dict = {}


def _candidate_pythons():
    """Interpreters to consider for the external metrics, best guess first."""
    cands = []
    env = os.environ.get("POSEVIEWER_PYTHON")
    if env:
        cands.append(env)
    cands.append(sys.executable)
    roots = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        roots.append(os.path.dirname(conda_prefix))
    home = os.path.expanduser("~")
    roots += [os.path.join(home, "miniconda3", "envs"),
              os.path.join(home, "anaconda3", "envs"),
              os.path.join(home, "Programs", "miniconda3", "envs"),
              os.path.join(home, "Programs", "anaconda3", "envs"),
              "/opt/conda/envs"]
    import glob as _glob
    for root in roots:
        if root and os.path.isdir(root):
            if os.name == "nt":
                # Windows conda envs put the interpreter directly in the env root.
                cands += sorted(_glob.glob(os.path.join(root, "*", "python.exe")))
            else:
                cands += sorted(_glob.glob(os.path.join(root, "*", "bin", "python")))
    seen, out = set(), []
    for c in cands:
        if c and c not in seen and os.path.exists(c):
            seen.add(c)
            out.append(c)
    return out


_HAS_MODULES_CACHE: dict = {}


def _python_has_modules(python_exe: str, modules) -> bool:
    """True when every module looks installed for python_exe.

    Checked by looking for the package in that interpreter's site-packages
    rather than by importing it — importing prolif/MDAnalysis costs seconds and
    this runs over every conda env on the machine.  A package that is present
    but broken is caught later, when the worker actually runs.
    """
    key = (python_exe, tuple(modules))
    if key in _HAS_MODULES_CACHE:
        return _HAS_MODULES_CACHE[key]
    import glob as _glob
    if os.name == "nt":
        # Windows conda env: interpreter sits directly in the env root, and
        # site-packages is "<env>\Lib\site-packages" (unversioned, one level up).
        prefix = os.path.dirname(python_exe)
        site_dirs = _glob.glob(os.path.join(prefix, "Lib", "site-packages"))
        site_dirs += _glob.glob(os.path.join(prefix, "lib", "site-packages"))
    else:
        prefix = os.path.dirname(os.path.dirname(python_exe))
        site_dirs = _glob.glob(os.path.join(prefix, "lib", "python*", "site-packages"))
        site_dirs += _glob.glob(os.path.join(prefix, "lib", "site-packages"))
    ok = bool(site_dirs)
    for mod in modules if ok else ():
        if not any(os.path.isdir(os.path.join(sd, mod)) or
                   os.path.exists(os.path.join(sd, mod + ".py"))
                   for sd in site_dirs):
            ok = False
            break
    _HAS_MODULES_CACHE[key] = ok
    return ok


def _external_python_candidates(metric: str):
    """Interpreters that look able to run `metric`, best first.

    Environments that also cover the other external metric are ranked first, so
    both end up in the same interpreter when one environment has everything.
    Once a candidate has actually produced results it is cached and used alone.
    """
    if _EXTERNAL_PYTHON.get(metric):
        return [_EXTERNAL_PYTHON[metric]]
    needed = _EXTERNAL_REQUIRES[metric]
    every = tuple(sorted({m for mods in _EXTERNAL_REQUIRES.values() for m in mods}))
    found = [c for c in _candidate_pythons() if _python_has_modules(c, needed)]
    found.sort(key=lambda c: not _python_has_modules(c, every))
    return found


def _looks_like_missing_module(error: str) -> bool:
    """True when an error suggests trying the next candidate interpreter."""
    return any(s in error for s in ("ModuleNotFoundError", "ImportError",
                                    "No module named", "worker produced no result"))


def _kill_child(proc, hard=False):
    """Terminate proc and anything it spawned (PoseBusters starts its own pool).

    proc.terminate()/.kill() only signal the one process we hold a handle to --
    on POSIX the pool workers are reaped too because start_new_session puts the
    whole tree in one process group that os.killpg can target, but Windows has
    no process-group equivalent here, so terminate()/kill() alone would leave
    PoseBusters' ProcessPoolExecutor workers running as orphans, each still
    holding a CPU core. taskkill /T walks the process tree by parent PID and
    kills it in one shot -- the actual fix for that on Windows.
    """
    try:
        if os.name == "posix" and hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid),
                      signal.SIGKILL if hard else signal.SIGTERM)
            return
    except Exception:
        pass
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=10)
            return
        except Exception:
            pass
    try:
        proc.kill() if hard else proc.terminate()
    except Exception:
        pass


def _stream_process(proc, cancel, on_line):
    """Feed proc's stdout lines to on_line until it exits; kill it if cancelled.

    A reader thread per pipe rather than select(), which on Windows works only
    on sockets — the previous version failed there before reading a byte and the
    caller could only report the generic "worker produced no result".  Polling
    the exit status means a cancel is noticed within ~0.1 s even when the child
    stays silent for minutes (PLIF emits nothing until it is done).

    Returns collected stderr text.
    """
    err_chunks: List[str] = []

    def reader(stream, sink):
        try:
            for raw in iter(stream.readline, b""):
                sink(raw.decode("utf-8", "replace").rstrip())
        except Exception:
            pass

    threads = [threading.Thread(target=reader, args=(proc.stdout, on_line),
                                daemon=True),
               threading.Thread(target=reader, args=(proc.stderr, err_chunks.append),
                                daemon=True)]
    for t in threads:
        t.start()

    while proc.poll() is None:
        if cancel is not None and cancel.is_set():
            _kill_child(proc)
            break
        time.sleep(0.1)

    try:
        proc.wait(timeout=10)
    except Exception:
        _kill_child(proc, hard=True)
    for t in threads:
        t.join(timeout=2)
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except Exception:
            pass
    return "\n".join(err_chunks)


def _run_external_metric(metric, python_exe, workdir, poses_sdf, ref_sdf,
                         receptor_path, n_poses, report, cancel):
    """Run one external metric in a subprocess. Returns (values, summary, error)."""
    worker_py = os.path.join(workdir, "pv_metric_worker.py")
    if not os.path.exists(worker_py):
        with open(worker_py, "w") as fh:
            fh.write(_WORKER_SRC)
    job_path = os.path.join(workdir, f"{metric}_job.json")
    out_path = os.path.join(workdir, f"{metric}_out.json")
    with open(job_path, "w") as fh:
        json.dump({"metric": metric, "poses": poses_sdf, "reference": ref_sdf,
                   "receptor": receptor_path, "out": out_path,
                   "n_poses": n_poses, "tmpdir": workdir}, fh)

    try:
        proc = subprocess.Popen(
            [python_exe, worker_py, job_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            # POSIX only: gives the worker its own process group so cancelling
            # takes PoseBusters' worker pool down with it.
            start_new_session=(os.name == "posix"))
    except Exception as e:
        return None, "", f"could not start {python_exe}: {e}"

    def on_line(line):
        if line.startswith("PROGRESS "):
            parts = line.split()
            try:
                report(int(parts[1]))
            except (IndexError, ValueError):
                pass

    stderr_text = _stream_process(proc, cancel, on_line)
    if cancel is not None and cancel.is_set():
        return None, "", "cancelled"
    if not os.path.exists(out_path):
        tail = stderr_text.strip().splitlines()[-3:]
        return None, "", "worker produced no result" + (
            ": " + " / ".join(tail) if tail else "")
    with open(out_path) as fh:
        result = json.load(fh)
    if "error" in result:
        return None, "", result["error"]
    return result.get("values") or [], result.get("summary", ""), None


# --- Job driver -------------------------------------------------------------

class _MetricsJob:
    """Computes selected metrics for a set of poses off the PyMOL main thread.

    All PyMOL API access happens in _collect_metric_inputs before the thread
    starts; the job itself only ever sees molblock strings and file paths.
    """

    def __init__(self, metrics, pose_keys, pose_blocks, ref_block, receptor_path,
                 workdir, precomputed=None):
        self.metrics = [m for m in METRIC_ORDER if m in metrics]
        self.pose_keys = pose_keys
        self.pose_blocks = pose_blocks
        self.ref_block = ref_block
        self.receptor_path = receptor_path
        self.workdir = workdir
        # Snapshot of _stepper.computed at collection time: {pose_key: {field: val}}.
        # A field already present here (yes, even "N/A") is skipped — re-running
        # Calculate after just stepping to a new pose, or adding a metric, must
        # not re-bust/re-fingerprint poses a previous run already finished.
        self.precomputed = precomputed if precomputed is not None else {}
        self._pose_sdf_cache = None  # (todo, path, written) — shared across metrics
                                      # that happen to need the exact same poses.
        self.results: Dict[tuple, dict] = {}
        self.errors: List[str] = []
        self.notes: List[str] = []
        self.finished = False
        self.total_n = max(1, len(pose_blocks) * len(self.metrics))
        self._done = 0
        self._base = 0
        self._message = "Starting…"
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    # -- driving ------------------------------------------------------------

    def start(self):
        threading.Thread(target=self.run, daemon=True, name="PoseViewer-metrics").start()

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def progress(self):
        with self._lock:
            return self._done, self.total_n, self._message

    def _report(self, done_in_metric, message=None):
        with self._lock:
            self._done = self._base + done_in_metric
            if message is not None:
                self._message = message

    def _set_message(self, message):
        with self._lock:
            self._message = message

    # -- computation --------------------------------------------------------

    def run(self):
        try:
            for metric in self.metrics:
                if self.cancelled:
                    break
                field = METRIC_FIELDS[metric]
                self._set_message(f"{METRIC_LABELS[metric]}…")
                try:
                    out = self._run_metric(metric)
                except Exception as e:
                    self.errors.append(f"{field}: {type(e).__name__}: {e}")
                    out = None
                if out is not None:
                    keys_out, values = out
                    for key, val in zip(keys_out, values):
                        self.results.setdefault(key, {})[field] = (
                            "N/A" if val is None else val)
                self._base += len(self.pose_blocks)
                self._report(0)
            self._set_message("Cancelled." if self.cancelled else "Done.")
        finally:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.finished = True

    def _run_metric(self, metric):
        if metric in EXTERNAL_METRICS:
            return self._run_external(metric)
        return self._run_rdkit(metric)

    def _run_rdkit(self, metric):
        """MCS_RMSD / Shape_Sim / Ref_Sim — RDKit only, one pose at a time."""
        field = METRIC_FIELDS[metric]
        label = METRIC_LABELS[metric]
        todo = [i for i, key in enumerate(self.pose_keys)
                if field not in self.precomputed.get(key, {})]
        if not todo:
            self._report(len(self.pose_blocks), f"{label}: cached")
            return [], []

        ref_mol = _mol_from_block(self.ref_block)
        if ref_mol is None:
            self.errors.append(f"{field}: could not read reference ligand")
            return None
        ref_fp = None
        if metric == "ref_sim":
            ref_fp = _morgan_generator().GetFingerprint(_standardize_2d(ref_mol))
        elif not ref_mol.GetNumConformers():
            self.errors.append(f"{field}: reference has no 3D coordinates")
            return None

        keys_out, values = [], []
        for n_done, i in enumerate(todo):
            if self.cancelled:
                break
            val = None
            try:
                pose_mol = _mol_from_block(self.pose_blocks[i])
                if pose_mol is not None:
                    if metric == "mcs_rmsd":
                        val = _calc_mcs_rmsd(ref_mol, pose_mol)
                    elif metric == "shape_sim":
                        val = _calc_shape_sim(ref_mol, pose_mol)
                    else:
                        val = _calc_ref_sim(ref_fp, pose_mol)
            except Exception:
                val = None
            keys_out.append(self.pose_keys[i])
            values.append(None if val is None else round(float(val), 4))
            if n_done % 5 == 0 or n_done == len(todo) - 1:
                self._report(n_done + 1, f"{label}: {n_done + 1}/{len(todo)}")
        return keys_out, values

    def _run_external(self, metric):
        """PLIF_Sim / PB_Flags — subprocess under an interpreter that has the deps.

        Poses whose field is already in self.precomputed are skipped, so a
        second Calculate on an unchanged set — the common case of stepping to
        a new pose then re-clicking — costs nothing instead of re-busting or
        re-fingerprinting every pose again.
        """
        field = METRIC_FIELDS[metric]
        label = METRIC_LABELS[metric]
        todo = [i for i, key in enumerate(self.pose_keys)
                if field not in self.precomputed.get(key, {})]
        skipped = len(self.pose_blocks) - len(todo)
        if not todo:
            self._report(len(self.pose_blocks), f"{label}: cached")
            return [], []

        candidates = _external_python_candidates(metric)
        if not candidates:
            self.errors.append(
                f"{field}: no Python found with "
                f"{'/'.join(_EXTERNAL_REQUIRES[metric])} installed "
                f"(set $POSEVIEWER_PYTHON to one)")
            return None
        if not self.receptor_path:
            self.errors.append(f"{field}: no receptor available")
            return None

        # Write sanitized (H-bearing) SDFs once and reuse across external
        # metrics that happen to need the exact same poses (the common case:
        # both freshly requested on a set with nothing cached yet).
        cache = self._pose_sdf_cache
        if cache is not None and cache[0] == todo:
            poses_sdf, written = cache[1], cache[2]
        else:
            poses_sdf = os.path.join(self.workdir, f"poses_{field}.sdf")
            written = self._write_pose_sdf(poses_sdf, todo)
            self._pose_sdf_cache = (todo, poses_sdf, written)
        if not written:
            self.errors.append(f"{field}: no poses could be prepared for {field}")
            return None

        ref_sdf = os.path.join(self.workdir, "reference.sdf")
        if metric == "plif_sim" and not os.path.exists(ref_sdf):
            if not self._write_ref_sdf(ref_sdf):
                self.errors.append(f"{field}: could not prepare reference ligand")
                return None

        n = len(written)
        note = f" ({skipped} cached)" if skipped else ""
        report = (lambda done: self._report(
            min(done, len(self.pose_blocks)), f"{label}: {done}/{n}{note}"))

        # The filesystem check only says a package is present; if that
        # interpreter cannot actually import it, move on to the next candidate.
        values = summary = error = None
        for python_exe in candidates[:3]:
            values, summary, error = _run_external_metric(
                metric, python_exe, self.workdir, poses_sdf, ref_sdf,
                self.receptor_path, n, report, self._cancel)
            if error is None:
                _EXTERNAL_PYTHON[metric] = python_exe
                break
            if error == "cancelled" or not _looks_like_missing_module(error):
                break
        if error:
            if error != "cancelled":
                self.errors.append(f"{field}: {error}")
            return None
        if summary:
            self.notes.append(f"{field} most common failures — {summary}")

        # Map worker results (SDF record order) back onto todo order.
        out = [None] * len(todo)
        for rec_i, local_i in enumerate(written):
            if rec_i < len(values):
                out[local_i] = values[rec_i]
        self._report(len(self.pose_blocks), f"{label}: done{note}")
        return [self.pose_keys[i] for i in todo], out

    def _write_pose_sdf(self, path, indices=None):
        """Write sanitized poses (Hs added) at `indices` (default: all) to path.

        Returns positions within `indices` (0-based, write order) that
        succeeded — i.e. SDF record order maps back to a pose via
        indices[returned_position]. Bond-order/valence fixing all happens
        here so the worker needs none of its own.
        """
        Chem = _RDKIT["Chem"]
        idx_list = list(range(len(self.pose_blocks))) if indices is None else indices
        written = []
        with open(path, "w") as fh:
            for pos, i in enumerate(idx_list):
                if self.cancelled:
                    break
                mol = _mol_from_block(self.pose_blocks[i], keep_hs=True)
                if mol is None:
                    continue
                if not mol.GetPropsAsDict().get("_Name", "").strip():
                    mol.SetProp("_Name", f"pose_{i + 1}")
                try:
                    # ProLIF perceives H-bond donors from explicit Hs, and poses
                    # loaded from a PDB often have none.
                    mol = Chem.AddHs(mol, addCoords=True)
                    fh.write(Chem.MolToMolBlock(mol) + "\n$$$$\n")
                except Exception:
                    continue
                written.append(pos)
        return written

    def _write_ref_sdf(self, path):
        Chem = _RDKIT["Chem"]
        mol = _mol_from_block(self.ref_block, keep_hs=True)
        if mol is None:
            return False
        try:
            mol = Chem.AddHs(mol, addCoords=True)
            with open(path, "w") as fh:
                fh.write(Chem.MolToMolBlock(mol) + "\n$$$$\n")
        except Exception:
            return False
        return True


def _collect_metric_inputs(metrics):
    """Dump poses, reference and receptor from the live session.

    Runs on the PyMOL main thread — the returned job only holds strings and
    paths, so nothing downstream touches the PyMOL API.  Raises RuntimeError
    with a message suitable for display when a requested metric can't be run.
    """
    metrics = [m for m in METRIC_ORDER if m in metrics]
    if not metrics:
        raise RuntimeError("No metrics selected.")
    if not _load_rdkit():
        raise RuntimeError(f"RDKit not available in this PyMOL: {_RDKIT.get('error', '')}")
    if not _stepper.poses:
        raise RuntimeError("No poses loaded — click Setup first.")

    needs_ref = any(m in REF_METRICS for m in metrics)
    ref_block = ""
    if needs_ref:
        ref = _stepper.ref_ligand
        if not ref:
            raise RuntimeError("Select a reference ligand first.")
        try:
            ref_block = cmd.get_str("mol", f"({ref})", 1)
        except Exception as e:
            raise RuntimeError(f"Could not export reference '{ref}': {e}")
        if not ref_block:
            raise RuntimeError(f"Reference '{ref}' has no atoms.")

    workdir = tempfile.mkdtemp(prefix="poseviewer_metrics_")
    receptor_path = ""
    if any(m in RECEPTOR_METRICS for m in metrics):
        prot = _stepper.protein_sel or "polymer.protein"
        try:
            if cmd.count_atoms(f"({prot})", state=1) == 0:
                raise RuntimeError(f"Receptor selection '{prot}' has no atoms.")
            receptor_path = os.path.join(workdir, "receptor.pdb")
            cmd.save(receptor_path, f"({prot})", 1)
        except RuntimeError:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(workdir, ignore_errors=True)
            raise RuntimeError(f"Could not export receptor '{prot}': {e}")

    pose_keys, pose_blocks = [], []
    for obj, state in _stepper.poses:
        try:
            block = cmd.get_str("mol", f"({obj})", state)
        except Exception:
            block = ""
        pose_keys.append((obj, state))
        pose_blocks.append(block)

    # Snapshot already-computed fields so the job can skip poses that don't
    # need (re)computing — see _MetricsJob.precomputed.
    precomputed = {key: dict(_stepper.computed.get(key, {})) for key in pose_keys}

    return _MetricsJob(metrics, pose_keys, pose_blocks, ref_block, receptor_path,
                       workdir, precomputed)


# ---------------------------------------------------------------------------
# Keybindings
# ---------------------------------------------------------------------------

def _bind_keys():
    cmd.set_key("right", ci_next)
    cmd.set_key("left", ci_prev)
    print("PoseViewer: LEFT/RIGHT arrow keys bound.")

def _unbind_keys():
    try:
        cmd.set_key("right", lambda: None)
        cmd.set_key("left", lambda: None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auto-split helper
# ---------------------------------------------------------------------------

def _auto_split_ligands(ligands_sel):
    """Extract each unique (chain, resn, resi) in ligands_sel into its own object.

    Returns list of created object names, or [] when only one residue is found.
    Cleans up any objects created by a previous auto-split call first.
    """
    _cleanup_autosplit()
    try:
        space = {"residues": []}
        cmd.iterate(ligands_sel,
                    "residues.append((chain, resn, resi))",
                    space=space)
        unique = list(dict.fromkeys(space["residues"]))
    except Exception as e:
        print(f"PoseViewer: auto-split failed: {e}")
        return []

    if len(unique) == 0:
        return []

    existing = set(cmd.get_names("objects"))
    created = []
    info = []
    next_idx = [1]   # shared counter so each object gets the next free obj## slot
    for chain, resn, resi in unique:
        # Mirror PyMOL's own obj01/obj02/... naming for manually extracted objects.
        # Scan upward so collisions with existing objects always yield obj## (never obj01_1).
        while (name := f"{_AUTOSPLIT_PREFIX}{next_idx[0]:02d}") in existing:
            next_idx[0] += 1
        next_idx[0] += 1
        chain_part = f"chain {chain} and " if chain.strip() else ""
        sel = f"({ligands_sel}) and {chain_part}resn {resn} and resi {resi}"
        try:
            cmd.create(name, sel)
            _track(name)
            existing.add(name)
            created.append(name)
            info.append(f"{name}({resn}/{chain}/{resi})")
        except Exception as e:
            print(f"PoseViewer: could not extract '{name}': {e}")

    if created:
        print(f"PoseViewer: auto-split {len(created)} ligand(s): {', '.join(info)}")
        # Suppress the source atoms so the split copies are the sole visible representation.
        # Without this the original object still shows all ligand atoms including all H.
        try:
            cmd.hide("everything", ligands_sel)
        except Exception:
            pass
    return created


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def ci_setup(protein="polymer.protein", ligands="organic", mode="auto"):
    """
DESCRIPTION
    Setup PoseViewer.

USAGE
    ci_setup [protein [, ligands [, mode]]]

EXAMPLES
    ci_setup
    ci_setup protein=chain A, ligands=LIG1,LIG2,LIG3
    ci_setup protein=polymer.protein, ligands=poses, mode=states
    """
    if mode == "auto":
        names = cmd.get_names("objects")
        if ligands in names and cmd.count_states(ligands) > 1:
            mode = "states"
        else:
            mode = "objects"
            multi_state = []
            for n in names:
                if n in _created_objects:
                    continue
                try:
                    if (cmd.count_atoms(f"{n} and ({ligands})", state=1) > 0 and
                            cmd.count_atoms(f"{n} and polymer.protein", state=1) == 0 and
                            cmd.count_states(n) > 1):
                        multi_state.append(n)
                except CmdException:
                    pass
            if len(multi_state) == 1:
                # exactly one multi-state ligand — use states mode for ref-ligand support
                mode = "states"
                ligands = multi_state[0]
            # else: multiple multi-state ligands or none → stay in objects mode

    if mode == "states":
        _stepper.setup_states(protein, ligands)
        print(f"PoseViewer: {cmd.count_states(ligands)} states.")
    else:
        _cleanup_autosplit()
        if "," in ligands:
            ligs = [l.strip() for l in ligands.split(",")]
            ref_lig = None
        else:
            all_n = cmd.get_names("objects")
            multi_ligs, single_ligs = [], []
            for n in all_n:
                if n in _created_objects:
                    continue
                try:
                    if (cmd.count_atoms(f"{n} and ({ligands})", state=1) > 0 and
                            cmd.count_atoms(f"{n} and polymer.protein", state=1) == 0):
                        (multi_ligs if cmd.count_states(n) > 1 else single_ligs).append(n)
                except CmdException:
                    pass
            # Multi-state objects are docking poses; single-state are ref-lig candidates
            if multi_ligs:
                ligs = multi_ligs
                ref_lig = single_ligs[0] if single_ligs else None
            else:
                # All single-state: if a scores SDF is loaded, use its molecule
                # names to separate pose ligands (appear in SDF) from ref candidates
                # (don't appear in SDF). Without SDF, treat everything as poses.
                if _stepper.sdf_records:
                    import re as _re
                    # Normalise SDF titles the same way PyMOL does when it creates
                    # object names: replace non-word characters with underscores.
                    sdf_names = {_re.sub(r'[^\w]', '_', r.get("_name", ""))
                                 for r in _stepper.sdf_records if r.get("_name")}
                    matched   = [n for n in single_ligs if n in sdf_names]
                    unmatched = [n for n in single_ligs if n not in sdf_names]
                    if matched:
                        ligs    = matched
                        ref_lig = unmatched[0] if len(unmatched) == 1 else None
                    else:
                        ligs    = single_ligs
                        ref_lig = None
                else:
                    ligs    = single_ligs
                    ref_lig = None
            if not ligs:
                # If protein is a named object, scope the split to that object
                # only — otherwise "organic" spans all loaded structures.
                # Use "model NAME" so PyMOL parses the object name unambiguously
                # (bare names starting with a digit can be mis-parsed as arithmetic).
                prot_is_obj = protein in set(cmd.get_names("objects"))
                lig_scope = f"({ligands}) and model {protein}" if prot_is_obj else ligands
                auto = _auto_split_ligands(lig_scope)
                if auto:
                    ligs = auto
                    ref_lig = None
                elif prot_is_obj:
                    print(f"PoseViewer: no organic ligands found in '{protein}'.")
                    ligs = []
                    ref_lig = None
                else:
                    ligs = [ligands]
                    ref_lig = None
        _stepper.setup_objects(protein, ligs, ref_lig=ref_lig)
        print(f"PoseViewer: {len(ligs)} ligand(s).")

    _warn_if_no_scores()
    _print_summary()
    _bind_keys()


def _warn_if_no_scores():
    """Say why the Pose Data table is bare, instead of leaving it a mystery.

    SD tags are read straight off a loaded SDF only by Incentive PyMOL; anywhere
    else the Scores field (or ci_load_scores) is what fills the table.
    """
    if _stepper.sdf_records or not _stepper.all_properties:
        return
    if any(set(props) - {"_name"} for props in _stepper.all_properties):
        return
    print("PoseViewer: no per-pose score columns found. SD data is read from a "
          "loaded SDF only by Incentive PyMOL — otherwise point the "
          "Scores (SDF) field at the poses file (or run ci_load_scores <path>) "
          "and press Setup again.")


def _print_summary():
    print(_stepper.summary())

def ci_next():
    _stepper.next(); _print_summary()

def ci_prev():
    _stepper.prev(); _print_summary()

def ci_goto(index=0):
    _stepper.goto(int(index)); _print_summary()

def ci_update():
    _stepper._show_current(); _print_summary()

def ci_refresh():
    """Re-detect interactions for the current PyMOL state (follows state slider)."""
    _stepper.refresh(); _print_summary()

def ci_load_scores(path=""):
    """Load per-pose SD properties from an SDF file.

USAGE
    ci_load_scores /path/to/poses.sdf
    """
    if not path:
        print("PoseViewer: ci_load_scores requires a file path.")
        return
    _stepper.sdf_records = _parse_sdf_records(path)
    print(f"PoseViewer: loaded {len(_stepper.sdf_records)} score records from '{path}'.")

def ci_calc(metrics="", quiet=0):
    """Compute pose metrics for the current setup, blocking until done.

USAGE
    ci_calc                          # MCS_RMSD, Shape_Sim, Ref_Sim
    ci_calc all
    ci_calc mcs_rmsd,plif_sim

    Names: mcs_rmsd, shape_sim, ref_sim, plif_sim, posebusters (or the field
    names MCS_RMSD, Shape_Sim, Ref_Sim, PLIF_Sim, PB_Flags).  Reference-based
    metrics need a reference ligand (ci_setup picks one up automatically, or
    set _stepper.ref_ligand / choose one in the GUI).
    """
    import time
    by_field = {v.lower(): k for k, v in METRIC_FIELDS.items()}
    text = str(metrics).replace(",", " ").split()
    if not text:
        wanted = ["mcs_rmsd", "shape_sim", "ref_sim"]
    elif len(text) == 1 and text[0].lower() == "all":
        wanted = list(METRIC_ORDER)
    else:
        wanted = []
        for name in text:
            key = name.lower()
            key = key if key in METRIC_FIELDS else by_field.get(key)
            if key is None:
                print(f"PoseViewer: unknown metric '{name}'.")
                return
            wanted.append(key)

    try:
        job = _collect_metric_inputs(wanted)
    except RuntimeError as e:
        print(f"PoseViewer: {e}")
        return

    job.start()
    last = ""
    while not job.finished:
        done, total, msg = job.progress()
        if not int(quiet) and msg != last:
            last = msg
            print(f"PoseViewer: {msg}")
        time.sleep(0.25)

    _stepper.merge_metrics(job.results)
    _stepper._table_dirty = True      # picked up by the GUI's sync timer
    for note in job.notes:
        print(f"PoseViewer: {note}")
    for err in job.errors:
        print(f"PoseViewer: {err}")
    for metric in job.metrics:
        field = METRIC_FIELDS[metric]
        n = sum(1 for v in job.results.values()
                if isinstance(v.get(field), (int, float)))
        print(f"PoseViewer: {field:9s} {n}/{len(job.pose_keys)} poses")


def ci_hbond_angle(degrees=""):
    """Show or set the minimum D-H...A angle for H-bonds.

USAGE
    ci_hbond_angle          # show the current value
    ci_hbond_angle 130      # only keep contacts at 130 deg or straighter
    ci_hbond_angle 0        # off: keep whatever PyMOL's polar contacts report

    PyMOL's own h_bond_max_angle is measured at the donor heavy atom, between
    the D-H bond and the D...A vector, so its 63 deg default admits contacts
    whose proton points ~100 deg away from the acceptor — a good N...O distance
    with the hydrogen aimed elsewhere.  This filter measures the angle at the
    proton instead.  Contacts with no explicit hydrogen are never filtered,
    since there is nothing to measure.
    """
    if degrees == "":
        print(f"PoseViewer: hbond_min_angle = {_stepper.hbond_min_angle:.0f} deg"
              f"{' (off)' if not _stepper.hbond_min_angle else ''}")
        return
    try:
        value = float(degrees)
    except (TypeError, ValueError):
        print(f"PoseViewer: '{degrees}' is not a number.")
        return
    if not 0.0 <= value <= 180.0:
        print("PoseViewer: angle must be between 0 and 180 degrees.")
        return
    _stepper.hbond_min_angle = value
    print(f"PoseViewer: hbond_min_angle = {value:.0f} deg"
          f"{' (off)' if not value else ''}")
    if _stepper.poses:
        ci_update()


def ci_clear():
    _clear_all(); _unbind_keys(); print("PoseViewer: cleared.")

def ci_bookmarks():
    """List all bookmarked poses to the console."""
    marked = [(i, key) for i, key in enumerate(_stepper.poses)
              if key in _stepper._bookmarks]
    if not marked:
        print("PoseViewer: no bookmarks.")
        return
    print(f"PoseViewer: {len(marked)} bookmark(s):")
    for i, (obj, st) in marked:
        label = f"{obj} state {st}" if cmd.count_states(obj) > 1 else obj
        print(f"  [{i + 1}] {label}")


def ci_export(path="", selection="all"):
    """Write the pose table — SDF properties plus anything ci_calc computed — to CSV.

USAGE
    ci_export /path/to/poses.csv
    ci_export /path/to/marked.csv, bookmarked

    A .tsv or .txt extension switches the delimiter to tab.
    """
    if not path:
        print("PoseViewer: ci_export requires a file path.")
        return
    if not _stepper.all_properties:
        print("PoseViewer: no pose data to export — run ci_setup first.")
        return
    only_marked = str(selection).strip().lower() in ("bookmarked", "bookmarks", "marked")
    cols = _stepper.table_columns()
    headers = ["Pose", "State", "Bookmarked"] + [
        "Ligand_ID" if c == "_name" else c for c in cols]
    rows = []
    for i, props in enumerate(_stepper.all_properties):
        key = _stepper.pose_key(i)
        if key is None:
            continue
        if only_marked and key not in _stepper._bookmarks:
            continue
        obj, st = key
        rows.append([obj, st, "yes" if key in _stepper._bookmarks else ""] +
                    [props.get(c, "") for c in cols])
    delim = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=delim)
            writer.writerow(headers)
            writer.writerows(rows)
    except OSError as e:
        print(f"PoseViewer: could not write '{path}': {e}")
        return
    print(f"PoseViewer: exported {len(rows)} pose(s) x {len(headers)} column(s) "
          f"to '{path}'.")


def _export_poses_sdf(path, only_marked=True):
    """Write poses (bookmarked-only by default) to path as a combined SDF.

    One record per pose, geometry straight from the live PyMOL state (no
    RDKit needed — same `cmd.get_str("mol", ...)` used to feed the external
    metrics). Each record's title line becomes its Ligand_ID (SDF-supplied
    name, per-state SDF title, or the PyMOL object name, in that order) so
    downstream tools can identify poses by name. Pose-table properties —
    scores loaded from SDF plus anything ci_calc computed — are carried over
    as SD tags. Returns the number of poses written.
    """
    cols = [c for c in _stepper.table_columns() if c != "_name"]
    n = 0
    with open(path, "w") as fh:
        for i, key in enumerate(_stepper.poses):
            if only_marked and key not in _stepper._bookmarks:
                continue
            obj, st = key
            try:
                block = cmd.get_str("mol", f"({obj})", st)
            except Exception:
                block = ""
            if not block:
                continue
            props = _stepper.all_properties[i] if i < len(_stepper.all_properties) else {}
            name = str(props.get("_name") or _pose_title(obj, st) or obj)
            name = name.replace("\n", " ").strip()
            nl = block.find("\n")
            if nl != -1:
                block = name + block[nl:]
            fh.write(block if block.endswith("\n") else block + "\n")
            tags = ([("Pose", obj), ("State", st),
                     ("Bookmarked", "yes" if key in _stepper._bookmarks else "")] +
                    [(c, props.get(c, "")) for c in cols])
            for tag, val in tags:
                if val in ("", None):
                    continue
                fh.write(f"> <{tag}>\n{val}\n\n")
            fh.write("$$$$\n")
            n += 1
    return n


def ci_export_sdf(path="", selection="bookmarked"):
    """Write poses to a combined SDF, one record per pose, titled by Ligand_ID.

USAGE
    ci_export_sdf /path/to/bookmarks.sdf
    ci_export_sdf /path/to/all_poses.sdf, all

    Geometry is read straight from the live PyMOL state. Pose-table
    properties (SDF-loaded scores, ci_calc results) are written as SD tags.
    Default selection is the bookmarked poses; pass "all" for every pose.
    """
    if not path:
        print("PoseViewer: ci_export_sdf requires a file path.")
        return
    if not _stepper.poses:
        print("PoseViewer: no poses loaded — run ci_setup first.")
        return
    only_marked = str(selection).strip().lower() not in ("all", "*")
    if only_marked and not _stepper._bookmarks:
        print("PoseViewer: no bookmarks to export "
              "(pass selection='all' to export every pose).")
        return
    try:
        n = _export_poses_sdf(path, only_marked)
    except OSError as e:
        print(f"PoseViewer: could not write '{path}': {e}")
        return
    print(f"PoseViewer: exported {n} pose(s) to '{path}'.")


def ci_gui():
    _open_gui()

cmd.extend("ci_setup", ci_setup)
cmd.extend("ci_next", ci_next)
cmd.extend("ci_prev", ci_prev)
cmd.extend("ci_goto", ci_goto)
cmd.extend("ci_update", ci_update)
cmd.extend("ci_refresh", ci_refresh)
cmd.extend("ci_load_scores", ci_load_scores)
cmd.extend("ci_calc", ci_calc)
cmd.extend("ci_clear", ci_clear)
cmd.extend("ci_bookmarks", ci_bookmarks)
cmd.extend("ci_export", ci_export)
cmd.extend("ci_export_sdf", ci_export_sdf)
cmd.extend("ci_hbond_angle", ci_hbond_angle)
cmd.extend("ci_gui", ci_gui)


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------

_gui_window = None

def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("PoseViewer", _open_gui)


def _open_gui():
    global _gui_window

    if _gui_window is not None:
        try:
            _gui_window.raise_(); _gui_window.activateWindow(); return
        except RuntimeError:
            _gui_window = None

    from pymol.Qt import QtWidgets, QtCore, QtGui

    win = QtWidgets.QWidget()
    _gui_window = win
    win.setWindowTitle("PoseViewer")
    win.setMinimumWidth(440)
    win.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    win.destroyed.connect(lambda: _set_gui_none())

    root = QtWidgets.QVBoxLayout(win)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(5)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    root.addWidget(splitter)

    top_w = QtWidgets.QWidget()
    top_l = QtWidgets.QVBoxLayout(top_w)
    top_l.setContentsMargins(0, 0, 0, 0)
    top_l.setSpacing(5)
    splitter.addWidget(top_w)

    bot_w = QtWidgets.QWidget()
    bot_l = QtWidgets.QVBoxLayout(bot_w)
    bot_l.setContentsMargins(0, 0, 0, 0)
    bot_l.setSpacing(5)
    splitter.addWidget(bot_w)

    # Setup
    g_s = QtWidgets.QGroupBox("Setup")
    l_s = QtWidgets.QGridLayout(g_s)
    l_s.addWidget(QtWidgets.QLabel("Protein:"), 0, 0)
    e_prot = QtWidgets.QComboBox(); e_prot.setEditable(True)
    e_prot.addItem("polymer.protein")
    l_s.addWidget(e_prot, 0, 1)
    l_s.addWidget(QtWidgets.QLabel("Ligand(s):"), 1, 0)
    e_lig = QtWidgets.QLineEdit("organic")
    l_s.addWidget(e_lig, 1, 1)

    l_s.addWidget(QtWidgets.QLabel("Scores (SDF):"), 2, 0)
    hl_sf = QtWidgets.QHBoxLayout()
    e_scores = QtWidgets.QLineEdit()
    e_scores.setPlaceholderText("optional")
    b_browse = QtWidgets.QPushButton("…"); b_browse.setFixedWidth(26)
    hl_sf.addWidget(e_scores); hl_sf.addWidget(b_browse)
    l_s.addLayout(hl_sf, 2, 1)

    hl_b = QtWidgets.QHBoxLayout()
    b_setup = QtWidgets.QPushButton("Setup")
    b_clear = QtWidgets.QPushButton("Clear")
    hl_b.addWidget(b_setup); hl_b.addWidget(b_clear)
    l_s.addLayout(hl_b, 3, 0, 1, 2)
    top_l.addWidget(g_s)

    # Navigate
    g_n = QtWidgets.QGroupBox("Navigate")
    l_n = QtWidgets.QVBoxLayout(g_n)
    lbl = QtWidgets.QLabel("Ready - click Setup")
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    f = lbl.font(); f.setBold(True); f.setPointSize(f.pointSize()+1)
    lbl.setFont(f)
    l_n.addWidget(lbl)

    hl_pn = QtWidgets.QHBoxLayout()
    b_prev = QtWidgets.QPushButton("◀  Prev  [←]")
    b_refresh = QtWidgets.QPushButton("Refresh")
    b_next = QtWidgets.QPushButton("[→]  Next  ▶")
    hl_pn.addWidget(b_prev); hl_pn.addWidget(b_refresh); hl_pn.addWidget(b_next)
    l_n.addLayout(hl_pn)

    hl_j = QtWidgets.QHBoxLayout()
    hl_j.addWidget(QtWidgets.QLabel("Go to #:"))
    sp = QtWidgets.QSpinBox(); sp.setMinimum(1); sp.setMaximum(99999)
    sp.setFixedWidth(70); hl_j.addWidget(sp)
    b_go = QtWidgets.QPushButton("Go"); b_go.setFixedWidth(50)
    hl_j.addWidget(b_go); hl_j.addStretch()
    l_n.addLayout(hl_j)
    cb_docking = QtWidgets.QCheckBox("Docking poses")
    cb_docking.setChecked(False)
    l_n.addWidget(cb_docking)
    cb_cmp_hb = QtWidgets.QCheckBox("H-bonds in compare mode")
    cb_cmp_hb.setChecked(False)
    cb_cmp_hb.setEnabled(False)   # enabled only when Docking poses is checked
    l_n.addWidget(cb_cmp_hb)
    hl_bm = QtWidgets.QHBoxLayout()
    b_bm = QtWidgets.QPushButton("☆ Bookmark")
    b_bm.setFixedWidth(120)
    b_bm_list = QtWidgets.QPushButton("List bookmarks")
    b_bm_export = QtWidgets.QPushButton("Export bookmarks…")
    b_bm_export.setToolTip("Write bookmarked poses to a combined SDF (title = Ligand_ID)")
    hl_bm.addWidget(b_bm); hl_bm.addWidget(b_bm_list); hl_bm.addWidget(b_bm_export)
    hl_bm.addStretch()
    l_n.addLayout(hl_bm)
    top_l.addWidget(g_n)

    # Reference ligand
    g_ref = QtWidgets.QGroupBox("Reference ligand")
    l_ref = QtWidgets.QHBoxLayout(g_ref)
    l_ref.addWidget(QtWidgets.QLabel("Object:"))
    ref_combo = QtWidgets.QComboBox(); ref_combo.addItem("(none)")
    ref_combo.setMinimumWidth(100)
    l_ref.addWidget(ref_combo, 1)
    cb_show_ref = QtWidgets.QCheckBox("Show ref"); cb_show_ref.setChecked(True)
    l_ref.addWidget(cb_show_ref)
    cb_show_pose = QtWidgets.QCheckBox("Show pose"); cb_show_pose.setChecked(True)
    l_ref.addWidget(cb_show_pose)
    top_l.addWidget(g_ref)

    # Pose Data
    g_pd = QtWidgets.QGroupBox("Pose Data")
    g_pd.setCheckable(True); g_pd.setChecked(True)
    l_pd = QtWidgets.QVBoxLayout(g_pd)

    class _SortItem(QtWidgets.QTableWidgetItem):
        def __lt__(self, other):
            a = self.data(QtCore.Qt.UserRole + 1)
            b = other.data(QtCore.Qt.UserRole + 1)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a < b
            return self.text() < other.text()

    tw_pd = QtWidgets.QTableWidget(0, 0)
    tw_pd.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    tw_pd.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    tw_pd.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    tw_pd.setMinimumHeight(80)
    tw_pd.horizontalHeader().setStretchLastSection(True)
    tw_pd.horizontalHeader().setSectionsMovable(True)
    tw_pd.verticalHeader().setDefaultSectionSize(18)
    tw_pd.setAlternatingRowColors(True)
    l_pd.addWidget(tw_pd)

    hl_pd = QtWidgets.QHBoxLayout()
    b_copy = QtWidgets.QPushButton("Copy all")
    b_copy.setToolTip("Copy the whole table as tab-separated text, ready to "
                      "paste into a spreadsheet.\nCtrl+C in the table copies "
                      "just the selected rows.")
    b_export = QtWidgets.QPushButton("Export…")
    b_export.setToolTip("Write the table to a CSV or TSV file")
    b_copy.setFixedWidth(90)
    b_export.setFixedWidth(80)
    lbl_pd = QtWidgets.QLabel("")
    lbl_pd.setStyleSheet("color: grey;")
    hl_pd.addWidget(b_copy); hl_pd.addWidget(b_export)
    hl_pd.addWidget(lbl_pd, 1)
    l_pd.addLayout(hl_pd)
    top_l.addWidget(g_pd, 1)

    # Swatch+checkbox helper
    def _cb(lay, text, hex_color, checked=True):
        row = QtWidgets.QHBoxLayout()
        sw = QtWidgets.QLabel(); sw.setFixedSize(14, 14)
        sw.setStyleSheet(f"background-color: {hex_color}; border: none;")
        row.addWidget(sw)
        cb = QtWidgets.QCheckBox(text); cb.setChecked(checked)
        row.addWidget(cb); row.addStretch()
        lay.addLayout(row)
        return cb

    def _section(title, enabled=True, expanded=True, show_enable=True):
        """Return (QGroupBox, enable_cb_or_None, body_widget, body_layout).

        enable_cb toggles the group's interactions on/off independently of the
        collapse arrow, which only shows/hides the body widget.
        show_enable=False: header shows a plain bold label (e.g. Display group).
        """
        g = QtWidgets.QGroupBox()
        gl = QtWidgets.QVBoxLayout(g)
        gl.setContentsMargins(6, 4, 6, 6)
        gl.setSpacing(2)
        hl = QtWidgets.QHBoxLayout()
        if show_enable:
            en = QtWidgets.QCheckBox(title)
            en.setChecked(enabled)
            _f = en.font(); _f.setBold(True); en.setFont(_f)
            hl.addWidget(en)
        else:
            lbl = QtWidgets.QLabel(title)
            _f = lbl.font(); _f.setBold(True); lbl.setFont(_f)
            hl.addWidget(lbl)
            en = None
        hl.addStretch()
        btn_col = QtWidgets.QToolButton()
        btn_col.setCheckable(True); btn_col.setChecked(expanded)
        btn_col.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        btn_col.setAutoRaise(True)
        hl.addWidget(btn_col)
        gl.addLayout(hl)
        body = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(body)
        bl.setContentsMargins(0, 2, 0, 0)
        bl.setSpacing(2)
        body.setVisible(expanded)
        gl.addWidget(body)
        def _toggle(checked):
            body.setVisible(checked)
            btn_col.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        btn_col.toggled.connect(_toggle)
        return g, en, body, bl

    # Calculate — metrics derived from the session instead of read from the SDF
    g_calc, _calc_en, _gcb, l_calc = _section("Calculate", expanded=False,
                                              show_enable=False)
    cb_metrics = {}
    for _key in METRIC_ORDER:
        _cb_m = QtWidgets.QCheckBox(METRIC_LABELS[_key])
        # External metrics (prolif / posebusters) are slow and need a second
        # interpreter, so they are opt-in.
        _cb_m.setChecked(_key not in EXTERNAL_METRICS)
        l_calc.addWidget(_cb_m)
        cb_metrics[_key] = _cb_m
    hl_calc = QtWidgets.QHBoxLayout()
    b_calc = QtWidgets.QPushButton("Calculate")
    b_calc_stop = QtWidgets.QPushButton("Cancel")
    b_calc_stop.setEnabled(False)
    hl_calc.addWidget(b_calc); hl_calc.addWidget(b_calc_stop)
    l_calc.addLayout(hl_calc)
    pb_calc = QtWidgets.QProgressBar()
    pb_calc.setVisible(False)
    l_calc.addWidget(pb_calc)
    lbl_calc = QtWidgets.QLabel("")
    lbl_calc.setWordWrap(True)
    l_calc.addWidget(lbl_calc)
    bot_l.addWidget(g_calc)

    # Non-covalent bonds
    g1, g1_en, _g1b, l1 = _section("Non-covalent bonds", enabled=True, expanded=False)
    cb_hb = _cb(l1, "Hydrogen bonds",  "#ffd900")
    hl_hba = QtWidgets.QHBoxLayout()
    hl_hba.addSpacing(20)
    _lbl_hba = QtWidgets.QLabel("min D–H···A angle:")
    sp_hba = QtWidgets.QDoubleSpinBox()
    sp_hba.setRange(0.0, 180.0); sp_hba.setDecimals(0); sp_hba.setSingleStep(5.0)
    sp_hba.setValue(_stepper.hbond_min_angle); sp_hba.setSuffix("°")
    sp_hba.setFixedWidth(70)
    _tip = ("Drop polar contacts whose hydrogen points away from the acceptor.\n"
            "PyMOL's own h_bond_max_angle is measured at the donor heavy atom, "
            "so its\n63° default admits contacts with a D–H···A angle near 100° — "
            "a good\nN···O distance with the proton aimed elsewhere.\n"
            "0 = off. Contacts with no explicit hydrogen are never filtered.")
    sp_hba.setToolTip(_tip); _lbl_hba.setToolTip(_tip)
    hl_hba.addWidget(_lbl_hba); hl_hba.addWidget(sp_hba); hl_hba.addStretch()
    l1.addLayout(hl_hba)
    cb_xb = _cb(l1, "Halogen bonds",   "#9933e6")
    cb_sb = _cb(l1, "Salt bridges",    "#e633e6")
    cb_ah = _cb(l1, "Aromatic H-Bond", "#4dd97f")
    cb_wb = _cb(l1, "Water bridges",   "#4dccf2")
    bot_l.addWidget(g1)

    # Pi interactions
    g2, g2_en, _g2b, l2 = _section("Pi interactions", enabled=True, expanded=False)
    cb_pp = _cb(l2, "Pi-pi stacking",  "#4dc0ff")
    cb_pc = _cb(l2, "Pi-cation",       "#33cc33")
    bot_l.addWidget(g2)

    # Contacts / clashes (disabled + collapsed by default)
    g3, g3_en, _g3b, l3 = _section("Contacts / Clashes", enabled=False, expanded=False)
    cb_cg = _cb(l3, "Good",  "#33cc33", checked=False)
    cb_cb = _cb(l3, "Bad",   "#ff9900", checked=False)
    cb_cu = _cb(l3, "Ugly",  "#ff2626", checked=False)
    bot_l.addWidget(g3)

    # Display options — collapsible + group enable (hides all display effects when off)
    g_disp, g_disp_en, _gdb, l_disp = _section("Display", enabled=True, expanded=False)
    cb_lb   = QtWidgets.QCheckBox("Show distance labels");  cb_lb.setChecked(True)
    cb_surf = QtWidgets.QCheckBox("Show surface");          cb_surf.setChecked(True)
    cb_rlbl = QtWidgets.QCheckBox("Show residue labels");   cb_rlbl.setChecked(True)
    cb_zoom = QtWidgets.QCheckBox("Auto-zoom to binding site");   cb_zoom.setChecked(True)
    cb_zoom.setToolTip("Frame the residue shell around the ligand on every pose "
                       "change.\nSet _stepper.zoom_to_shell = False to zoom the "
                       "ligand alone instead.")
    cb_lig_h = QtWidgets.QCheckBox("Show nonpolar H on ligands"); cb_lig_h.setChecked(False)
    cb_cstype = QtWidgets.QCheckBox("Color surface by charge"); cb_cstype.setChecked(True)
    cmb_cs = QtWidgets.QComboBox(); cmb_cs.addItems(["ramp", "tiers"])
    cmb_cs.setCurrentText(_stepper.surf_charge_style)
    cmb_cs.setFixedWidth(90)
    _tip_cs = ("tiers — flat colour on charged/polar functional atoms, two "
               "intensities per sign.\n"
               "ramp — smooth red→white→blue by ff14SB partial charge.")
    cmb_cs.setToolTip(_tip_cs)
    l_disp.addWidget(cb_lb)
    l_disp.addWidget(cb_surf)
    hl_sd = QtWidgets.QHBoxLayout()
    hl_sd.addSpacing(20)
    _lbl_sd = QtWidgets.QLabel("surface reach:")
    sp_sd = QtWidgets.QDoubleSpinBox()
    sp_sd.setRange(2.0, 12.0); sp_sd.setDecimals(1); sp_sd.setSingleStep(0.5)
    sp_sd.setValue(_stepper.surf_dist); sp_sd.setSuffix(" Å")
    sp_sd.setFixedWidth(70)
    _tip_sd = ("How far the pocket surface extends around the ligand.\n"
               "It is carved from the protein's molecular surface at this "
               "radius; larger shows more of the pocket wall.")
    sp_sd.setToolTip(_tip_sd); _lbl_sd.setToolTip(_tip_sd)
    hl_sd.addWidget(_lbl_sd); hl_sd.addWidget(sp_sd); hl_sd.addStretch()
    l_disp.addLayout(hl_sd)
    l_disp.addWidget(cb_rlbl)
    l_disp.addWidget(cb_zoom)
    l_disp.addWidget(cb_lig_h)
    l_disp.addWidget(cb_cstype)
    hl_cs = QtWidgets.QHBoxLayout()
    hl_cs.addSpacing(20)
    _lbl_cs = QtWidgets.QLabel("charge style:"); _lbl_cs.setToolTip(_tip_cs)
    hl_cs.addWidget(_lbl_cs); hl_cs.addWidget(cmb_cs); hl_cs.addStretch()
    l_disp.addLayout(hl_cs)
    bot_l.addWidget(g_disp)
    bot_l.addStretch()

    splitter.setSizes([460, 200])

    # Callbacks
    _sel_order: list = []   # table row indices in arrival order (max 2)
    _sel_busy  = [False]    # re-entrancy guard for programmatic selection changes

    def rebuild_table():
        tw_pd.setSortingEnabled(False)
        tw_pd.clearContents()
        tw_pd.setRowCount(0)
        all_props = _stepper.all_properties
        if not all_props:
            tw_pd.setColumnCount(0)
            return
        data_cols = _stepper.table_columns()
        # Column 0 = bookmark ★, then data columns
        tw_pd.setColumnCount(1 + len(data_cols))
        headers = ["★"] + ["Ligand_ID" if c == "_name" else c for c in data_cols]
        tw_pd.setHorizontalHeaderLabels(headers)
        tw_pd.setRowCount(len(all_props))
        for r, props in enumerate(all_props):
            # Bookmark column
            bm_item = _SortItem("★" if _stepper.is_bookmarked(r) else "")
            bm_item.setData(QtCore.Qt.UserRole, r)
            bm_item.setTextAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            tw_pd.setItem(r, 0, bm_item)
            # Data columns
            for c, key in enumerate(data_cols):
                val = props.get(key, "")
                if isinstance(val, int):
                    display = str(val)
                elif isinstance(val, float):
                    display = f"{val:.2f}"
                elif val == "":
                    display = ""
                else:
                    display = str(val)
                item = _SortItem(display)
                item.setData(QtCore.Qt.UserRole, r)
                if isinstance(val, (int, float)):
                    item.setData(QtCore.Qt.UserRole + 1, val)
                    item.setTextAlignment(
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                tw_pd.setItem(r, 1 + c, item)
        tw_pd.resizeColumnsToContents()
        tw_pd.setColumnWidth(0, 24)  # bookmark column stays compact
        tw_pd.setSortingEnabled(True)
        _sel_order.clear()
        _highlight_current_row()

    def _table_snapshot(selected_only):
        """(headers, rows) exactly as the table currently reads.

        Follows the visual order, so a re-sorted or re-ordered table exports the
        way it looks.  Numeric cells fall back to the value stashed in
        UserRole+1 rather than the 2-decimal display text, so precision survives
        the trip into a spreadsheet.
        """
        header = tw_pd.horizontalHeader()
        cols = [header.logicalIndex(v) for v in range(tw_pd.columnCount())]
        cols = [c for c in cols if c >= 0 and not tw_pd.isColumnHidden(c)]
        headers = []
        for c in cols:
            item = tw_pd.horizontalHeaderItem(c)
            headers.append("Bookmarked" if c == 0 else (item.text() if item else ""))
        if selected_only:
            row_idx = sorted({ix.row() for ix in
                              tw_pd.selectionModel().selectedRows()})
        else:
            row_idx = list(range(tw_pd.rowCount()))
        rows = []
        for r in row_idx:
            row = []
            for c in cols:
                item = tw_pd.item(r, c)
                if item is None:
                    row.append("")
                    continue
                raw = item.data(QtCore.Qt.UserRole + 1)
                row.append(str(raw) if isinstance(raw, (int, float))
                           else item.text())
            rows.append(row)
        return headers, rows

    def do_copy(selected_only=False):
        """Whole table from the button, selected rows from Ctrl+C.

        Kept as two explicit paths rather than one that guesses: the current
        pose is always selected in the table, so "copy the selection when there
        is one" would never copy more than a single row.
        """
        if tw_pd.rowCount() == 0:
            lbl_pd.setText("Nothing to copy.")
            return
        headers, rows = _table_snapshot(selected_only)
        if not rows:
            lbl_pd.setText("No rows selected.")
            return
        text = "\n".join(["\t".join(headers)] + ["\t".join(r) for r in rows])
        try:
            QtWidgets.QApplication.clipboard().setText(text)
        except Exception as e:
            lbl_pd.setText(f"Clipboard unavailable: {e}")
            return
        lbl_pd.setText(f"Copied {len(rows)} row(s)"
                       f"{' (selection)' if selected_only else ''}.")

    def do_export():
        if tw_pd.rowCount() == 0:
            lbl_pd.setText("Nothing to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            win, "Export pose table", "pose_table.csv",
            "CSV files (*.csv);;Tab-separated (*.tsv);;All files (*)")
        if not path:
            return
        headers, rows = _table_snapshot(False)
        delim = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh, delimiter=delim)
                writer.writerow(headers)
                writer.writerows(rows)
        except OSError as e:
            lbl_pd.setText(f"Could not write file: {e}")
            return
        lbl_pd.setText(f"Exported {len(rows)} row(s) to {os.path.basename(path)}")

    def _update_bookmark_col():
        """Refresh ★ column text to reflect current _stepper._bookmarks."""
        for r in range(tw_pd.rowCount()):
            item0 = tw_pd.item(r, 0)
            if item0 is not None:
                pi = item0.data(QtCore.Qt.UserRole)
                if pi is not None:
                    item0.setText("★" if _stepper.is_bookmarked(pi) else "")

    def _highlight_current_row():
        """Sync table selection to current single-pose index (no-op in compare mode)."""
        if _stepper._in_compare:
            return
        cur = _stepper.current_index
        tw_pd.blockSignals(True)
        try:
            _sel_order.clear()
            for r in range(tw_pd.rowCount()):
                item0 = tw_pd.item(r, 0)
                if item0 is not None and item0.data(QtCore.Qt.UserRole) == cur:
                    tw_pd.selectRow(r)
                    _sel_order.append(r)
                    tw_pd.scrollToItem(
                        item0, QtWidgets.QAbstractItemView.PositionAtCenter)
                    return
            tw_pd.clearSelection()
        finally:
            tw_pd.blockSignals(False)

    def on_selection_changed():
        if _sel_busy[0]:
            return
        selected = {idx.row() for idx in tw_pd.selectionModel().selectedRows()}
        # prune deselected rows then append new arrivals in sorted order
        _sel_order[:] = [r for r in _sel_order if r in selected]
        for r in sorted(selected):
            if r not in _sel_order:
                _sel_order.append(r)
        # rolling window: drop oldest when a third row is selected
        if len(_sel_order) > 2:
            oldest = _sel_order.pop(0)
            _sel_busy[0] = True
            try:
                tw_pd.selectionModel().select(
                    tw_pd.model().index(oldest, 0),
                    QtCore.QItemSelectionModel.Deselect |
                    QtCore.QItemSelectionModel.Rows)
            finally:
                _sel_busy[0] = False

        pose_indices = []
        for r in _sel_order:
            item0 = tw_pd.item(r, 0)
            if item0 is not None:
                pi = item0.data(QtCore.Qt.UserRole)
                if pi is not None:
                    pose_indices.append(pi)

        if len(pose_indices) == 2:
            _stepper.show_comparison(pose_indices)
            update_ui()
        elif len(pose_indices) == 1:
            ci_goto(pose_indices[0])
            update_ui()

    def update_ui():
        try:
            c = _stepper._count()
            if c > 0:
                if _stepper._in_compare:
                    lbl.setText(_stepper.compare_label())
                else:
                    lbl.setText(f"{_stepper._label()}  "
                                f"({_stepper.current_index + 1}/{c})")
                sp.setMaximum(c)
                sp.setValue(_stepper.current_index + 1)
                cur = _stepper.current_index
                b_bm.setText("★ Unbookmark" if _stepper.is_bookmarked(cur)
                             else "☆ Bookmark")
            else:
                lbl.setText("Ready - click Setup")
                b_bm.setText("☆ Bookmark")
            if g_pd.isChecked() and not _stepper._in_compare:
                _highlight_current_row()
        except RuntimeError:
            pass  # Qt widget deleted (window closed while callback was in flight)

    def do_browse():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            win, "Select scores SDF", "",
            "SDF files (*.sdf *.SDF);;All files (*)")
        if path:
            e_scores.setText(path)

    def _populate_prot_combo():
        """Refill the protein combo with objects that contain protein atoms."""
        prev = e_prot.currentText()
        e_prot.blockSignals(True)
        e_prot.clear()
        e_prot.addItem("polymer.protein")
        for n in cmd.get_names("objects"):
            if n.startswith("_"):
                continue
            try:
                if cmd.count_atoms(f"{n} and polymer.protein", state=1) > 0:
                    e_prot.addItem(n)
            except Exception:
                pass
        idx = e_prot.findText(prev)
        if idx >= 0:
            e_prot.setCurrentIndex(idx)
        elif prev:
            e_prot.setEditText(prev)
        e_prot.blockSignals(False)

    def _populate_ref_combo():
        """Refill ref_combo with current organic objects, preserving selection."""
        prev = ref_combo.currentText()
        ref_combo.blockSignals(True)
        ref_combo.clear()
        ref_combo.addItem("(none)")
        for n in cmd.get_names("objects"):
            if n.startswith("_ci_") or n in _created_objects:
                continue
            try:
                if (cmd.count_atoms(f"{n} and organic", state=1) > 0 and
                        cmd.count_atoms(f"{n} and polymer.protein", state=1) == 0):
                    ref_combo.addItem(n)
            except Exception:
                pass
        # Select auto-detected ref, then fall back to previous selection
        target = _stepper.ref_ligand or prev
        idx = ref_combo.findText(target) if target else -1
        ref_combo.setCurrentIndex(max(idx, 0))
        ref_combo.blockSignals(False)

    def _update_cmp_hb_ui():
        """Set the Docking poses checkbox from a heuristic, then gate H-bond compare on it.

        Multi-state objects are unambiguously docking poses.  Single-state objects
        could be either (one pose per ligand) or extracted protein ligands, so the
        heuristic defaults to False — the user can override by ticking the box.
        """
        has_multistate = any(cmd.count_states(o) > 1 for o, _ in _stepper.poses)
        cb_docking.blockSignals(True)
        cb_docking.setChecked(has_multistate)
        cb_docking.blockSignals(False)
        cb_cmp_hb.setEnabled(has_multistate)
        if not has_multistate:
            cb_cmp_hb.blockSignals(True)
            cb_cmp_hb.setChecked(False)
            cb_cmp_hb.blockSignals(False)
            _stepper.show_cmp_hbonds = False

    def do_setup():
        sf = e_scores.text().strip()
        if sf:
            ci_load_scores(sf)
        else:
            _stepper.sdf_records = []
        ci_setup(protein=e_prot.currentText(), ligands=e_lig.text(), mode="auto")
        _populate_prot_combo()
        _populate_ref_combo()
        _update_cmp_hb_ui()
        update_ui()
        rebuild_table()

    def on_ref_changed(text):
        prev = _stepper.ref_ligand
        _stepper.ref_ligand = None if text == "(none)" else text
        if prev != _stepper.ref_ligand and _stepper.invalidate_metrics(
                [METRIC_FIELDS[m] for m in REF_METRICS]):
            rebuild_table()
            lbl_calc.setText("Reference changed — recalculate to update "
                             "reference-based metrics.")
        if _stepper.ref_ligand:
            _color_ref_ligand(_stepper.ref_ligand)
            _stepper.show_ref = True
            cb_show_ref.blockSignals(True)
            cb_show_ref.setChecked(True)
            cb_show_ref.blockSignals(False)
        else:
            if prev:
                try:
                    cmd.disable(prev)
                except Exception:
                    pass
            if _OBJ_REF_PTS in _created_objects:
                try:
                    cmd.delete(_OBJ_REF_PTS)
                    _created_objects.discard(_OBJ_REF_PTS)
                except Exception:
                    pass
            _stepper.show_ref = False
            cb_show_ref.blockSignals(True)
            cb_show_ref.setChecked(False)
            cb_show_ref.blockSignals(False)
        ci_update(); update_ui()

    def on_show_ref(state):
        _stepper.show_ref = cb_show_ref.isChecked()
        ci_update(); update_ui()

    def on_show_pose(state):
        _stepper.show_pose = cb_show_pose.isChecked()
        ci_update(); update_ui()

    def do_clear():
        ci_clear()
        _stepper.sdf_records = []
        _stepper.all_properties = []
        _stepper.computed.clear()
        _stepper._bookmarks.clear()
        _stepper._sdf_mismatch = None
        _stepper.ref_ligand = None
        _sel_order.clear()
        ref_combo.blockSignals(True)
        ref_combo.clear(); ref_combo.addItem("(none)")
        ref_combo.blockSignals(False)
        lbl.setText("Ready - click Setup")
        tw_pd.setSortingEnabled(False)
        tw_pd.clearContents()
        tw_pd.setRowCount(0)
        tw_pd.setColumnCount(0)
        cb_docking.setChecked(False)  # triggers on_docking_toggle → disables cb_cmp_hb

    def _table_adjacent(delta):
        """Navigate to the row delta steps from the current row in table order."""
        n_rows = tw_pd.rowCount()
        if n_rows == 0 or tw_pd.columnCount() == 0:
            # No table (or table has rows but no columns / items) — fall back to poses order
            if delta > 0:
                ci_next()
            else:
                ci_prev()
            update_ui()
            return
        cur = _stepper.current_index
        # Find the row that matches the current pose
        cur_row = -1
        for r in range(n_rows):
            item0 = tw_pd.item(r, 0)
            if item0 is not None and item0.data(QtCore.Qt.UserRole) == cur:
                cur_row = r
                break
        next_row = (cur_row + delta) % n_rows
        item0 = tw_pd.item(next_row, 0)
        if item0 is not None:
            pose_idx = item0.data(QtCore.Qt.UserRole)
            if pose_idx is not None:
                ci_goto(pose_idx)
        update_ui()

    def do_prev():
        _table_adjacent(-1)

    def do_next():
        _table_adjacent(1)

    def do_refresh():
        ci_refresh(); update_ui()

    def do_go():
        ci_goto(sp.value() - 1); update_ui()

    def _tog(cb, attr):
        def h(state):
            setattr(_stepper, attr, cb.isChecked())
            ci_update(); update_ui()
        return h

    def _group_tog(group_en, cb_attr_pairs):
        def h(checked):
            for cb, attr in cb_attr_pairs:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
                setattr(_stepper, attr, checked)
            ci_update(); update_ui()
        return h

    _calc_job: list = [None]

    def do_calculate():
        running = _calc_job[0]
        if running is not None and not running.finished:
            return
        wanted = [k for k in METRIC_ORDER if cb_metrics[k].isChecked()]
        if not wanted:
            lbl_calc.setText("Select at least one metric.")
            return
        missing = [k for k in wanted
                   if k in EXTERNAL_METRICS and not _external_python_candidates(k)]
        if missing:
            mods = ", ".join("/".join(_EXTERNAL_REQUIRES[k]) for k in missing)
            lbl_calc.setText(f"No Python found with {mods} installed — "
                             f"set $POSEVIEWER_PYTHON to one.")
            return
        try:
            job = _collect_metric_inputs(wanted)
        except RuntimeError as e:
            lbl_calc.setText(str(e))
            return
        _calc_job[0] = job
        job.start()
        b_calc.setEnabled(False)
        b_calc_stop.setEnabled(True)
        pb_calc.setRange(0, job.total_n)
        pb_calc.setValue(0)
        pb_calc.setVisible(True)
        lbl_calc.setText("Starting…")
        calc_timer.start()

    def do_calc_cancel():
        job = _calc_job[0]
        if job is not None:
            job.cancel()
            lbl_calc.setText("Cancelling…")
        b_calc_stop.setEnabled(False)

    def _calc_poll():
        job = _calc_job[0]
        if job is None:
            calc_timer.stop()
            return
        try:
            done, total, msg = job.progress()
            pb_calc.setRange(0, max(1, total))
            pb_calc.setValue(done)
            if not job.finished:
                lbl_calc.setText(msg)
                return
            calc_timer.stop()
            _calc_job[0] = None
            b_calc.setEnabled(True)
            b_calc_stop.setEnabled(False)
            pb_calc.setVisible(False)
            _stepper.merge_metrics(job.results)
            _stepper._table_dirty = False
            rebuild_table()
            update_ui()
            parts = ["Cancelled."] if job.cancelled else []
            for metric in job.metrics:
                field = METRIC_FIELDS[metric]
                n = sum(1 for v in job.results.values()
                        if isinstance(v.get(field), (int, float)))
                parts.append(f"{field}: {n}/{len(job.pose_keys)}")
            for note in job.notes:
                print(f"PoseViewer: {note}")
            for err in job.errors:
                print(f"PoseViewer: {err}")
                parts.append(err)
            lbl_calc.setText("   ".join(parts))
        except RuntimeError:
            calc_timer.stop()   # window closed mid-job

    # Parented to win so Qt stops and destroys it when the window closes.
    calc_timer = QtCore.QTimer(win)
    calc_timer.setInterval(150)
    calc_timer.timeout.connect(_calc_poll)
    win._calc_timer = calc_timer

    b_calc.clicked.connect(do_calculate)
    b_calc_stop.clicked.connect(do_calc_cancel)
    b_browse.clicked.connect(do_browse)
    b_setup.clicked.connect(do_setup)
    b_clear.clicked.connect(do_clear)
    ref_combo.currentTextChanged.connect(on_ref_changed)
    cb_show_ref.stateChanged.connect(on_show_ref)
    cb_show_pose.stateChanged.connect(on_show_pose)
    b_prev.clicked.connect(do_prev)
    b_refresh.clicked.connect(do_refresh)
    b_next.clicked.connect(do_next)
    b_go.clicked.connect(do_go)


    cb_hb.stateChanged.connect(_tog(cb_hb, "show_hbonds"))
    sp_hba.valueChanged.connect(lambda v: [
        setattr(_stepper, "hbond_min_angle", float(v)),
        ci_update(), update_ui()])
    cb_xb.stateChanged.connect(_tog(cb_xb, "show_halogen"))
    cb_sb.stateChanged.connect(_tog(cb_sb, "show_salt"))
    cb_ah.stateChanged.connect(_tog(cb_ah, "show_arom_hb"))
    cb_wb.stateChanged.connect(_tog(cb_wb, "show_water"))
    cb_pp.stateChanged.connect(_tog(cb_pp, "show_pipi"))
    cb_pc.stateChanged.connect(_tog(cb_pc, "show_pi_cation"))
    cb_cg.stateChanged.connect(_tog(cb_cg, "show_clash_good"))
    cb_cb.stateChanged.connect(_tog(cb_cb, "show_clash_bad"))
    cb_cu.stateChanged.connect(_tog(cb_cu, "show_clash_ugly"))
    cb_zoom.stateChanged.connect(lambda s: setattr(_stepper, "auto_zoom", cb_zoom.isChecked()))

    def do_bookmark():
        _stepper.toggle_bookmark()
        update_ui()
        _update_bookmark_col()

    def do_bm_list():
        ci_bookmarks()

    def do_bm_export():
        if not _stepper._bookmarks:
            lbl_pd.setText("No bookmarks to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            win, "Export bookmarked poses", "bookmarks.sdf",
            "SDF files (*.sdf);;All files (*)")
        if not path:
            return
        try:
            n = _export_poses_sdf(path, only_marked=True)
        except OSError as e:
            lbl_pd.setText(f"Could not write file: {e}")
            return
        lbl_pd.setText(f"Exported {n} bookmarked pose(s) to {os.path.basename(path)}")

    def do_toggle_cstype(state=None):
        _stepper.color_surf_by_type = cb_cstype.isChecked()
        cmb_cs.setEnabled(cb_cstype.isChecked())
        if _OBJ_SURF in _created_objects:
            if _stepper.color_surf_by_type:
                _color_surface_by_type(_OBJ_SURF)
            else:
                cmd.set("surface_color", "grey80", _OBJ_SURF)
    cb_cstype.stateChanged.connect(do_toggle_cstype)

    def do_cs_style(txt):
        _stepper.surf_charge_style = txt
        if _stepper.color_surf_by_type and _OBJ_SURF in _created_objects:
            _color_surface_by_type(_OBJ_SURF)
    cmb_cs.currentTextChanged.connect(do_cs_style)

    b_bm.clicked.connect(do_bookmark)
    b_bm_list.clicked.connect(do_bm_list)
    b_bm_export.clicked.connect(do_bm_export)

    g1_en.stateChanged.connect(_group_tog(g1_en, [
        (cb_hb, "show_hbonds"), (cb_xb, "show_halogen"),
        (cb_sb, "show_salt"),   (cb_ah, "show_arom_hb"),
        (cb_wb, "show_water"),
    ]))
    g2_en.stateChanged.connect(_group_tog(g2_en, [
        (cb_pp, "show_pipi"), (cb_pc, "show_pi_cation"),
    ]))
    g3_en.stateChanged.connect(_group_tog(g3_en, [
        (cb_cg, "show_clash_good"), (cb_cb, "show_clash_bad"), (cb_cu, "show_clash_ugly"),
    ]))

    def do_toggle_surf(state=None):
        _stepper.show_surface = cb_surf.isChecked()
        if _OBJ_SURF in _created_objects:
            if _stepper.show_surface:
                cmd.show("surface", _OBJ_SURF)
            else:
                cmd.hide("surface", _OBJ_SURF)
    cb_surf.stateChanged.connect(do_toggle_surf)

    def do_surf_dist(v):
        _stepper.surf_dist = float(v)
        if _stepper.poses:
            _stepper.rebuild_shell()
    sp_sd.valueChanged.connect(do_surf_dist)

    def do_toggle_rlbl(state=None):
        if _shell_sel is not None:
            sel = f"({_shell_sel}) and name CA"
            if cb_rlbl.isChecked():
                cmd.show("labels", sel)
            else:
                cmd.hide("labels", sel)
    cb_rlbl.stateChanged.connect(do_toggle_rlbl)

    _disp_saved: dict = {}

    def do_disp_group_tog(checked):
        """Turn every display option off together, then put them back as they were.

        Ticking the group used to switch everything on, which silently enabled
        nonpolar ligand H even though that defaults to off.
        """
        boxes = (cb_lb, cb_surf, cb_rlbl, cb_zoom, cb_lig_h, cb_cstype)
        if checked:
            wanted = [_disp_saved.get(cb, cb is not cb_lig_h) for cb in boxes]
        else:
            _disp_saved.clear()
            _disp_saved.update({cb: cb.isChecked() for cb in boxes})
            wanted = [False] * len(boxes)
        for cb, want in zip(boxes, wanted):
            cb.blockSignals(True)
            cb.setChecked(want)
            cb.blockSignals(False)
        _stepper.show_labels = cb_lb.isChecked()
        _stepper.show_lig_h  = cb_lig_h.isChecked()
        _stepper.auto_zoom   = cb_zoom.isChecked()
        _stepper.color_surf_by_type = cb_cstype.isChecked()
        do_toggle_surf()
        do_toggle_rlbl()
        do_toggle_cstype()
        ci_update(); update_ui()
    g_disp_en.stateChanged.connect(do_disp_group_tog)

    cb_lb.stateChanged.connect(lambda s: [
        setattr(_stepper, "show_labels", cb_lb.isChecked()),
        ci_update(), update_ui()])

    cb_lig_h.stateChanged.connect(lambda s: [
        setattr(_stepper, "show_lig_h", cb_lig_h.isChecked()),
        ci_update(), update_ui()])

    def on_docking_toggle(checked):
        cb_cmp_hb.setEnabled(checked)
        if not checked:
            cb_cmp_hb.blockSignals(True)
            cb_cmp_hb.setChecked(False)
            cb_cmp_hb.blockSignals(False)
            _stepper.show_cmp_hbonds = False
    cb_docking.stateChanged.connect(on_docking_toggle)

    def on_cmp_hb(state):
        _stepper.show_cmp_hbonds = cb_cmp_hb.isChecked()
        if _stepper._in_compare:
            _stepper.show_comparison(_stepper._cmp_indices)
            update_ui()
    cb_cmp_hb.stateChanged.connect(on_cmp_hb)

    b_copy.clicked.connect(lambda: do_copy(False))
    b_export.clicked.connect(do_export)
    sc_copy = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+C"), tw_pd)
    sc_copy.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
    sc_copy.activated.connect(lambda: do_copy(True))

    tw_pd.itemSelectionChanged.connect(on_selection_changed)

    g_pd.toggled.connect(lambda checked: update_ui())

    for key, fn in [(QtCore.Qt.Key_Right, do_next),
                    (QtCore.Qt.Key_Left, do_prev)]:
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), win)
        sc.activated.connect(fn)

    # Detect external PyMOL state/object changes (built-in slider / play buttons /
    # ci_next from CLI / clicking enable on a different object in the panel /
    # objects added or deleted from the session).
    # IMPORTANT: timer must be stored on `win` (not as a local variable) so that
    # neither the QTimer C++ object nor its Python wrapper are garbage-collected
    # when _open_gui() returns.  Passing `win` as parent also lets Qt stop and
    # destroy the timer automatically when the window is closed.
    win._known_pymol_objs = set(cmd.get_names("objects"))
    win._sync_tick = 0

    def _sync_if_external_change():
        if _stepper._table_dirty:
            # ci_calc merged new metrics from the command line (another thread),
            # so refresh the table here, in the GUI thread.
            _stepper._table_dirty = False
            rebuild_table()
            update_ui()
        if _stepper._in_compare or not _stepper.poses:
            return
        win._sync_tick += 1
        try:
            pymol_st = cmd.get_state()
        except Exception:
            return

        # --- Object list change detection (cheap set diff; atom-counting only on change) ---
        try:
            current_objs = set(cmd.get_names("objects"))
        except Exception:
            current_objs = win._known_pymol_objs
        if current_objs != win._known_pymol_objs:
            removed = win._known_pymol_objs - current_objs
            added   = current_objs - win._known_pymol_objs
            win._known_pymol_objs = current_objs
            _populate_prot_combo()

            if _stepper.mode == "objects":
                pose_objs = {o for o, _ in _stepper.poses}
                deleted = pose_objs & removed
                if deleted:
                    _stepper.poses = [(o, s) for o, s in _stepper.poses if o not in deleted]
                    _stepper.current_index = min(_stepper.current_index, max(0, len(_stepper.poses) - 1))
                    if not _stepper.poses:
                        _stepper._cleanup_cmp()
                        _clear_contacts()
                        _populate_ref_combo()
                        update_ui()
                        return
                    _stepper._build_obj_colors()
                    _stepper._prefetch_all_properties()
                    _stepper._show_current()
                    _populate_ref_combo()
                    update_ui()
                    rebuild_table()
                    return

                new_ligs = []
                for n in sorted(added):
                    if n in _created_objects:
                        continue
                    try:
                        if (cmd.count_atoms(f"{n} and organic", state=1) > 0 and
                                cmd.count_atoms(
                                    f"{n} and ({_stepper.protein_sel or 'polymer.protein'})",
                                    state=1) == 0):
                            new_ligs.append(n)
                    except Exception:
                        pass
                if new_ligs:
                    for n in new_ligs:
                        for st in range(1, max(1, cmd.count_states(n)) + 1):
                            _stepper.poses.append((n, st))
                    _stepper._build_obj_colors()
                    _stepper._prefetch_all_properties()
                    update_ui()
                    rebuild_table()
                _populate_ref_combo()

            elif _stepper.mode == "states":
                if _stepper.state_object in removed:
                    _stepper.poses = []
                    _populate_ref_combo()
                    update_ui()
                    return
                _populate_ref_combo()

        # --- States-count sync: detect states added/removed from the tracked object ---
        # Only run every 4th tick (every 2 s) — state count rarely changes.
        if (win._sync_tick % 4 == 0 and
                _stepper.mode == "states" and _stepper.state_object and _stepper.poses):
            try:
                n_states = cmd.count_states(_stepper.state_object)
            except Exception:
                n_states = len(_stepper.poses)
            if n_states != len(_stepper.poses):
                _stepper.poses = [(_stepper.state_object, st)
                                  for st in range(1, n_states + 1)]
                if not _stepper.poses:
                    _clear_contacts()
                    update_ui()
                    return
                new_idx = min(_stepper.current_index, n_states - 1)
                if new_idx != _stepper.current_index:
                    _stepper.current_index = new_idx
                    _stepper._show_current()
                _stepper._prefetch_all_properties()
                update_ui()
                rebuild_table()

        if not _stepper.poses:
            return
        cur_obj, cur_st = _stepper.poses[_stepper.current_index]

        if _stepper.mode == "objects":
            # Detect if a different pose object was enabled externally.
            # get_names("objects", 1) returns only enabled objects.
            try:
                enabled = set(cmd.get_names("objects", 1))
            except Exception:
                enabled = set()
            pose_objs = list(dict.fromkeys(o for o, _ in _stepper.poses))
            enabled_poses = [o for o in pose_objs if o in enabled]
            # A new object is active if cur_obj was disabled or another pose
            # object became enabled alongside it.
            if cur_obj not in enabled:
                new_obj = enabled_poses[0] if enabled_poses else None
            elif len(enabled_poses) > 1:
                new_obj = next((o for o in enabled_poses if o != cur_obj), None)
            else:
                new_obj = None

            if new_obj is not None:
                # Prefer (new_obj, pymol_st); fall back to first state of new_obj.
                for i, (o, s) in enumerate(_stepper.poses):
                    if o == new_obj and s == pymol_st:
                        _stepper.current_index = i
                        break
                else:
                    for i, (o, s) in enumerate(_stepper.poses):
                        if o == new_obj:
                            _stepper.current_index = i
                            break
                _stepper._show_current()
                update_ui()
                return

            # Object unchanged — check for a state change within cur_obj.
            if cur_st != pymol_st:
                for i, (o, s) in enumerate(_stepper.poses):
                    if o == cur_obj and s == pymol_st:
                        _stepper.current_index = i
                        _stepper._update(o, state=s)
                        update_ui()
                        break

        else:
            # States mode: only state sync matters.
            if cur_st == pymol_st:
                return
            for i, (o, s) in enumerate(_stepper.poses):
                if s == pymol_st:
                    _stepper.current_index = i
                    _stepper._update(o, state=s)
                    update_ui()
                    break

    win._sync_timer = QtCore.QTimer(win)
    win._sync_timer.setInterval(500)
    win._sync_timer.timeout.connect(_sync_if_external_change)
    win._sync_timer.start()

    _populate_prot_combo()   # fill combo with objects already loaded at open time
    win.show(); win.raise_()


def _set_gui_none():
    global _gui_window
    _gui_window = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

__version__ = "1.9.8"
print(f"PoseViewer v{__version__} loaded.")
print("  ci_gui     - open GUI panel")
print("  ci_setup   - setup from command line")
print("  ci_refresh     - sync to current PyMOL state / state slider")
print("  ci_load_scores - load per-pose properties from SDF file")
print("  ci_calc        - compute MCS_RMSD / Shape_Sim / Ref_Sim / PLIF_Sim / PB_Flags")
print("  ci_export      - write the pose table to CSV/TSV")
print("  ci_export_sdf  - write bookmarked (or all) poses to a combined SDF")
print("  ci_hbond_angle - min D-H...A angle for H-bonds (default "
      f"{HBOND_MIN_DHA_ANGLE:.0f} deg, 0 = off)")
print("  LEFT/RIGHT arrow keys after setup")
