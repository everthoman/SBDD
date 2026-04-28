"""
PoseViewer - PyMOL Plugin
================================
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
Date:    2026-03-24
Version: 1.0
License: MIT
"""

from __future__ import annotations

import math
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
# H-bonds handled entirely by PyMOL's cmd.distance(mode=2)

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

SHELL_DIST = 5.0

DASH_RADIUS = 0.06
DASH_GAP = 0.35
DASH_LENGTH = 0.20
LABEL_SIZE = 14

# ---------------------------------------------------------------------------
# Object tracking
# ---------------------------------------------------------------------------

_created_objects: Set[str] = set()
_shell_sel: Optional[str] = None   # selection used for lines/labels (no copy object)

_OBJ_PTS        = "_ci_pts"
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
}

def _track(name):
    _created_objects.add(name)

def _clear_shell():
    global _shell_sel
    if _shell_sel is not None:
        try:
            cmd.hide("lines",  _shell_sel)
            cmd.hide("labels", f"({_shell_sel}) and name CA")
        except Exception:
            pass
        _shell_sel = None
    if _OBJ_SURF in _created_objects:
        try: cmd.delete(_OBJ_SURF)
        except Exception: pass
        _created_objects.discard(_OBJ_SURF)

def _clear_all():
    _clear_shell()
    for name in list(_created_objects):
        try: cmd.delete(name)
        except Exception: pass
    _created_objects.clear()
    _stepper._cleanup_cmp()

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _norm(v):
    if np is not None:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v] if mag > 1e-9 else v

def _dot(a, b):
    if np is not None:
        return float(np.dot(a, b))
    return sum(x * y for x, y in zip(a, b))

def _dist(a, b):
    if np is not None:
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _centroid(pts):
    if np is not None:
        return np.mean(pts, axis=0)
    n = len(pts)
    return [sum(p[i] for p in pts) / n for i in range(3)]

def _angle_normals(n1, n2):
    c = abs(_dot(_norm(n1), _norm(n2)))
    return math.degrees(math.acos(min(1.0, max(0.0, c))))

# ---------------------------------------------------------------------------
# Ring detection
# ---------------------------------------------------------------------------

def _nonpolar_h_indices(model):
    """Return indices of H atoms bonded only to C (nonpolar H, excluded from clash detection)."""
    adj = defaultdict(set)
    for bond in model.bond:
        i, j = bond.index
        adj[i].add(j); adj[j].add(i)
    nonpolar = set()
    for i, a in enumerate(model.atom):
        if a.symbol.strip().capitalize() != "H":
            continue
        if not any(model.atom[nb].symbol.strip().capitalize() in {"N", "O", "S", "F"}
                   for nb in adj[i]):
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
    adj: Dict[int, Set[int]] = defaultdict(set)
    for bond in model.bond:
        i, j = bond.index
        adj[i].add(j); adj[j].add(i)

    ring_elems = {"C", "N", "O", "S"}
    rings: List[List[int]] = []
    seen: Set[frozenset] = set()

    for start in range(n):
        if model.atom[start].symbol not in ring_elems:
            continue
        _dfs_rings(adj, start, start, [start], set(), rings, seen, model, 6)

    out = []
    for ring in rings:
        if len(ring) not in (5, 6):
            continue
        ok = True
        for idx in ring:
            a = model.atom[idx]
            if a.symbol not in ring_elems:
                ok = False; break
            # sp2 carbon: exactly 3 total bonds (2 ring + 1 substituent/H)
            # sp3 carbon: 4 total bonds → not aromatic
            if a.symbol == "C":
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
        if a.symbol != "C":
            continue
        for nb in adj[idx]:
            if model.atom[nb].symbol == "H":
                ch_atoms.append((a, tuple(a.coord), tuple(model.atom[nb].coord)))
                break
    return ch_atoms


# ---------------------------------------------------------------------------
# Interaction result (for non-hbond types detected geometrically)
# ---------------------------------------------------------------------------

class InteractionResult:
    def __init__(self):
        self.hbond_count: int = 0   # set by visualize() from cmd.distance return value
        self.halogen: List[dict] = []
        self.salt_bridges: List[dict] = []
        self.arom_hbonds: List[dict] = []
        self.pipi: List[dict] = []
        self.pi_cation: List[dict] = []
        self.clash_good: List[dict] = []
        self.clash_bad: List[dict] = []
        self.clash_ugly: List[dict] = []


# ---------------------------------------------------------------------------
# Detection (non-hbond interactions only; H-bonds use PyMOL mode=2)
# ---------------------------------------------------------------------------

def detect_interactions(
    lig_sel, prot_sel, state=-1,
    do_halogen=True, do_salt=True, do_arom_hb=True,
    do_pipi=True, do_pi_cation=True,
    do_clash_good=False, do_clash_bad=False, do_clash_ugly=False,
) -> InteractionResult:

    result = InteractionResult()
    search = max(HALOGEN_DIST_MAX, SALT_BRIDGE_DIST_MAX,
                 PI_CATION_DIST_MAX, PIPI_ETF_DIST_MAX) + 2.0

    _TMP = "_ci_tmp"
    lig_model = cmd.get_model(lig_sel, state=state)
    # cmd.select evaluates proximity at the current global state (ligand pose),
    # then cmd.get_model extracts protein coords at state=1 (protein is single-state).
    cmd.select(_TMP, f"({prot_sel}) within {search} of ({lig_sel})")
    prot_model = cmd.get_model(_TMP, state=1)
    cmd.delete(_TMP)
    if not lig_model.atom or not prot_model.atom:
        return result

    lig_atoms = [(i, a, tuple(a.coord)) for i, a in enumerate(lig_model.atom)]
    prot_atoms = [(i, a, tuple(a.coord)) for i, a in enumerate(prot_model.atom)]

    do_any_clash = do_clash_good or do_clash_bad or do_clash_ugly
    if do_any_clash:
        lig_nonpolar_h = _nonpolar_h_indices(lig_model)
        prot_nonpolar_h = _nonpolar_h_indices(prot_model)
    else:
        lig_nonpolar_h = prot_nonpolar_h = set()

    def _el(a): return a.symbol.strip().capitalize()
    def _il(a): return f"{a.resn} {a.name}"
    def _ip(a): return f"{a.chain}/{a.resn}{a.resi}.{a.name}"

    # --- Pairwise ---
    for li, la, lc in lig_atoms:
        le = _el(la)
        for pi, pa, pc in prot_atoms:
            pe = _el(pa)
            d = _dist(lc, pc)

            if do_halogen and d <= HALOGEN_DIST_MAX:
                if ((le in HALOGEN_DONORS and pe in HALOGEN_ACCEPTORS) or
                    (pe in HALOGEN_DONORS and le in HALOGEN_ACCEPTORS)):
                    result.halogen.append({
                        "p1": lc, "p2": pc, "dist": d,
                        "info1": _il(la), "info2": _ip(pa)})

            if do_salt and d <= SALT_BRIDGE_DIST_MAX:
                lpos = (le == "N" and la.formal_charge > 0)
                lneg = (le == "O" and la.formal_charge < 0)
                ppos = (pa.resn in ("ARG","LYS","HIS","HID","HIE","HIP")
                        and pa.name in ("NH1","NH2","NE","NZ","ND1","NE2"))
                pneg = ((pa.resn == "ASP" and pa.name in ("OD1","OD2"))
                        or (pa.resn == "GLU" and pa.name in ("OE1","OE2")))
                if (lpos and pneg) or (lneg and ppos):
                    result.salt_bridges.append({
                        "p1": lc, "p2": pc, "dist": d,
                        "info1": _il(la), "info2": _ip(pa)})

            if (do_any_clash
                    and li not in lig_nonpolar_h
                    and pi not in prot_nonpolar_h):
                vdw = VDW_RADII.get(le, 1.70) + VDW_RADII.get(pe, 1.70)
                max_d = vdw * CLASH_GOOD_FRAC
                if d <= max_d:
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
                    elif le != "H" and pe != "H":
                        if do_clash_good:
                            result.clash_good.append({
                                "p1": lc, "p2": pc, "dist": d, "vdw": vdw,
                                "quality": "good",
                                "info1": _il(la), "info2": _ip(pa)})

    # --- Ring-based ---
    lig_rings_info = []
    prot_rings_info = []
    _prm = None
    _prm_adj = None
    _lm = None
    _lm_adj = None

    need_rings = do_pipi or do_arom_hb or do_pi_cation
    if need_rings:
        try:
            _lm = cmd.get_model(lig_sel, state=state)
            lig_rings_info, _lm_adj = _get_rings_info(_lm)
        except Exception:
            pass
        try:
            prs = (f"({prot_sel} within {PI_CATION_DIST_MAX+3} of ({lig_sel}))"
                   f" and (resn PHE+TYR+TRP+HIS+HIE+HID+HIP)")
            cmd.select(_TMP, prs)
            _prm = cmd.get_model(_TMP, state=1)
            cmd.delete(_TMP)
            prot_rings_info, _prm_adj = _get_rings_info(_prm)
        except Exception:
            pass

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
                        "angle": ang, "type": "face-to-face", "info2": info})
                elif (d <= PIPI_ETF_DIST_MAX and
                      PIPI_ETF_ANGLE_MIN <= ang <= PIPI_ETF_ANGLE_MAX):
                    result.pipi.append({
                        "p1": list(lrc), "p2": list(prc), "dist": d,
                        "angle": ang, "type": "edge-to-face", "info2": info})

    # Aromatic H-bonds: aromatic C-H ... acceptor (O/N/S)
    if do_arom_hb:
        if _lm and _lm_adj and lig_rings_info:
            lig_ring_indices = [r for r, _, _ in lig_rings_info]
            lig_ch = _get_aromatic_ch_atoms(_lm, lig_ring_indices, _lm_adj)
            for ca, cc, hc in lig_ch:
                for _pi, pa, pc in prot_atoms:
                    if _el(pa) not in AROM_HBOND_ACCEPTORS:
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
                for _li, la, lc in lig_atoms:
                    if _el(la) not in AROM_HBOND_ACCEPTORS:
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
        pcs = (f"({prot_sel} within {PI_CATION_DIST_MAX+1} of ({lig_sel}))"
               f" and ((resn ARG and name CZ) or (resn LYS and name NZ))")
        try:
            cmd.select(_TMP, pcs)
            pcm = cmd.get_model(_TMP, state=1)
            cmd.delete(_TMP)
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
        if _prm:
            for _li, la, lc in lig_atoms:
                if la.formal_charge > 0:
                    for pri, prc, _ in prot_rings_info:
                        d = _dist(lc, prc)
                        if d <= PI_CATION_DIST_MAX:
                            a0 = _prm.atom[pri[0]]
                            result.pi_cation.append({
                                "p1": lc, "p2": list(prc), "dist": d,
                                "info1": _il(la),
                                "info2": f"{a0.chain}/{a0.resn}{a0.resi} ring"})

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _clear_contacts():
    keep = {_OBJ_SURF}
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


def _add_pair(pts, pid, p1, p2, dist_name):
    r = str(pid)
    cmd.pseudoatom(pts, pos=list(p1), resi=r, name="L", chain="X")
    cmd.pseudoatom(pts, pos=list(p2), resi=r, name="P", chain="X")
    cmd.distance(dist_name,
                 f"{pts} and resi {r} and name L",
                 f"{pts} and resi {r} and name P")


def visualize(lig_sel, prot_sel, result: InteractionResult,
              show_hbonds=True, show_labels=True, state=-1,
              name_prefix="", clear=True):
    """Visualize all interactions.

    H-bonds are created via PyMOL's built-in polar contact detection
    (cmd.distance mode=2). All other types use pseudoatom pairs.

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

    # --- H-bonds via PyMOL polar contacts (mode=2) ---
    hb_name = name_prefix + _INTERACTION_NAMES["hbonds"]
    if show_hbonds:
        try:
            n_hb = cmd.distance(hb_name, lig_sel, prot_sel, mode=2)
            if n_hb is not None and n_hb > 0:
                result.hbond_count = int(n_hb)
                _track(hb_name)
                _style(hb_name, "ci_hbond", radius=DASH_RADIUS * r_scale)
            else:
                result.hbond_count = 0
                try: cmd.delete(hb_name)
                except Exception: pass
        except Exception:
            result.hbond_count = 0
            try: cmd.delete(hb_name)
            except Exception: pass

    # --- All other types via pseudoatom pairs ---
    pts = _OBJ_REF_PTS if name_prefix else _OBJ_PTS
    _track(pts)
    pid = 0

    def _draw(items, obj_name, color, **kw):
        nonlocal pid
        if not items:
            return
        _track(obj_name)
        for it in items:
            _add_pair(pts, pid, it["p1"], it["p2"], obj_name)
            pid += 1
        _style(obj_name, color, **kw)

    N = {k: name_prefix + v for k, v in _INTERACTION_NAMES.items()}
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

    # Contacts/clashes: never show distance labels (too cluttered)
    for q in ("clash_good", "clash_bad", "clash_ugly"):
        if N[q] in _created_objects:
            cmd.hide("labels", N[q])

    if pid > 0:
        cmd.hide("everything", pts)

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


# ---------------------------------------------------------------------------
# Residue shell
# ---------------------------------------------------------------------------

def _create_shell(protein_sel, ligand_sels, dist=SHELL_DIST):
    global _shell_sel
    lig_union = _lig_union(ligand_sels)

    _clear_shell()

    sel = f"byres (({protein_sel}) within {dist} of ({lig_union}))"
    try:
        cmd.show("lines", sel)
        cmd.hide("lines", f"({sel}) and elem H and not (neighbor (elem N+O+S))")
        cmd.label(f"({sel}) and name CA", '"%s %s" % (resn, resi)')
        _shell_sel = sel
    except Exception as e:
        print(f"PoseViewer: shell setup warning: {e}")
        return

    # Transparent surface on the atom-based shell (no byres expansion)
    atom_surf_sel = f"({protein_sel}) within {dist} of ({lig_union})"
    try:
        cmd.create(_OBJ_SURF, atom_surf_sel)
        _track(_OBJ_SURF)
        cmd.hide("everything", _OBJ_SURF)
        cmd.show("surface", _OBJ_SURF)
        cmd.set("surface_color", "grey80", _OBJ_SURF)
        cmd.set("transparency", 0.4, _OBJ_SURF)
    except Exception:
        pass



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
        self.auto_zoom = True
        self.last_properties: dict = {}
        self.sdf_records: list = []   # populated by ci_load_scores / GUI browse
        self.all_properties: list = []  # one dict per pose, built at setup time
        self.poses: list = []          # [(obj_name, state_1based), ...]
        self.ref_ligand: Optional[str] = None
        self.show_ref: bool = True
        self.show_pose: bool = True
        self.show_lig_h: bool = False
        self._cmp_objs:      List[str]      = []
        self._cmp_indices:   List[int]      = []
        self._in_compare:    bool           = False
        self._obj_colors:    Dict[str, str] = {}
        self.show_cmp_hbonds: bool          = False

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
            if n == obj:
                continue
            try:
                if (cmd.count_atoms(f"{n} and organic") > 0 and
                        cmd.count_atoms(f"{n} and ({prot})") == 0):
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
        _create_shell(prot, all_ligs)
        self._show_current()
        self._prefetch_all_properties()
        self._build_obj_colors()

    def _count(self):
        return len(self.poses)

    def _label(self):
        if not self.poses:
            return "none"
        obj, st = self.poses[self.current_index]
        return f"{obj} state {st}" if cmd.count_states(obj) > 1 else obj

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
                cmd.zoom(obj, buffer=3.0, animate=1, state=st2)
            self._update(obj, state=st2)
        else:
            self._show_current()

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
        if self.auto_zoom:
            if self.show_pose:
                cmd.zoom(obj, buffer=3.0, animate=1, state=st)
            elif self.ref_ligand and self.show_ref:
                cmd.zoom(self.ref_ligand, buffer=3.0, animate=1, state=1)
        self._update(obj, state=st)

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
                do_clash_ugly=self.show_clash_ugly)
            self.last_result = r
            if self.sdf_records:
                idx = self.current_index
                self.last_properties = (self.sdf_records[idx]
                                        if 0 <= idx < len(self.sdf_records)
                                        else {})
            else:
                self.last_properties = _get_pose_properties(
                    lig, state if state > 0 else 1)
            if self.show_pose:
                visualize(lig, self.protein_sel, r,
                          show_hbonds=self.show_hbonds,
                          show_labels=self.show_labels, state=state)
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
                        do_clash_ugly=self.show_clash_ugly)
                    visualize(self.ref_ligand, self.protein_sel, r_ref,
                              show_hbonds=self.show_hbonds,
                              show_labels=self.show_labels, state=1,
                              name_prefix="ref_", clear=not self.show_pose)
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
        if not self.sdf_records:
            self.all_properties = [_get_pose_properties(obj, st)
                                    for obj, st in self.poses]
            return

        if self.mode == "states":
            # Single-object states mode: records are 1-to-1 with poses in SDF order
            self.all_properties = list(self.sdf_records)
            return

        # Objects mode with SDF loaded: match records to poses by _name.
        # Group SDF records by molecule title, preserving their in-file order.
        # That order corresponds to state indices (1st occurrence = state 1, etc.)
        # because docking software emits poses per compound in score order.
        from collections import defaultdict
        by_name: dict = defaultdict(list)
        for rec in self.sdf_records:
            by_name[rec.get("_name", "")].append(rec)

        obj_names = {o for o, _ in self.poses}
        if obj_names <= set(by_name.keys()):
            # _name matches object names — use name-based alignment
            counters: dict = defaultdict(int)
            self.all_properties = []
            for obj, _st in self.poses:
                recs = by_name[obj]
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

        # H-bond count stored by visualize() from cmd.distance return value
        if r.hbond_count > 0:
            lines.append(f"\nH-bonds: {r.hbond_count} (PyMOL polar contacts)")

        def _s(title, items, extra_fn=None):
            if not items: return
            lines.append(f"\n{title} ({len(items)}):")
            for it in items:
                ex = extra_fn(it) if extra_fn else ""
                i1 = it.get("info1", "")
                i2 = it.get("info2", "")
                lines.append(f"  {i1} -- {i2}  {it['dist']:.2f} A{ex}")

        _s("Halogen bonds", r.halogen)
        _s("Salt bridges", r.salt_bridges)
        _s("Aromatic H-bonds", r.arom_hbonds)
        _s("Pi-pi", r.pipi,
           lambda x: f"  {x['angle']:.0f} deg [{x['type']}]")
        _s("Pi-cation", r.pi_cation)
        _s("Clash good", r.clash_good)
        _s("Clash bad", r.clash_bad)
        _s("Clash ugly", r.clash_ugly)

        total = sum(len(getattr(r, a)) for a in (
            "halogen","salt_bridges","arom_hbonds",
            "pipi","pi_cation","clash_good","clash_bad","clash_ugly"))
        if total == 0 and r.hbond_count == 0:
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
        for block in content.split("$$$$"):
            block = block.strip()
            if not block:
                continue
            props = {}
            lines = block.splitlines()
            if lines and lines[0].strip():
                props["_name"] = lines[0].strip()
            for m in re.finditer(r">\s*<([^>]+)>\s*\n([^\n]*)", block):
                key = m.group(1).strip()
                raw = m.group(2).strip()
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
                try:
                    if (cmd.count_atoms(f"{n} and ({ligands})") > 0 and
                            cmd.count_atoms(f"{n} and ({protein})") == 0 and
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
        if "," in ligands:
            ligs = [l.strip() for l in ligands.split(",")]
            ref_lig = None
        else:
            all_n = cmd.get_names("objects")
            multi_ligs, single_ligs = [], []
            for n in all_n:
                try:
                    if (cmd.count_atoms(f"{n} and ({ligands})") > 0 and
                            cmd.count_atoms(f"{n} and ({protein})") == 0):
                        (multi_ligs if cmd.count_states(n) > 1 else single_ligs).append(n)
                except CmdException:
                    pass
            # Multi-state objects are docking poses; single-state are ref-lig candidates
            if multi_ligs:
                ligs = multi_ligs
                ref_lig = single_ligs[0] if single_ligs else None
            else:
                ligs = single_ligs   # all single-state — treat all as poses
                ref_lig = None
            if not ligs:
                ligs = [ligands]
                ref_lig = None
        _stepper.setup_objects(protein, ligs, ref_lig=ref_lig)
        print(f"PoseViewer: {len(ligs)} ligand(s).")

    _print_summary()
    _bind_keys()


def _print_summary():
    """Print summary to console only when the GUI is not open."""
    if _gui_window is None:
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

def ci_clear():
    _clear_all(); _unbind_keys(); print("PoseViewer: cleared.")

def ci_gui():
    _open_gui()

cmd.extend("ci_setup", ci_setup)
cmd.extend("ci_next", ci_next)
cmd.extend("ci_prev", ci_prev)
cmd.extend("ci_goto", ci_goto)
cmd.extend("ci_update", ci_update)
cmd.extend("ci_refresh", ci_refresh)
cmd.extend("ci_load_scores", ci_load_scores)
cmd.extend("ci_clear", ci_clear)
cmd.extend("ci_gui", ci_gui)


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------

_gui_window = None

def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("PoseViewer", _open_gui)


def _open_gui():
    global _gui_window, _stepper

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
    e_prot = QtWidgets.QLineEdit("polymer.protein")
    l_s.addWidget(e_prot, 0, 1)
    l_s.addWidget(QtWidgets.QLabel("Ligand(s):"), 1, 0)
    e_lig = QtWidgets.QLineEdit("organic")
    l_s.addWidget(e_lig, 1, 1)

    hl_m = QtWidgets.QHBoxLayout()
    hl_m.addWidget(QtWidgets.QLabel("Mode:"))
    bg_m = QtWidgets.QButtonGroup(win)
    rb_m = {}
    for m in ("auto", "objects", "states"):
        rb = QtWidgets.QRadioButton(m)
        if m == "auto": rb.setChecked(True)
        bg_m.addButton(rb); rb_m[m] = rb; hl_m.addWidget(rb)
    hl_m.addStretch()
    l_s.addLayout(hl_m, 2, 0, 1, 2)

    l_s.addWidget(QtWidgets.QLabel("Scores (SDF):"), 3, 0)
    hl_sf = QtWidgets.QHBoxLayout()
    e_scores = QtWidgets.QLineEdit()
    e_scores.setPlaceholderText("optional")
    b_browse = QtWidgets.QPushButton("…"); b_browse.setFixedWidth(26)
    hl_sf.addWidget(e_scores); hl_sf.addWidget(b_browse)
    l_s.addLayout(hl_sf, 3, 1)

    hl_b = QtWidgets.QHBoxLayout()
    b_setup = QtWidgets.QPushButton("Setup")
    b_clear = QtWidgets.QPushButton("Clear")
    hl_b.addWidget(b_setup); hl_b.addWidget(b_clear)
    l_s.addLayout(hl_b, 4, 0, 1, 2)
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
    b_prev = QtWidgets.QPushButton("<  Prev")
    b_refresh = QtWidgets.QPushButton("Refresh")
    b_next = QtWidgets.QPushButton("Next  >")
    hl_pn.addWidget(b_prev); hl_pn.addWidget(b_refresh); hl_pn.addWidget(b_next)
    l_n.addLayout(hl_pn)

    hl_j = QtWidgets.QHBoxLayout()
    hl_j.addWidget(QtWidgets.QLabel("Go to #:"))
    sp = QtWidgets.QSpinBox(); sp.setMinimum(1); sp.setMaximum(99999)
    sp.setFixedWidth(70); hl_j.addWidget(sp)
    b_go = QtWidgets.QPushButton("Go"); b_go.setFixedWidth(50)
    hl_j.addWidget(b_go); hl_j.addStretch()
    l_n.addLayout(hl_j)
    cb_cmp_hb = QtWidgets.QCheckBox("H-bonds in compare mode")
    cb_cmp_hb.setChecked(False)
    l_n.addWidget(cb_cmp_hb)
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

    # Non-covalent bonds
    g1, g1_en, _g1b, l1 = _section("Non-covalent bonds", enabled=True, expanded=False)
    cb_hb = _cb(l1, "Hydrogen bonds",  "#ffd900")
    cb_xb = _cb(l1, "Halogen bonds",   "#9933e6")
    cb_sb = _cb(l1, "Salt bridges",    "#e633e6")
    cb_ah = _cb(l1, "Aromatic H-Bond", "#4dd97f")
    bot_l.addWidget(g1)

    # Pi interactions
    g2, g2_en, _g2b, l2 = _section("Pi interactions", enabled=True, expanded=False)
    cb_pp = _cb(l2, "Pi-pi stacking",  "#4dc0ff")
    cb_pc = _cb(l2, "Pi-cation",       "#33cc33")
    bot_l.addWidget(g2)

    # Contacts / clashes (disabled + collapsed by default)
    g3, g3_en, _g3b, l3 = _section("Contacts / Clashes", enabled=False, expanded=False)
    cb_cg = _cb(l3, "Good",  "#33cc33", checked=True)
    cb_cb = _cb(l3, "Bad",   "#ff9900", checked=True)
    cb_cu = _cb(l3, "Ugly",  "#ff2626", checked=True)
    bot_l.addWidget(g3)

    # Display options — collapsible + group enable (hides all display effects when off)
    g_disp, g_disp_en, _gdb, l_disp = _section("Display", enabled=True, expanded=False)
    cb_lb   = QtWidgets.QCheckBox("Show distance labels");  cb_lb.setChecked(True)
    cb_surf = QtWidgets.QCheckBox("Show surface");          cb_surf.setChecked(True)
    cb_rlbl = QtWidgets.QCheckBox("Show residue labels");   cb_rlbl.setChecked(True)
    cb_zoom = QtWidgets.QCheckBox("Auto-zoom to pose");     cb_zoom.setChecked(True)
    cb_lig_h = QtWidgets.QCheckBox("Show nonpolar H on ligands"); cb_lig_h.setChecked(False)
    for _w in (cb_lb, cb_surf, cb_rlbl, cb_zoom, cb_lig_h):
        l_disp.addWidget(_w)
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
        seen: dict = {}
        for p in all_props:
            for k in p:
                if k not in seen:
                    seen[k] = None
        cols = list(seen.keys())
        cols = (["_name"] if "_name" in cols else []) + [
            c for c in cols if c != "_name" and "rank" not in c.lower()]
        tw_pd.setColumnCount(len(cols))
        headers = ["Ligand_ID" if c == "_name" else c for c in cols]
        tw_pd.setHorizontalHeaderLabels(headers)
        tw_pd.setRowCount(len(all_props))
        for r, props in enumerate(all_props):
            for c, key in enumerate(cols):
                val = props.get(key, "")
                if isinstance(val, (int, float)):
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
                tw_pd.setItem(r, c, item)
        tw_pd.resizeColumnsToContents()
        tw_pd.setSortingEnabled(True)
        _sel_order.clear()
        _highlight_current_row()

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
        c = _stepper._count()
        if c > 0:
            if _stepper._in_compare:
                lbl.setText(_stepper.compare_label())
            else:
                lbl.setText(f"{_stepper._label()}  "
                            f"({_stepper.current_index + 1}/{c})")
            sp.setMaximum(c)
            sp.setValue(_stepper.current_index + 1)
        else:
            lbl.setText("Ready - click Setup")
        if g_pd.isChecked() and not _stepper._in_compare:
            _highlight_current_row()

    def do_browse():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            win, "Select scores SDF", "",
            "SDF files (*.sdf *.SDF);;All files (*)")
        if path:
            e_scores.setText(path)

    def _populate_ref_combo():
        """Refill ref_combo with current organic objects, preserving selection."""
        prev = ref_combo.currentText()
        ref_combo.blockSignals(True)
        ref_combo.clear()
        ref_combo.addItem("(none)")
        for n in cmd.get_names("objects"):
            if n.startswith("_ci_") or n in (
                    _OBJ_PTS, _OBJ_REF_PTS, _OBJ_SURF):
                continue
            try:
                if (cmd.count_atoms(f"{n} and organic") > 0 and
                        cmd.count_atoms(f"{n} and ({_stepper.protein_sel or 'polymer.protein'})") == 0):
                    ref_combo.addItem(n)
            except Exception:
                pass
        # Select auto-detected ref, then fall back to previous selection
        target = _stepper.ref_ligand or prev
        idx = ref_combo.findText(target) if target else -1
        ref_combo.setCurrentIndex(max(idx, 0))
        ref_combo.blockSignals(False)

    def do_setup():
        mode = next(m for m, rb in rb_m.items() if rb.isChecked())
        sf = e_scores.text().strip()
        if sf:
            ci_load_scores(sf)
        else:
            _stepper.sdf_records = []
        ci_setup(protein=e_prot.text(), ligands=e_lig.text(), mode=mode)
        _populate_ref_combo()
        update_ui()
        rebuild_table()

    def on_ref_changed(text):
        prev = _stepper.ref_ligand
        _stepper.ref_ligand = None if text == "(none)" else text
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

    def _table_adjacent(delta):
        """Navigate to the row delta steps from the current row in table order."""
        n_rows = tw_pd.rowCount()
        if n_rows == 0:
            # No table — fall back to poses order
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

    def _tog(cb, attr, group_en):
        def h(state):
            setattr(_stepper, attr, group_en.isChecked() and cb.isChecked())
            ci_update(); update_ui()
        return h

    def _group_tog(group_en, cb_attr_pairs):
        def h(checked):
            for cb, attr in cb_attr_pairs:
                setattr(_stepper, attr, checked and cb.isChecked())
            ci_update(); update_ui()
        return h

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


    cb_hb.stateChanged.connect(_tog(cb_hb, "show_hbonds",     g1_en))
    cb_xb.stateChanged.connect(_tog(cb_xb, "show_halogen",    g1_en))
    cb_sb.stateChanged.connect(_tog(cb_sb, "show_salt",       g1_en))
    cb_ah.stateChanged.connect(_tog(cb_ah, "show_arom_hb",    g1_en))
    cb_pp.stateChanged.connect(_tog(cb_pp, "show_pipi",       g2_en))
    cb_pc.stateChanged.connect(_tog(cb_pc, "show_pi_cation",  g2_en))
    cb_cg.stateChanged.connect(_tog(cb_cg, "show_clash_good", g3_en))
    cb_cb.stateChanged.connect(_tog(cb_cb, "show_clash_bad",  g3_en))
    cb_cu.stateChanged.connect(_tog(cb_cu, "show_clash_ugly", g3_en))
    cb_zoom.stateChanged.connect(lambda s: setattr(_stepper, "auto_zoom", cb_zoom.isChecked()))

    g1_en.stateChanged.connect(_group_tog(g1_en, [
        (cb_hb, "show_hbonds"), (cb_xb, "show_halogen"),
        (cb_sb, "show_salt"),   (cb_ah, "show_arom_hb"),
    ]))
    g2_en.stateChanged.connect(_group_tog(g2_en, [
        (cb_pp, "show_pipi"), (cb_pc, "show_pi_cation"),
    ]))
    g3_en.stateChanged.connect(_group_tog(g3_en, [
        (cb_cg, "show_clash_good"), (cb_cb, "show_clash_bad"), (cb_cu, "show_clash_ugly"),
    ]))

    def do_toggle_surf(state=None):
        if _OBJ_SURF in _created_objects:
            if g_disp_en.isChecked() and cb_surf.isChecked():
                cmd.show("surface", _OBJ_SURF)
            else:
                cmd.hide("surface", _OBJ_SURF)
    cb_surf.stateChanged.connect(do_toggle_surf)

    def do_toggle_rlbl(state=None):
        if _shell_sel is not None:
            sel = f"({_shell_sel}) and name CA"
            if g_disp_en.isChecked() and cb_rlbl.isChecked():
                cmd.show("labels", sel)
            else:
                cmd.hide("labels", sel)
    cb_rlbl.stateChanged.connect(do_toggle_rlbl)

    def do_disp_group_tog(checked):
        _stepper.show_labels = checked and cb_lb.isChecked()
        _stepper.show_lig_h  = checked and cb_lig_h.isChecked()
        do_toggle_surf()
        do_toggle_rlbl()
        ci_update(); update_ui()
    g_disp_en.stateChanged.connect(do_disp_group_tog)

    cb_lb.stateChanged.connect(lambda s: [
        setattr(_stepper, "show_labels", g_disp_en.isChecked() and cb_lb.isChecked()),
        ci_update(), update_ui()])

    cb_lig_h.stateChanged.connect(lambda s: [
        setattr(_stepper, "show_lig_h", g_disp_en.isChecked() and cb_lig_h.isChecked()),
        ci_update(), update_ui()])

    def on_cmp_hb(state):
        _stepper.show_cmp_hbonds = cb_cmp_hb.isChecked()
        if _stepper._in_compare:
            _stepper.show_comparison(_stepper._cmp_indices)
            update_ui()
    cb_cmp_hb.stateChanged.connect(on_cmp_hb)

    tw_pd.itemSelectionChanged.connect(on_selection_changed)
    g_pd.toggled.connect(lambda checked: update_ui())

    for key, fn in [(QtCore.Qt.Key_Right, do_next),
                    (QtCore.Qt.Key_Left, do_prev)]:
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), win)
        sc.activated.connect(fn)

    win.show(); win.raise_()


def _set_gui_none():
    global _gui_window
    _gui_window = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

print("PoseViewer loaded.")
print("  ci_gui     - open GUI panel")
print("  ci_setup   - setup from command line")
print("  ci_refresh     - sync to current PyMOL state / state slider")
print("  ci_load_scores - load per-pose properties from SDF file")
print("  LEFT/RIGHT arrow keys after setup")
