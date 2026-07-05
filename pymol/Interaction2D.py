"""
Interaction2D - PyMOL Plugin
============================
Generates classic 2D protein-ligand interaction diagrams (LigPlot/PoseView
style): the ligand is drawn as a real 2D chemical structure (via RDKit),
with contacting residues arranged as labeled nodes around it and connected
by lines for H-bonds, pi-stacking and salt bridges. Hydrophobic contacts
are shown by coloring the residue node only, no line — matching Maestro,
and sidestepping the fact that there are usually too many, too densely
packed, for a line to ever be drawn without crossing the ligand.

Features
--------
- RDKit-based 2D depiction of the ligand (proper bonds/rings, not a
  schematic network graph)
- Automatic detection of H-bonds, hydrophobic contacts, pi-stacking
  (face-to-face & edge-to-face) and salt bridges
- Rounded-rectangle residue nodes colored by residue type, and interaction
  line colors, matched to a Schrodinger Maestro Ligand Interaction Diagram
  (hydrophobic green / polar blue / charged orange & purple; H-bond
  magenta; pi-stacking green with end markers; salt bridge mauve)
- Residue-node placement minimizes H-bond/pi-stacking/salt-bridge lines
  crossing the ligand's own drawing
- Solvent-exposure halos on ligand atoms: real relative-SASA calculation
  (bound-complex vs. isolated-ligand), not a heuristic
- Distance labels on H-bonds and salt bridges
- Live preview in the plugin window; export to PNG/SVG

Requires RDKit and matplotlib in the same Python environment PyMOL is
running in.

Installation
------------
  Plugin > Plugin Manager > Install New Plugin > choose this file
  — or —
  run /path/to/Interaction2D.py   then   i2d_gui

CLI commands
------------
  i2d_list_ligands selection=organic, state=1     list candidate ligand residues
  i2d_generate protein=polymer.protein, ligand=organic, ligand_index=None,
               state=1, filename="", hbond=1, hydrophobic=1, pipi=1, salt=1,
               solvent=1
  i2d_gui                          open the GUI

Authors: Evert J. Homan, PhD; Claude (Anthropic)
Date:    2026-07-04
Version: 1.0
License: MIT
"""

from __future__ import annotations

import io
import math
import os
import shutil
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.colors import to_rgb

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:
    Chem = None
    AllChem = None
    rdMolDraw2D = None

from pymol import cmd

# ---------------------------------------------------------------------------
# Thresholds / chemistry tables
# ---------------------------------------------------------------------------

HBOND_ELEMENTS = {"N", "O", "S", "F"}
HBOND_DIST_MAX = 3.5

SALT_DIST_MAX = 4.0
CHARGED_BASIC = {"ARG", "LYS", "HIS", "HID", "HIE", "HIP"}
BASIC_ATOMS = {"NH1", "NH2", "NE", "NZ", "ND1", "NE2"}
CHARGED_ACIDIC = {"ASP", "GLU"}
ACIDIC_ATOMS = {"OD1", "OD2", "OE1", "OE2"}

HYDROPHOBIC_DIST_MAX = 4.0
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

PIPI_FTF_DIST_MAX = 4.8
PIPI_FTF_ANGLE_MAX = 40.0
PIPI_ETF_DIST_MAX = 5.5
PIPI_ETF_ANGLE_MIN = 45.0
PIPI_ETF_ANGLE_MAX = 90.0

# Aromatic side-chain ring atoms used for protein-side pi-stacking.
# HIS uses the 5-membered imidazole ring; TRP uses the 6-membered benzo ring.
RING_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HID": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIE": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIP": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

# Colors sampled directly from a Schrodinger Maestro Ligand Interaction
# Diagram legend, so this plugin's output reads like a familiar LID at a
# glance: H-bond magenta, pi-stacking green (with circular end markers),
# salt bridge muted mauve. Hydrophobic contacts have a style entry (for the
# residue-type legend / potential future use) but no line is ever drawn for
# them — see the render loop, which skips "hydrophobic" items entirely,
# matching Maestro (too many, too dense, to route without crossing).
STYLE = {
    "hbond":       dict(color="#CC33FF", linestyle="--", linewidth=1.6, label="H-bond"),
    "salt":        dict(color="#B98CC2", linestyle="--", linewidth=1.6, label="Salt bridge"),
    "pipi":        dict(color="#22A022", linestyle="-",  linewidth=1.8, label="Pi-stacking"),
    "hydrophobic": dict(color="#808080", linestyle=":",  linewidth=1.4, label="Hydrophobic"),
}
KIND_ORDER = ["hbond", "salt", "pipi", "hydrophobic"]

# Gap (px) an H-bond line stops short of whichever ligand atom it touches
# (arrowhead end or plain tail end), so it doesn't overlap the atom
# label/bonds — matches Maestro's small visible gap there too (see
# 3FCI_lid.png). 10px looked right measured against the raw data but was
# visually swallowed by the atom label's own glyph stroke width and (at
# the tail end) the dash pattern's own on/off phase — bumped until the gap
# reads as an unambiguous white margin, not just a number that's technically
# nonzero.
ARROW_LIGAND_GAP_PX = 22

# When a residue node lands close to the ligand atom it's H-bonded to (a
# short interaction distance doesn't imply the *label* is placed far away
# — it can end up right next door), the box-edge pull-back and the
# ligand-atom gap above can together consume the *entire* line, leaving a
# barely-there sliver or, in the worst case, a fully degenerate
# zero-length arrow (both ends collapse to the same point). If the two
# pull-backs combined would eat more than this much of the real distance,
# both are scaled down proportionally so at least this many px of visible
# dashed line always remains.
MIN_VISIBLE_ARROW_PX = 20.0

# Residue node fill colors, also sampled from Maestro's legend — categorized
# by residue type (not just charge), matching what Maestro's diagrams show.
HYDROPHOBIC_RESN = {"PHE", "TRP", "TYR", "VAL", "LEU", "ILE", "MET", "ALA", "PRO"}
POLAR_RESN = {"SER", "THR", "ASN", "GLN", "CYS", "HIS", "HID", "HIE"}
CHARGED_NEG_RESN = {"ASP", "GLU"}
CHARGED_POS_RESN = {"ARG", "LYS", "HIP"}
GLYCINE_RESN = {"GLY"}

RESIDUE_CATEGORY_COLORS = {
    "hydrophobic": "#CCE57C",
    "polar":       "#8FDCF7",
    "charged_neg": "#F0A67B",
    "charged_pos": "#A2A2FE",
    "glycine":     "#F0F1D7",
    "other":       "#B6B6B6",
}
RESIDUE_CATEGORY_ORDER = ["hydrophobic", "polar", "charged_neg", "charged_pos", "glycine", "other"]
RESIDUE_CATEGORY_LABELS = {
    "hydrophobic": "Hydrophobic",
    "polar": "Polar",
    "charged_neg": "Charged (negative)",
    "charged_pos": "Charged (positive)",
    "glycine": "Glycine",
    "other": "Other",
}

NODE_HALF_W = 46
NODE_HALF_H = 13
NODE_RADIAL_GAP = 60
CANVAS_SIZE = 520

# A node only relocates away from its natural (anchor-atom) angle if its
# connector line's crossing cost (see _line_crossing_cost, in px of the
# line overlapping drawn content) at that natural angle exceeds this —
# a couple of px is just a grazing touch near a bond, not a real crossing,
# and isn't worth reshuffling the whole layout over. Requires the best
# alternative to actually clear that crossing by a real margin too, so a
# node never jumps far for a marginal, barely-visible improvement.
NODE_LINE_CROSS_MIN_PX = 5.0
NODE_LINE_CROSS_MIN_IMPROVEMENT_PX = 3.0

# Added to a candidate angle's cost per *other* hbond/salt/pipi line it
# would cross (checked against lines already placed earlier in the same
# pass — see _place_residue_nodes). Large relative to typical ligand-content
# crossing costs (a few px to a few tens of px) so the search strongly
# prefers a line-free angle, but isn't infinite: if every candidate angle
# crosses something, the least-bad option still wins rather than nothing
# being placed.
LINE_CROSS_PENALTY_PX = 40.0

# Relative SASA (bound-complex SASA / isolated-ligand SASA) at or above this
# is drawn as a solvent-exposure halo, matching Maestro's grey-circle marker.
SOLVENT_REL_SASA_THRESHOLD = 0.15
# Ring atoms hindered by their own substituents (e.g. a pyrimidine carbon
# flanked by two ring N's) can have a tiny isolated-ligand SASA baseline, so
# a few Ų^2 of bound-complex SASA divides out to a misleadingly high ratio
# even though the absolute exposure is negligible. Require this minimum
# absolute bound-complex SASA (Ų^2) too, so the ratio can't be triggered by
# noise near zero on both sides of the division. Kept well above the ~2 Ų^2
# scale of that instability while still admitting genuinely-if-modestly
# exposed atoms as small circles (Maestro shows these too).
SOLVENT_MIN_ABS_SASA = 3.0
SOLVENT_EXPOSURE_COLOR = "lightgrey"
# Maestro encodes exposure *amount* as halo size, not shading intensity —
# every circle is drawn flat/opaque (no alpha blending into a haze), and the
# radius scales with the atom's absolute bound-complex SASA (Ų^2), not the
# relative ratio (the ratio can be misleadingly large for atoms with a tiny
# isolated-ligand baseline, see _compute_relative_sasa). Scaled per-diagram
# against the min/max absolute SASA among *this molecule's* exposed atoms,
# not a fixed Ų^2 scale, so the full radius range is used even when they
# all happen to sit in a narrow absolute band.
SOLVENT_HALO_RADIUS_MIN = 4
SOLVENT_HALO_RADIUS_MAX = 26
# Soft-aura look (matches Maestro): each disc is flat out to its radius,
# then ramps down to 0 alpha over this many extra pixels — a manual blur
# that avoids a real Gaussian-filter dependency. Discs are combined via
# elementwise max (see _solvent_glow_layer), not stacked, so the peak
# opacity below is the *only* alpha any pixel ever reaches, however many
# circles overlap there.
SOLVENT_GLOW_BLUR_WIDTH = 10
SOLVENT_GLOW_MAX_ALPHA = 0.75


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def _dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _centroid(pts) -> Tuple[float, float, float]:
    return tuple(np.mean(np.array(pts), axis=0))


def _norm(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _dot(a, b) -> float:
    return float(np.dot(a, b))


def _angle_normals(n1, n2) -> float:
    c = abs(_dot(_norm(n1), _norm(n2)))
    return math.degrees(math.acos(min(1.0, max(0.0, c))))


def _newell_normal(coords):
    """Ring normal via Newell's method (robust for non-perfectly-planar rings)."""
    n = len(coords)
    nm = [0.0, 0.0, 0.0]
    for i in range(n):
        j = (i + 1) % n
        ci, cj = coords[i], coords[j]
        nm[0] += (ci[1] - cj[1]) * (ci[2] + cj[2])
        nm[1] += (ci[2] - cj[2]) * (ci[0] + cj[0])
        nm[2] += (ci[0] - cj[0]) * (ci[1] + cj[1])
    return _norm(nm)


def _nearest_neighbor_pos(idx: int, coords: List[Tuple[str, tuple, str]]) -> Optional[tuple]:
    """Nearest other *heavy* atom in the same residue, used as a
    covalent-neighbor proxy when no explicit bond graph is available
    (protein atoms). Hydrogens are excluded even when present (some source
    structures are pre-protonated, e.g. via a PDB2PQR/Reduce-style prep
    pipeline) — an atom's own H is almost always its closest neighbor
    (~1.0 Å vs ~1.5 Å+ to a real heavy substituent), which would otherwise
    make this proxy point "away from my own H" instead of "away from my
    heavy substituents" — inverting the acceptor-direction check it feeds
    (`_outward_ok`) for exactly the atoms that have one, and silently
    rejecting genuine H-bonds those atoms donate through their own H
    (caught via a real, very short N-H...O contact — GLN144 backbone —
    that should have been an obvious H-bond but wasn't detected)."""
    pos = coords[idx][1]
    best, best_d = None, None
    for j, (_name, c, el) in enumerate(coords):
        if j == idx or el == "H":
            continue
        d = _dist(pos, c)
        if best_d is None or d < best_d:
            best_d, best = d, c
    return best


def _outward_ok(atom_pos: tuple, neighbor_pos: Optional[tuple], partner_pos: tuple) -> bool:
    """True if partner_pos lies roughly away from atom_pos's own covalent
    neighbor(s) (i.e. in the lone-pair hemisphere), rather than behind the
    atom relative to its substituents — a coarse, hydrogen-free proxy for
    *acceptor* directionality (no explicit H to check a real donor angle
    against). Used for protein atoms, and for ligand atoms with no explicit H."""
    if neighbor_pos is None:
        return True
    v_away = _norm([atom_pos[i] - neighbor_pos[i] for i in range(3)])
    v_to = _norm([partner_pos[i] - atom_pos[i] for i in range(3)])
    return _dot(v_away, v_to) > 0.0


def _donor_angle_ok(x_pos: tuple, h_positions: List[tuple], partner_pos: tuple,
                     angle_min: float = 100.0) -> bool:
    """True if at least one explicit H on atom x_pos points roughly at
    partner_pos (angle at the H, X-H...partner >= angle_min)."""
    for h_pos in h_positions:
        v1 = _norm([x_pos[i] - h_pos[i] for i in range(3)])
        v2 = _norm([partner_pos[i] - h_pos[i] for i in range(3)])
        ang = math.degrees(math.acos(min(1.0, max(-1.0, _dot(v1, v2)))))
        if ang >= angle_min:
            return True
    return False


def _ligand_atom_geometry_ok(x_pos: tuple, h_positions: List[tuple],
                              heavy_centroid: Optional[tuple], partner_pos: tuple) -> bool:
    """Ligand atoms have explicit H (from the mol file), so donor geometry
    can be checked directly via the real D-H...A angle rather than the
    heavy-atom proxy — the proxy alone would wrongly reject genuine donor
    H-bonds (e.g. an -OH oxygen's 'away from substituents' direction points
    into its lone pairs, not along the O-H bond that's actually donating).
    Passes if EITHER a real donor angle is plausible OR (as an acceptor,
    using only heavy neighbors) the partner isn't blocked by substituents."""
    if h_positions and _donor_angle_ok(x_pos, h_positions, partner_pos):
        return True
    if heavy_centroid is not None:
        return _outward_ok(x_pos, heavy_centroid, partner_pos)
    return True


def _ligand_hbond_role(atom, x_pos: tuple, h_positions: List[tuple], partner_pos: tuple) -> str:
    """"donor" or "acceptor" for the *ligand* atom in one H-bond, used to
    draw the arrowhead (conventionally donor -> acceptor). An atom can be
    geometrically plausible as both (e.g. an -OH oxygen), so this only
    decides which role to *draw* for this specific contact.

    Real donor-H geometry (an explicit H actually pointing at the partner)
    is used when available, since it's the most direct signal. But most PDB
    ligands have *no* resolved (or added) hydrogens at all — X-ray doesn't
    resolve H, and nothing in this pipeline adds them unless the source mol
    already had explicit H (ivermectin's PDB entry happens to; most don't,
    e.g. 2GE). Without that, falls back to RDKit's implicit valence via
    `GetTotalNumHs()`: an atom with no H at all (ether O, ester/carbonyl O,
    tertiary amine N, aromatic ring N) can only be an acceptor; an atom
    with at least one H is *most commonly* depicted as the donor in these
    diagrams (this is the same convention tools like LigPlot use, since
    they're in the same no-resolved-H boat). This fallback previously
    didn't exist, so every H-bond on a no-explicit-H ligand silently drew
    as "ligand accepts" regardless of real chemistry — caught by checking
    2GE's amine-to-Asp contacts, which should clearly be ligand-donor."""
    if h_positions and _donor_angle_ok(x_pos, h_positions, partner_pos):
        return "donor"
    return "donor" if atom.GetTotalNumHs() > 0 else "acceptor"


def _residue_category(resn: str) -> str:
    if resn in CHARGED_NEG_RESN:
        return "charged_neg"
    if resn in CHARGED_POS_RESN:
        return "charged_pos"
    if resn in GLYCINE_RESN:
        return "glycine"
    if resn in HYDROPHOBIC_RESN:
        return "hydrophobic"
    if resn in POLAR_RESN:
        return "polar"
    return "other"


def _residue_color(resn: str) -> str:
    return RESIDUE_CATEGORY_COLORS[_residue_category(resn)]


# ---------------------------------------------------------------------------
# Ligand mol construction (PyMOL selection -> RDKit mol with 2D coords)
# ---------------------------------------------------------------------------

def _fix_carboxylate_double_double(mol_h) -> bool:
    """Repair one known PyMOL mol-export artifact: a carbon whose bonds to
    two oxygens both got guessed as double (correct pattern is one
    single/one double) — seen when a carboxylate's two C-O distances in
    the source structure are close enough to be ambiguous. Demotes the
    bond to whichever of the two oxygens already carries a negative
    formal charge, since that's the one a real carboxylate resonance
    structure would draw single-bonded. Returns True if anything changed
    (caller should retry sanitization); mutates mol_h in place."""
    fixed = False
    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        o_double = [b for b in atom.GetBonds()
                    if b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtom(atom).GetSymbol() == "O"]
        if len(o_double) != 2:
            continue
        negative = [b for b in o_double if b.GetOtherAtom(atom).GetFormalCharge() < 0]
        if len(negative) == 1:
            negative[0].SetBondType(Chem.BondType.SINGLE)
            fixed = True
    return fixed


def _build_ligand(ligand_sel: str, state: int):
    """Return (lig_model, mol_h, mol_2d, heavy_of).

    lig_model: chempy Model from cmd.get_model(ligand_sel, state=state)
    mol_h:     RDKit mol with all atoms (incl. H) and the same 3D coords/
               atom order as lig_model (verified: cmd.save(format='mol')
               followed by Chem.MolFromMolFile preserves PyMOL atom order).
    mol_2d:    heavy-atom-only mol with a computed 2D layout.
    heavy_of:  maps a full-atom index (== index into lig_model.atom / mol_h)
               to its index in mol_2d. H atoms are absent from this map.
    """
    if Chem is None:
        raise RuntimeError(
            "RDKit is not available in this Python environment. Install it "
            "(e.g. `conda install -c conda-forge rdkit`) in the environment "
            "PyMOL is running in, then restart PyMOL.")

    lig_model = cmd.get_model(ligand_sel, state=state)
    if not lig_model.atom:
        raise ValueError(f"Ligand selection '{ligand_sel}' is empty (state {state}).")

    keys = {(a.chain, a.resn, a.resi) for a in lig_model.atom}
    if len(keys) > 1:
        found = ", ".join(f"{c}/{n}{i}" for c, n, i in sorted(keys))
        raise ValueError(
            f"Ligand selection matches {len(keys)} distinct residues ({found}); "
            "narrow the selection (e.g. 'organic and resi 403'), pick one with "
            "i2d_generate(..., ligand_index=N) after i2d_list_ligands(...), or "
            "use the Ligand dropdown in the GUI.")

    tmpdir = tempfile.mkdtemp(prefix="i2d_")
    try:
        molpath = os.path.join(tmpdir, "lig.mol")
        cmd.save(molpath, ligand_sel, state=state, format="mol")
        mol_h = Chem.MolFromMolFile(molpath, sanitize=False, removeHs=False)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if mol_h is not None:
        try:
            Chem.SanitizeMol(mol_h)
        except Exception:
            # PyMOL's bond-order guessing can double-count a carboxylate
            # when the crystal's two C-O distances are close enough to be
            # ambiguous, assigning *both* C-O bonds as double (valence 5 on
            # that carbon) instead of one single/one double. Recognizable
            # and fixable: the correctly-placed formal charge (-1) still
            # picks out which oxygen should have been the single-bonded
            # one. Only retried once, and only for this specific pattern —
            # any other sanitization failure still surfaces as a clear error.
            if _fix_carboxylate_double_double(mol_h):
                try:
                    Chem.SanitizeMol(mol_h)
                except Exception:
                    mol_h = None
            else:
                mol_h = None

    if mol_h is None:
        raise ValueError(
            "RDKit could not parse the ligand geometry (bond perception or "
            "sanitization failed). Check for missing atoms or bad valences.")
    if mol_h.GetNumAtoms() != len(lig_model.atom):
        raise ValueError(
            "Atom count mismatch between the PyMOL selection and the RDKit "
            "molecule; cannot reliably map interactions onto the 2D depiction.")

    # Parsing 3D coordinates can tag an atom as a chiral center from noise
    # in its real (crystal/MM) geometry even when it isn't one — e.g. a
    # plain -CH2- can never be a real stereocenter (its two H's are
    # identical), but slightly asymmetric 3D placement of those H's can
    # still get it a CHI_TETRAHEDRAL_* flag. Left alone, that flag draws a
    # spurious wedge/hash bond on an achiral atom. `cleanIt=True` clears
    # any chiral tag that isn't a genuine stereocenter (verified on
    # ivermectin, which has many *real* ones: 40 raw tags, 21 survive
    # cleanIt — the other 19 were exactly this artifact, on ring CH2's).
    Chem.AssignStereochemistry(mol_h, cleanIt=True, force=True)

    heavy_of: Dict[int, int] = {}
    heavy_i = 0
    for i, a in enumerate(mol_h.GetAtoms()):
        if a.GetAtomicNum() != 1:
            heavy_of[i] = heavy_i
            heavy_i += 1

    mol_2d = Chem.RemoveHs(mol_h)
    try:
        from rdkit.Chem import rdCoordGen
        rdCoordGen.AddCoords(mol_2d)
    except Exception:
        AllChem.Compute2DCoords(mol_2d)

    return lig_model, mol_h, mol_2d, heavy_of


def _compute_relative_sasa(lig_model, ligand_sel: str, protein_sel: str,
                            state: int) -> Tuple[List[float], List[float]]:
    """Per-ligand-atom (relative SASA, absolute bound-complex SASA in Ų^2).
    Relative = bound-complex SASA / isolated-ligand-alone SASA (0 = fully
    buried, 1 = as exposed as free in solvent) — used to decide *whether*
    an atom counts as exposed. The absolute value is returned alongside
    because the ratio is a poor proxy for *how much* an atom is exposed: a
    ligand atom hindered by its own folded-back conformation (e.g. a
    macrocycle's own terminal methyl tucked against the ring) can have a
    tiny isolated-ligand baseline, so a few Ų^2 of real exposure divides out
    to a ratio near 1.0 — larger than an atom with genuinely more absolute
    exposed area but a bigger (less self-hindered) baseline. Order matches
    lig_model.atom.

    Two gotchas this works around:
    - Hetero/ligand atoms default to PyMOL's `ignore` flag for area/surface
      calculations (get_area silently returns 0 for them otherwise) — cleared
      here for the atoms involved.
    - get_area computed on a *selection* still lets other atoms in the same
      object occlude it — computing the true "isolated" SASA requires an
      actually separate object (`cmd.create`), not just a narrower selection
      string within the original object(s).
    """
    cmd.set("dot_solvent", 1)
    cmd.set("dot_density", 3)

    combined_sel = f"({ligand_sel}) or ({protein_sel})"
    cmd.flag("ignore", combined_sel, "clear")
    cmd.get_area(combined_sel, state=state, load_b=1)
    complex_sasa = [a.b for a in cmd.get_model(ligand_sel, state=state).atom]

    tmp = "_i2d_sasa_alone"
    cmd.create(tmp, ligand_sel, state, 1)
    cmd.flag("ignore", tmp, "clear")
    cmd.get_area(tmp, state=1, load_b=1)
    alone_sasa = [a.b for a in cmd.get_model(tmp, state=1).atom]
    cmd.delete(tmp)

    if len(complex_sasa) != len(lig_model.atom) or len(alone_sasa) != len(lig_model.atom):
        return [0.0] * len(lig_model.atom), [0.0] * len(lig_model.atom)

    rel = [c / a if a > 1e-6 and c >= SOLVENT_MIN_ABS_SASA else 0.0
           for c, a in zip(complex_sasa, alone_sasa)]
    return rel, complex_sasa


def _pixel_anchor(li: int, heavy_of: Dict[int, int], mol_h, drawer) -> Optional[Tuple[float, float]]:
    """Pixel position (as rendered) for full-atom index li. H atoms are
    redirected to their heavy neighbor since only heavy atoms are drawn."""
    if li in heavy_of:
        p = drawer.GetDrawCoords(heavy_of[li])
        return (p.x, p.y)
    atom = mol_h.GetAtomWithIdx(li)
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() in heavy_of:
            p = drawer.GetDrawCoords(heavy_of[nbr.GetIdx()])
            return (p.x, p.y)
    return None


def _pixel_anchor_for_item(it: dict, heavy_of: Dict[int, int], mol_h, drawer) -> Optional[Tuple[float, float]]:
    """Pixel anchor for one interaction. Pi-stacking anchors on the ligand
    ring's own 2D centroid (mean of its drawn ring-atom positions) rather
    than a single ring atom, so the line visibly emanates from the ring."""
    if "ring_atoms" in it:
        pts = [drawer.GetDrawCoords(heavy_of[i]) for i in it["ring_atoms"] if i in heavy_of]
        if not pts:
            return None
        return (sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))
    return _pixel_anchor(it["li"], heavy_of, mol_h, drawer)


# ---------------------------------------------------------------------------
# Ligand candidate listing (for structures with multiple ligands/copies)
# ---------------------------------------------------------------------------

def _list_ligand_candidates(ligand_scope: str, state: int) -> List[dict]:
    """Distinct (chain, resn, resi) residues found within ligand_scope."""
    model = cmd.get_model(ligand_scope, state=state)
    groups: Dict[tuple, int] = defaultdict(int)
    for a in model.atom:
        groups[(a.chain, a.resn, a.resi)] += 1

    def sort_key(item):
        (chain, resn, resi), _ = item
        digits = "".join(ch for ch in resi if ch.isdigit())
        return (chain, int(digits) if digits else 0, resn)

    out = []
    for (chain, resn, resi), n in sorted(groups.items(), key=sort_key):
        out.append(dict(chain=chain, resn=resn, resi=resi, natoms=n))
    return out


def _ligand_residue_selection(chain: str, resn: str, resi: str) -> str:
    parts = [f"resn {resn}", f"resi {resi}"]
    parts.append(f'chain "{chain}"' if chain else 'chain ""')
    return " and ".join(parts)


# ---------------------------------------------------------------------------
# Interaction detection
# ---------------------------------------------------------------------------

def _detect_pairwise(lig_model, mol_h, ligand_sel: str, protein_sel: str, state: int,
                      enabled: Set[str]) -> List[dict]:
    result: List[dict] = []
    search = max(HBOND_DIST_MAX, SALT_DIST_MAX, HYDROPHOBIC_DIST_MAX) + 2.0
    tmp = "_i2d_tmp_p"
    # byres so every same-residue atom is available for the H-bond geometry
    # check below (nearest-other-atom-in-residue as a covalent-neighbor proxy),
    # not just the atoms that individually fall within the distance cutoff.
    cmd.select(tmp, f"byres (({protein_sel}) within {search} of ({ligand_sel}))", state=state)
    prot_model = cmd.get_model(tmp, state=1)
    cmd.delete(tmp)
    if not prot_model.atom:
        return result

    def elem(a):
        return a.symbol.strip().capitalize()

    res_atoms: Dict[tuple, List[Tuple[str, tuple, str]]] = defaultdict(list)
    atom_res_idx = []
    for pa in prot_model.atom:
        key = (pa.chain, pa.resn, pa.resi)
        atom_res_idx.append((key, len(res_atoms[key])))
        res_atoms[key].append((pa.name, tuple(pa.coord), elem(pa)))

    conf = mol_h.GetConformer()
    hydrophobic_best: Dict[tuple, dict] = {}

    for li, la in enumerate(lig_model.atom):
        lc = tuple(la.coord)
        le = elem(la)
        lig_atom = mol_h.GetAtomWithIdx(li)
        lig_nbrs = lig_atom.GetNeighbors()
        lig_h_pos = [tuple(conf.GetAtomPosition(n.GetIdx())) for n in lig_nbrs if n.GetAtomicNum() == 1]
        lig_heavy_pos = [tuple(conf.GetAtomPosition(n.GetIdx())) for n in lig_nbrs if n.GetAtomicNum() != 1]
        lig_heavy_centroid = _centroid(lig_heavy_pos) if lig_heavy_pos else None

        for pi, pa in enumerate(prot_model.atom):
            pc = tuple(pa.coord)
            pe = elem(pa)
            d = _dist(lc, pc)

            if ("hbond" in enabled and d <= HBOND_DIST_MAX
                    and le in HBOND_ELEMENTS and pe in HBOND_ELEMENTS):
                res_key, idx_in_res = atom_res_idx[pi]
                prot_nbr = _nearest_neighbor_pos(idx_in_res, res_atoms[res_key])
                if (_ligand_atom_geometry_ok(lc, lig_h_pos, lig_heavy_centroid, pc)
                        and _outward_ok(pc, prot_nbr, lc)):
                    result.append(dict(kind="hbond", li=li, p2=pc, dist=d,
                                        ligand_role=_ligand_hbond_role(lig_atom, lc, lig_h_pos, pc),
                                        chain=pa.chain, resn=pa.resn, resi=pa.resi))

            if "salt" in enabled and d <= SALT_DIST_MAX:
                lpos = le == "N" and la.formal_charge > 0
                lneg = le == "O" and la.formal_charge < 0
                ppos = pa.resn in CHARGED_BASIC and pa.name in BASIC_ATOMS
                pneg = pa.resn in CHARGED_ACIDIC and pa.name in ACIDIC_ATOMS
                if (lpos and pneg) or (lneg and ppos):
                    result.append(dict(kind="salt", li=li, p2=pc, dist=d,
                                        chain=pa.chain, resn=pa.resn, resi=pa.resi))

            if ("hydrophobic" in enabled and le == "C" and pe == "C"
                    and pa.name not in BACKBONE_ATOMS and d <= HYDROPHOBIC_DIST_MAX):
                key = (pa.chain, pa.resn, pa.resi)
                cur = hydrophobic_best.get(key)
                if cur is None or d < cur["dist"]:
                    hydrophobic_best[key] = dict(kind="hydrophobic", li=li, p2=pc, dist=d,
                                                  chain=pa.chain, resn=pa.resn, resi=pa.resi)

    result.extend(hydrophobic_best.values())

    seen = set()
    deduped = []
    for it in result:
        key = (it["kind"], it["li"], it["chain"], it["resn"], it["resi"],
               tuple(round(x, 1) for x in it["p2"]))
        if key not in seen:
            seen.add(key)
            deduped.append(it)
    return deduped


def _ligand_aromatic_rings(mol_h, mol_2d, heavy_of):
    """Returns (full_atom_indices, 3D centroid, 3D normal) per aromatic ring.
    Keeping all ring atom indices (not just one) lets the pixel anchor be the
    ring's own 2D centroid rather than a single representative atom."""
    full_of_heavy = {v: k for k, v in heavy_of.items()}
    conf = mol_h.GetConformer()
    rings = []
    for ring in mol_2d.GetRingInfo().AtomRings():
        if not all(mol_2d.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        full_idxs = [full_of_heavy[i] for i in ring]
        coords = [tuple(conf.GetAtomPosition(i)) for i in full_idxs]
        rings.append((full_idxs, _centroid(coords), _newell_normal(coords)))
    return rings


def _protein_aromatic_rings(ligand_sel: str, protein_sel: str, state: int):
    search = PIPI_ETF_DIST_MAX + 3.0
    resn_list = "+".join(RING_ATOMS.keys())
    tmp = "_i2d_tmp_r"
    cmd.select(tmp, f"(({protein_sel}) within {search} of ({ligand_sel})) and resn {resn_list}",
               state=state)
    model = cmd.get_model(tmp, state=1)
    cmd.delete(tmp)

    groups: Dict[tuple, dict] = defaultdict(dict)
    for a in model.atom:
        groups[(a.chain, a.resn, a.resi)][a.name] = tuple(a.coord)

    rings = []
    for (chain, resn, resi), atoms in groups.items():
        names = RING_ATOMS.get(resn)
        if not names or not all(n in atoms for n in names):
            continue
        coords = [atoms[n] for n in names]
        rings.append((chain, resn, resi, _centroid(coords), _newell_normal(coords)))
    return rings


def _detect_pipi(mol_h, mol_2d, heavy_of, ligand_sel: str, protein_sel: str, state: int) -> List[dict]:
    lig_rings = _ligand_aromatic_rings(mol_h, mol_2d, heavy_of)
    if not lig_rings:
        return []
    prot_rings = _protein_aromatic_rings(ligand_sel, protein_sel, state)

    out = []
    for ring_atoms, lcen, lnorm in lig_rings:
        for chain, resn, resi, pcen, pnorm in prot_rings:
            d = _dist(lcen, pcen)
            ang = _angle_normals(lnorm, pnorm)
            if d <= PIPI_FTF_DIST_MAX and ang <= PIPI_FTF_ANGLE_MAX:
                pass
            elif d <= PIPI_ETF_DIST_MAX and PIPI_ETF_ANGLE_MIN <= ang <= PIPI_ETF_ANGLE_MAX:
                pass
            else:
                continue
            out.append(dict(kind="pipi", ring_atoms=ring_atoms, p2=pcen, dist=d,
                             chain=chain, resn=resn, resi=resi))
    return out


def detect_all(lig_model, mol_h, mol_2d, heavy_of, ligand_sel: str, protein_sel: str,
                state: int, enabled: Set[str]) -> List[dict]:
    items = _detect_pairwise(lig_model, mol_h, ligand_sel, protein_sel, state, enabled)
    if "pipi" in enabled:
        items += _detect_pipi(mol_h, mol_2d, heavy_of, ligand_sel, protein_sel, state)
    return items


# ---------------------------------------------------------------------------
# Residue node layout
# ---------------------------------------------------------------------------

def _pull_back_point(p_from: Tuple[float, float], p_to: Tuple[float, float],
                      amount: float) -> Tuple[float, float]:
    """Point on the segment p_from->p_to, `amount` px short of p_to."""
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    dist = math.hypot(dx, dy)
    if dist <= amount:
        return p_from
    f = (dist - amount) / dist
    return (p_from[0] + dx * f, p_from[1] + dy * f)


def _box_edge_pullback_amount(p_from: Tuple[float, float], box_center: Tuple[float, float],
                               half_w: float, half_h: float, margin: float) -> float:
    """Distance from box_center, along the line from p_from, at which the
    segment crosses the axis-aligned box's edge (half_w/half_h) plus
    `margin` — rather than a fixed distance from the center. A fixed
    pull-back calibrated for a horizontal approach (sized off half_w) way
    overshoots a steeply angled line, since the box is much shorter
    vertically (half_h) than it is wide."""
    dx, dy = box_center[0] - p_from[0], box_center[1] - p_from[1]
    t_edge = min(half_w / abs(dx) if dx else float("inf"),
                 half_h / abs(dy) if dy else float("inf"))
    dist = math.hypot(dx, dy)
    return dist * min(t_edge, 1.0) + margin




def _content_mask(img) -> np.ndarray:
    """Boolean (h, w) mask of actually-rendered content (bonds, atom labels,
    wedges) vs. background, shared by the radial-profile calc below and by
    the line-crossing check used to keep residue-node connector lines off
    the molecule's own drawing."""
    return (img[..., :3] < 0.95).any(axis=-1)


def _line_crossing_cost(mask: np.ndarray, p0: Tuple[float, float], p1: Tuple[float, float],
                         margin_frac: float = 0.12, step_px: float = 2.0) -> float:
    """Approximate pixel-length of the segment p0->p1 that overlaps
    rendered ligand content, excluding a margin at each end (the segment
    naturally touches content right at the source atom, and can graze the
    node label near the far end — neither of those should count as
    "crossing the molecule"). Sampled at a fixed *pixel* step rather than a
    fixed sample count, so a real crossing (spanning several pixels of a
    bond) scores proportionally to how much of the molecule it actually
    cuts through — a fixed sample count would dilute a real crossing on a
    long line down to the same tiny fraction as one incidental point
    grazing a short line, making the two indistinguishable. Higher = the
    connector line cuts across more of the drawn structure."""
    h, w = mask.shape
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n_samples = max(2, int(length * (1.0 - 2 * margin_frac) / step_px))
    ts = np.linspace(margin_frac, 1.0 - margin_frac, n_samples)
    xs = p0[0] + (p1[0] - p0[0]) * ts
    ys = p0[1] + (p1[1] - p0[1]) * ts
    xi = np.clip(xs.astype(int), 0, w - 1)
    yi = np.clip(ys.astype(int), 0, h - 1)
    return float(mask[yi, xi].sum()) * step_px


def _segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                         p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """True if segment p1->p2 crosses segment p3->p4 (standard
    orientation-based test; shared endpoints don't count as a crossing)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (v > 1e-9) - (v < -1e-9)
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    return o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4)


def _content_radial_profile(img, center: Tuple[float, float], n_bins: int = 72):
    """Per-angle max radius (in pixels) of actually-rendered content (bonds,
    atom labels, wedges — anything non-background) around center. Used so
    residue nodes are placed just outside the real drawing footprint in
    their own direction, instead of a single bounding circle that can miss
    wide atom labels (e.g. 'NH2') sticking out past the atom coordinate."""
    cx, cy = center
    h, w = img.shape[0], img.shape[1]
    mask = _content_mask(img)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.full(n_bins, max(w, h) / 2.0)

    dx = xs - cx
    dy = ys - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    bin_idx = (((theta + math.pi) / (2 * math.pi)) * n_bins).astype(int) % n_bins

    profile = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.any():
            profile[b] = r[sel].max()
    empty = profile == 0
    if empty.any():
        profile[empty] = r.max()
    return profile


def _profile_radius(profile, angle: float) -> float:
    n = len(profile)
    idx = int(((angle + math.pi) / (2 * math.pi)) * n) % n
    return float(profile[idx])


def _place_residue_nodes(anchor_pts: Dict[tuple, Tuple[float, float]],
                          center: Tuple[float, float],
                          profile, baseline_radius: float,
                          residues: Optional[Dict[tuple, List[dict]]] = None,
                          content_mask: Optional[np.ndarray] = None) -> Dict[tuple, Tuple[float, float]]:
    """baseline_radius: the plain atom-center bounding radius (as in v1).
    Per-direction placement radius is max(baseline_radius, profile-at-angle)
    — never smaller than the even circular layout, only pushed further out
    where the actual rendered content (e.g. a wide 'NH2' label, a wedge)
    extends past it. This fixes label/atom overlap without over-shrinking
    the layout in sparse directions, which previously caused crowding.

    residues/content_mask (optional): when given, each node's *starting*
    angle (before crowding relaxation) is chosen from a full sweep around
    the circle to minimize how much its connector line(s) cross (a) the
    ligand's own drawn content and (b) other residues' hbond/salt/pipi
    lines already placed earlier in this same pass (tracked in
    `placed_segments`, checked via `_segments_intersect`) — an atom tucked
    in a concave part of the ligand (e.g. a ring-fusion oxygen) can have a
    "natural" direction that necessarily cuts back across the molecule (or
    through another residue's line), while a clear line is available from
    a completely different angle (e.g. the top, if that's open). Crossing
    another line is penalized heavily (`LINE_CROSS_PENALTY_PX`) so it's
    avoided whenever a clear-enough angle exists, but the search still
    picks the least-bad option rather than nothing if every angle crosses
    something. This is a greedy, order-dependent pass (each residue only
    avoids lines already finalized, not later ones) rather than a global
    optimum, but removes the obvious cases in practice. The crowding
    relaxation afterward still resolves label/label overlap exactly as
    before, just starting from these better angles instead of the raw
    anchor directions.

    Only residues with at least one hbond/salt/pipi item are considered
    for relocation — a residue with purely hydrophobic contacts is left at
    its natural angle. There are usually many more hydrophobic contacts
    than hbond/pipi/salt ones, densely enough that some crossing is often
    geometrically unavoidable (this matches Maestro, which doesn't even
    draw a ligand-to-residue line for hydrophobic contacts at all — see
    the "Visual style matched to Maestro" note above). Letting every
    hydrophobic-only node compete for the same clear angles as the sparser,
    more important hbond/pipi/salt ones risked crowding several of them
    into a tight arc and shifting some back into a crossing during the
    relaxation pass below — better to spend the relocation budget only on
    the interactions worth guaranteeing clean."""
    if not anchor_pts:
        return {}
    cx, cy = center
    keys = list(anchor_pts.keys())

    def content_r(a):
        return max(baseline_radius, _profile_radius(profile, a))

    natural_angles = {k: math.atan2(anchor_pts[k][1] - cy, anchor_pts[k][0] - cx) for k in keys}

    if residues is not None and content_mask is not None:
        candidate_angles = np.linspace(-math.pi, math.pi, 36, endpoint=False)
        start_angles = {}
        placed_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        def crossing_cost(line_xy, node_p):
            ligand_cost = sum(_line_crossing_cost(content_mask, xy, node_p) for xy in line_xy)
            other_lines = sum(1 for xy in line_xy for p3, p4 in placed_segments
                               if _segments_intersect(xy, node_p, p3, p4))
            return ligand_cost + LINE_CROSS_PENALTY_PX * other_lines

        for k in keys:
            its = residues.get(k, [])
            # Only hbond/salt/pipi items actually draw a connector line
            # (hydrophobic contacts don't — see the render loop), so those
            # are the only ones worth avoiding a crossing for, and the only
            # ones that should count as an "obstacle" for later residues.
            line_xy = [it["_xy"] for it in its if it["kind"] in ("hbond", "salt", "pipi")]
            nat = natural_angles[k]
            if not line_xy:
                start_angles[k] = nat
                continue
            r0 = content_r(nat) + NODE_RADIAL_GAP
            p0 = (cx + r0 * math.cos(nat), cy + r0 * math.sin(nat))
            cost0 = crossing_cost(line_xy, p0)
            if cost0 <= NODE_LINE_CROSS_MIN_PX:
                # Already clean (or only grazing) — leave it at its natural
                # angle rather than risk an unnecessary reshuffle.
                start_angles[k] = nat
                placed_segments.extend((xy, p0) for xy in line_xy)
                continue
            best_a, best_key = nat, (round(cost0, 2), 0.0)
            for a in candidate_angles:
                r = content_r(a) + NODE_RADIAL_GAP
                node_p = (cx + r * math.cos(a), cy + r * math.sin(a))
                cost = crossing_cost(line_xy, node_p)
                # Tie-break toward the raw anchor angle so a node doesn't
                # jump to a distant, equally-clear angle for no reason.
                dev = abs(math.atan2(math.sin(a - nat), math.cos(a - nat)))
                key = (round(cost, 2), dev)
                if key < best_key:
                    best_key, best_a = key, a
            improved = cost0 - best_key[0]
            chosen_a = best_a if improved >= NODE_LINE_CROSS_MIN_IMPROVEMENT_PX else nat
            start_angles[k] = chosen_a
            r_final = content_r(chosen_a) + NODE_RADIAL_GAP
            p_final = (cx + r_final * math.cos(chosen_a), cy + r_final * math.sin(chosen_a))
            placed_segments.extend((xy, p_final) for xy in line_xy)
    else:
        start_angles = natural_angles

    order = sorted(keys, key=lambda k: start_angles[k])
    ang_list = [start_angles[k] for k in order]
    n = len(order)

    for _ in range(200):
        moved = False
        radii = [content_r(a) + NODE_RADIAL_GAP for a in ang_list]
        for i in range(n - 1):
            gap = ang_list[i + 1] - ang_list[i]
            r_local = min(radii[i], radii[i + 1])
            min_gap = (2 * NODE_HALF_W + 10) / r_local if r_local > 1 else 0.4
            if gap < min_gap:
                deficit = (min_gap - gap) / 2.0
                ang_list[i] -= deficit
                ang_list[i + 1] += deficit
                moved = True
        # Wraparound pair: the most-negative and most-positive angles in
        # this sorted (non-circular) list can still be close together on
        # the actual circle (e.g. -177 deg and +178 deg are only ~5 deg
        # apart) without ever being checked above, since they're not
        # "adjacent" in sorted order — missing this left two nodes stacked
        # directly on top of each other whenever a cluster straddled the
        # +-180 deg boundary.
        if n >= 2:
            gap = (ang_list[0] + 2 * math.pi) - ang_list[-1]
            r_local = min(radii[0], radii[-1])
            min_gap = (2 * NODE_HALF_W + 10) / r_local if r_local > 1 else 0.4
            if gap < min_gap:
                deficit = (min_gap - gap) / 2.0
                ang_list[-1] -= deficit
                ang_list[0] += deficit
                moved = True
        if not moved:
            break

    positions = {}
    for k, a in zip(order, ang_list):
        r = content_r(a) + NODE_RADIAL_GAP
        positions[k] = (cx + r * math.cos(a), cy + r * math.sin(a))

    # Safety net: the angular relaxation above only resolves ADJACENT (in
    # sorted-angle order) pairs, and assumes there's always enough
    # circumferential room at a fixed radius for everyone competing for it
    # — that assumption can fail when several nodes are forced into a
    # narrow arc (e.g. the crossing-avoidance search relocates one node to
    # dodge a line, landing it right next to an unrelated node it never
    # competed with for space, with too little combined arc for both).
    # Direct check for real *rectangular* box overlap between every pair
    # (not a single circular distance — two boxes offset mostly vertically
    # can sit much closer center-to-center than two offset horizontally
    # without actually overlapping, since the box is wide and short); any
    # pair whose boxes still overlap gets pushed further out radially
    # (not just angularly) until clear — guaranteed to terminate since
    # radius only grows.
    extra_r = {k: 0.0 for k in order}
    for _ in range(60):
        moved = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                k1, k2 = order[i], order[j]
                dx = abs(positions[k2][0] - positions[k1][0])
                dy = abs(positions[k2][1] - positions[k1][1])
                if dx < 2 * NODE_HALF_W + 4 and dy < 2 * NODE_HALF_H + 4:
                    for k, a in ((k1, ang_list[i]), (k2, ang_list[j])):
                        extra_r[k] += 12
                        r = content_r(a) + NODE_RADIAL_GAP + extra_r[k]
                        positions[k] = (cx + r * math.cos(a), cy + r * math.sin(a))
                    moved = True
        if not moved:
            break

    return positions


def _solvent_glow_layer(circles: List[Tuple[float, float, float]], w: int, h: int) -> np.ndarray:
    """RGBA (h, w, 4) float layer with one soft, blurred-edge disc per
    (x, y, radius) in `circles`. Each disc is flat (alpha 1) inside its
    radius and ramps linearly to 0 over SOLVENT_GLOW_BLUR_WIDTH px beyond
    it, matching Maestro's soft-aura look. Overlapping discs are combined
    with an elementwise max (not summed/alpha-stacked), so overlapping
    circles never blend into a darker patch — same reasoning as the flat
    no-double-blend circles this replaces, just with a soft edge instead
    of a hard one.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    for cx, cy, radius in circles:
        d = np.hypot(xs - cx, ys - cy)
        a = np.clip(1.0 - (d - radius) / SOLVENT_GLOW_BLUR_WIDTH, 0.0, 1.0)
        alpha = np.maximum(alpha, a)
    rgb = np.array(to_rgb(SOLVENT_EXPOSURE_COLOR), dtype=np.float32)
    layer = np.empty((h, w, 4), dtype=np.float32)
    layer[..., :3] = rgb
    layer[..., 3] = alpha * SOLVENT_GLOW_MAX_ALPHA
    return layer


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(ligand_sel: str, protein_sel: str, state: int, enabled: Set[str]):
    lig_model, mol_h, mol_2d, heavy_of = _build_ligand(ligand_sel, state)
    items = detect_all(lig_model, mol_h, mol_2d, heavy_of, ligand_sel, protein_sel, state, enabled)

    W = H = CANVAS_SIZE
    drawer = rdMolDraw2D.MolDraw2DCairo(W, H)
    drawer.drawOptions().padding = 0.12
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()
    img = mpimg.imread(io.BytesIO(drawer.GetDrawingText()), format="png")
    # RGBA version with a transparent (rather than opaque white) background,
    # used only for display so solvent-exposure circles drawn *underneath*
    # show as flat solid discs instead of a translucent haze over the whole
    # canvas. `img` itself stays opaque/unchanged for _content_radial_profile,
    # which relies on its white-background convention.
    bg = (img[..., :3] >= 0.95).all(axis=-1)
    img_display = np.dstack([img[..., :3], (~bg).astype(img.dtype)])

    n_heavy = mol_2d.GetNumAtoms()
    atom_xy = [(drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y) for i in range(n_heavy)]
    cx = sum(x for x, _ in atom_xy) / len(atom_xy)
    cy = sum(y for _, y in atom_xy) / len(atom_xy)
    baseline_radius = max(math.hypot(x - cx, y - cy) for x, y in atom_xy)

    residues: Dict[tuple, List[dict]] = defaultdict(list)
    for it in items:
        p = _pixel_anchor_for_item(it, heavy_of, mol_h, drawer)
        if p is None:
            continue
        it["_xy"] = p
        residues[(it["chain"], it["resn"], it["resi"])].append(it)

    anchor_pts = {}
    for key, its in residues.items():
        xs = [it["_xy"][0] for it in its]
        ys = [it["_xy"][1] for it in its]
        anchor_pts[key] = (sum(xs) / len(xs), sum(ys) / len(ys))

    profile = _content_radial_profile(img, (cx, cy))
    node_pos = _place_residue_nodes(anchor_pts, (cx, cy), profile, baseline_radius,
                                     residues=residues, content_mask=_content_mask(img))

    xs = [x for x, _ in atom_xy]
    ys = [y for _, y in atom_xy]
    if node_pos:
        xs += [x for x, _ in node_pos.values()]
        ys += [y for _, y in node_pos.values()]
    pad = NODE_HALF_W + 20
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    fig = Figure(figsize=(8.0, 8.0), dpi=110)
    ax = fig.add_subplot(111)

    # Solvent-exposure auras are drawn *first*, underneath the molecule
    # image (which has a transparent background — see img_display above),
    # so they read as soft blurred discs sitting behind the drawing rather
    # than a translucent wash blended on top of the bonds.
    any_solvent_drawn = False
    if "solvent" in enabled:
        rel_sasa, abs_sasa = _compute_relative_sasa(lig_model, ligand_sel, protein_sel, state)
        exposed = [(li, abs_area) for li, (rel, abs_area) in enumerate(zip(rel_sasa, abs_sasa))
                   if rel >= SOLVENT_REL_SASA_THRESHOLD]
        circles = []
        if exposed:
            lo = min(a for _, a in exposed)
            hi = max(a for _, a in exposed)
            for li, abs_area in exposed:
                p = _pixel_anchor(li, heavy_of, mol_h, drawer)
                if p is None:
                    continue
                # Normalized against the spread of exposed atoms *in this
                # diagram* (not a fixed Ų^2 scale) so the full radius range
                # is always used, even when every exposed atom happens to
                # sit in a narrow absolute-SASA band.
                frac = (abs_area - lo) / (hi - lo) if hi > lo else 1.0
                radius = SOLVENT_HALO_RADIUS_MIN + frac * (SOLVENT_HALO_RADIUS_MAX - SOLVENT_HALO_RADIUS_MIN)
                circles.append((p[0], p[1], radius))
        if circles:
            glow = _solvent_glow_layer(circles, W, H)
            ax.imshow(glow, extent=(0, W, H, 0), zorder=0)
            any_solvent_drawn = True

    ax.imshow(img_display, extent=(0, W, H, 0), zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")

    for key, (nx, ny) in node_pos.items():
        chain, resn, resi = key
        for it in residues[key]:
            if it["kind"] == "hydrophobic":
                # Matches Maestro: hydrophobic contacts are shown only by
                # coloring the residue node, no ligand-to-residue line —
                # there are usually too many, too densely packed, for a
                # line to ever be drawn without crossing the ligand (see
                # the residue-node-placement notes below).
                continue
            xy = it["_xy"]
            style = STYLE[it["kind"]]
            is_pipi = it["kind"] == "pipi"
            if it["kind"] == "hbond":
                # Arrowhead shows donor -> acceptor: points away from the
                # ligand when the ligand atom donates the H-bond, toward
                # the ligand when it accepts one from the residue. Either
                # way, whichever end lands *on the ligand atom* — tip or
                # tail — is pulled back off it by the same gap, so the
                # line never overlaps the atom label/bonds at that end,
                # not just when it happens to be the arrowhead end.
                # box_edge_dist is not negotiable: it's the distance at
                # which the line actually crosses the box's boundary, so
                # node_p must be pulled back *at least* this far or the
                # arrow visibly stops short of the box with a dangling
                # gap — worse than not pulling back at all. The small
                # aesthetic margin on top of it is only needed when this
                # end carries the *arrowhead* (a solid triangle that would
                # otherwise render half-hidden under the box, drawn on top
                # at a higher zorder) — a plain tail has no such shape to
                # protect, so it can go flush to the box edge instead of
                # leaving a gap that (with no arrowhead there to visually
                # bridge it) reads as a disconnected floating line.
                # start,end below is (lig_p, node_p) when donor — arrow
                # drawn start->end with the head at "end" (arrowstyle
                # "-|>"), so the node end is the tip exactly when the
                # ligand is the donor.
                node_is_tip = it.get("ligand_role") == "donor"
                d_total = math.hypot(nx - xy[0], ny - xy[1])
                box_edge_dist = _box_edge_pullback_amount(xy, (nx, ny), NODE_HALF_W, NODE_HALF_H, margin=0.0)
                margin, lig_pull = (6.0 if node_is_tip else 0.0), ARROW_LIGAND_GAP_PX
                remaining = d_total - box_edge_dist
                if remaining < margin + lig_pull + MIN_VISIBLE_ARROW_PX:
                    flexible = margin + lig_pull
                    scale = max(0.0, remaining - MIN_VISIBLE_ARROW_PX) / flexible if flexible > 0 else 0.0
                    margin *= scale
                    lig_pull *= scale
                node_p = _pull_back_point(xy, (nx, ny), box_edge_dist + margin)
                lig_p = _pull_back_point((nx, ny), xy, lig_pull)
                start, end = (lig_p, node_p) if it.get("ligand_role") == "donor" else (node_p, lig_p)
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                              shrinkA=0, shrinkB=0, color=style["color"],
                                              linestyle=style["linestyle"], linewidth=style["linewidth"],
                                              zorder=2))
            else:
                ax.plot([xy[0], nx], [xy[1], ny], color=style["color"],
                         linestyle=style["linestyle"], linewidth=style["linewidth"],
                         marker="o" if is_pipi else None,
                         markersize=5 if is_pipi else None,
                         markerfacecolor=style["color"], markeredgecolor=style["color"],
                         zorder=2)
            if it["kind"] in ("hbond", "salt"):
                mx, my = (xy[0] + nx) / 2, (xy[1] + ny) / 2
                ax.text(mx, my, f'{it["dist"]:.1f}', fontsize=7, color=style["color"],
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

        box = FancyBboxPatch((nx - NODE_HALF_W, ny - NODE_HALF_H), 2 * NODE_HALF_W, 2 * NODE_HALF_H,
                              boxstyle="round,pad=0.02,rounding_size=6",
                              linewidth=1.0, edgecolor="#404040",
                              facecolor=_residue_color(resn), zorder=3)
        ax.add_patch(box)
        ax.text(nx, ny, f"{resn}{resi}", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4)

    # Legends are anchored to a plot corner, but that corner is fixed
    # relative to the *axes*, not to where the actual content ends up —
    # when a residue node's radial angle happens to land in the same
    # corner (seen on an elongated ligand where several nodes cluster to
    # one side), a fixed "lower left" legend silently overlaps it. Picked
    # dynamically instead: whichever corner sits farthest from any atom or
    # node, checked against the two legends' own corners so they don't
    # both land on the same one.
    legend_corners = {
        "lower left": (xmin, ymax), "lower right": (xmax, ymax),
        "upper left": (xmin, ymin), "upper right": (xmax, ymin),
    }
    content_pts = atom_xy + list(node_pos.values())

    def _emptiest_corner(candidates):
        def min_dist(corner):
            cx, cy = legend_corners[corner]
            return min((math.hypot(px - cx, py - cy) for px, py in content_pts), default=float("inf"))
        return max(candidates, key=min_dist)

    present_kinds = [k for k in KIND_ORDER if k != "hydrophobic"
                     and any(it["kind"] == k for its in residues.values() for it in its)]
    kind_handles = [Line2D([0], [0], color=STYLE[k]["color"], linestyle=STYLE[k]["linestyle"],
                            linewidth=STYLE[k]["linewidth"],
                            marker="o" if k == "pipi" else None, markersize=5,
                            label=STYLE[k]["label"])
                    for k in present_kinds]
    if any_solvent_drawn:
        kind_handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=10,
                                    markerfacecolor=SOLVENT_EXPOSURE_COLOR, markeredgecolor="none",
                                    alpha=1.0, label="Solvent exposure"))
    remaining_corners = list(legend_corners)
    if kind_handles:
        interactions_loc = _emptiest_corner(remaining_corners)
        remaining_corners.remove(interactions_loc)
        leg1 = ax.legend(handles=kind_handles, loc=interactions_loc, fontsize=8, frameon=True,
                          title="Interactions", title_fontsize=8)
        ax.add_artist(leg1)

    present_categories = [c for c in RESIDUE_CATEGORY_ORDER
                          if any(_residue_category(resn) == c for _, resn, _ in residues.keys())]
    res_handles = [Patch(facecolor=RESIDUE_CATEGORY_COLORS[c], edgecolor="#404040",
                          label=RESIDUE_CATEGORY_LABELS[c])
                   for c in present_categories]
    if res_handles:
        residue_loc = _emptiest_corner(remaining_corners)
        ax.legend(handles=res_handles, loc=residue_loc, fontsize=8, frameon=True,
                  title="Residue type", title_fontsize=8)

    lig_atom0 = lig_model.atom[0]
    ax.set_title(f"{lig_atom0.resn}{lig_atom0.resi}  (state {state})", fontsize=11)

    n_interactions = sum(len(v) for v in residues.values())
    stats = dict(heavy_atoms=n_heavy, residues=len(residues), interactions=n_interactions)
    return fig, stats


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def i2d_list_ligands(selection="organic", state=1):
    """Print the distinct (chain, resn, resi) residues found in `selection`,
    indexed for use with i2d_generate(..., ligand_index=N)."""
    state = int(state)
    cands = _list_ligand_candidates(selection, state)
    if not cands:
        print(f"Interaction2D: no ligand residues found for '{selection}' (state {state}).")
        return cands
    for i, c in enumerate(cands):
        chain_s = c["chain"] if c["chain"] else "(none)"
        print(f"[{i}] {c['resn']} {c['resi']}  chain {chain_s}  ({c['natoms']} atoms)")
    return cands


cmd.extend("i2d_list_ligands", i2d_list_ligands)


def i2d_generate(protein="polymer.protein", ligand="organic", ligand_index=None, state=1,
                  filename="", hbond=1, hydrophobic=1, pipi=1, salt=1, solvent=1):
    """ligand_index: if the ligand selection matches more than one residue,
    pick the Nth candidate from i2d_list_ligands(ligand, state) instead of
    raising an error."""
    state = int(state)
    lig_sel = ligand
    if ligand_index is not None:
        cands = _list_ligand_candidates(ligand, state)
        idx = int(ligand_index)
        if not (0 <= idx < len(cands)):
            raise ValueError(
                f"ligand_index {idx} out of range (0..{len(cands) - 1}); "
                f"call i2d_list_ligands('{ligand}') to see candidates.")
        c = cands[idx]
        lig_sel = _ligand_residue_selection(c["chain"], c["resn"], c["resi"])

    enabled = set()
    if int(hbond):
        enabled.add("hbond")
    if int(hydrophobic):
        enabled.add("hydrophobic")
    if int(pipi):
        enabled.add("pipi")
    if int(salt):
        enabled.add("salt")
    if int(solvent):
        enabled.add("solvent")

    fig, stats = render(lig_sel, protein, state, enabled)
    print(f"Interaction2D: {stats['heavy_atoms']} heavy atoms, "
          f"{stats['residues']} residues, {stats['interactions']} interactions")
    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Interaction2D: saved {filename}")
    return fig


cmd.extend("i2d_generate", i2d_generate)


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------

_gui_window = None


def _protein_candidates() -> List[str]:
    names = ["polymer.protein"]
    for obj in cmd.get_names("objects"):
        try:
            if cmd.count_atoms(f"({obj}) and polymer.protein") > 0:
                names.append(obj)
        except Exception:
            pass
    return names


def _set_gui_none():
    global _gui_window
    _gui_window = None


def _open_gui():
    global _gui_window

    if _gui_window is not None:
        try:
            _gui_window.raise_()
            _gui_window.activateWindow()
            return
        except RuntimeError:
            _gui_window = None

    from pymol.Qt import QtWidgets, QtCore
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    win = QtWidgets.QWidget()
    _gui_window = win
    win.setWindowTitle("Interaction2D")
    win.setMinimumSize(580, 680)
    win.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    win.destroyed.connect(lambda: _set_gui_none())

    root = QtWidgets.QVBoxLayout(win)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(6)

    g_s = QtWidgets.QGroupBox("Setup")
    l_s = QtWidgets.QGridLayout(g_s)
    l_s.addWidget(QtWidgets.QLabel("Protein:"), 0, 0)
    e_prot = QtWidgets.QComboBox()
    e_prot.setEditable(True)
    e_prot.addItems(_protein_candidates())
    l_s.addWidget(e_prot, 0, 1)

    l_s.addWidget(QtWidgets.QLabel("Ligand scope:"), 1, 0)
    hl_lig = QtWidgets.QHBoxLayout()
    e_lig = QtWidgets.QLineEdit("organic")
    b_find = QtWidgets.QPushButton("Find")
    b_find.setFixedWidth(50)
    hl_lig.addWidget(e_lig)
    hl_lig.addWidget(b_find)
    l_s.addLayout(hl_lig, 1, 1)

    l_s.addWidget(QtWidgets.QLabel("Ligand:"), 2, 0)
    combo_lig = QtWidgets.QComboBox()
    l_s.addWidget(combo_lig, 2, 1)

    l_s.addWidget(QtWidgets.QLabel("State:"), 3, 0)
    sp_state = QtWidgets.QSpinBox()
    sp_state.setMinimum(1)
    sp_state.setMaximum(99999)
    sp_state.setValue(1)
    l_s.addWidget(sp_state, 3, 1)
    root.addWidget(g_s)

    g_i = QtWidgets.QGroupBox("Interactions")
    l_i = QtWidgets.QHBoxLayout(g_i)
    cb_hbond = QtWidgets.QCheckBox("H-bonds"); cb_hbond.setChecked(True)
    cb_hydro = QtWidgets.QCheckBox("Hydrophobic"); cb_hydro.setChecked(True)
    cb_pipi = QtWidgets.QCheckBox("Pi-stacking"); cb_pipi.setChecked(True)
    cb_salt = QtWidgets.QCheckBox("Salt bridges"); cb_salt.setChecked(True)
    cb_solvent = QtWidgets.QCheckBox("Solvent exposure"); cb_solvent.setChecked(True)
    for cb in (cb_hbond, cb_hydro, cb_pipi, cb_salt, cb_solvent):
        l_i.addWidget(cb)
    root.addWidget(g_i)

    hl_btn = QtWidgets.QHBoxLayout()
    b_gen = QtWidgets.QPushButton("Generate")
    b_export = QtWidgets.QPushButton("Export…")
    b_export.setEnabled(False)
    hl_btn.addWidget(b_gen)
    hl_btn.addWidget(b_export)
    root.addLayout(hl_btn)

    lbl_status = QtWidgets.QLabel("Ready.")
    lbl_status.setWordWrap(True)
    root.addWidget(lbl_status)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    canvas_holder = QtWidgets.QWidget()
    canvas_layout = QtWidgets.QVBoxLayout(canvas_holder)
    scroll.setWidget(canvas_holder)
    root.addWidget(scroll, stretch=1)

    state_box = {"canvas": None, "fig": None}

    def _refresh_ligands():
        combo_lig.clear()
        try:
            cands = _list_ligand_candidates(e_lig.text().strip(), sp_state.value())
        except Exception as exc:
            lbl_status.setText(f"Error: {exc}")
            return
        if not cands:
            lbl_status.setText("No ligand residues found for this selection.")
            return
        for c in cands:
            label = f"{c['resn']} {c['resi']}"
            if c["chain"]:
                label += f"  chain {c['chain']}"
            label += f"  ({c['natoms']} atoms)"
            combo_lig.addItem(label, (c["chain"], c["resn"], c["resi"]))
        lbl_status.setText(f"Found {len(cands)} ligand residue(s).")

    def _on_generate():
        if combo_lig.count() == 0:
            lbl_status.setText("No ligand selected — click Find first.")
            return

        enabled = set()
        if cb_hbond.isChecked(): enabled.add("hbond")
        if cb_hydro.isChecked(): enabled.add("hydrophobic")
        if cb_pipi.isChecked(): enabled.add("pipi")
        if cb_salt.isChecked(): enabled.add("salt")
        if cb_solvent.isChecked(): enabled.add("solvent")

        chain, resn, resi = combo_lig.currentData()
        lig_sel = _ligand_residue_selection(chain, resn, resi)

        try:
            fig, stats = render(lig_sel, e_prot.currentText().strip(),
                                 sp_state.value(), enabled)
        except Exception as exc:
            lbl_status.setText(f"Error: {exc}")
            b_export.setEnabled(False)
            return

        if state_box["canvas"] is not None:
            canvas_layout.removeWidget(state_box["canvas"])
            state_box["canvas"].setParent(None)

        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumSize(CANVAS_SIZE, CANVAS_SIZE)
        canvas_layout.addWidget(canvas)
        state_box["canvas"] = canvas
        state_box["fig"] = fig

        lbl_status.setText(f"{stats['heavy_atoms']} heavy atoms, "
                            f"{stats['residues']} residues, "
                            f"{stats['interactions']} interactions")
        b_export.setEnabled(True)

    def _on_export():
        fig = state_box["fig"]
        if fig is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            win, "Export diagram", "", "PNG (*.png);;SVG (*.svg)")
        if path:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            lbl_status.setText(f"Saved {path}")

    b_find.clicked.connect(_refresh_ligands)
    b_gen.clicked.connect(_on_generate)
    b_export.clicked.connect(_on_export)

    _refresh_ligands()
    win.show()


def i2d_gui():
    _open_gui()


cmd.extend("i2d_gui", i2d_gui)


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("Interaction2D", _open_gui)
