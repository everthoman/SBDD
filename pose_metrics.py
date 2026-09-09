#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Pose Metrics — score docking poses against a reference ligand

Takes a protein structure (PDB), a reference ligand (SDF) and a stack of docking
poses (SDF, with whatever score fields the docking program wrote) and annotates
every pose with:

  Ref_Sim    2D Morgan ECFP4 (r=2, 2048 bit) Tanimoto to the reference ligand.
             Both sides are neutralized and tautomer-canonicalized first, so a
             different protomer/tautomer of the same compound still scores 1.0.
  MCS_RMSD   Heavy-atom RMSD over the maximum common substructure with the
             reference, in place (poses already sit in the binding site).
             --mcs-align superimposes on the MCS first instead.
  Shape_Sim  RDKit shape Tanimoto (1 - ShapeTanimotoDist) against the reference,
             in place; --shape-align does an Open3DAlign overlay first.
  PLIF_Sim   Protein-ligand interaction fingerprint Tanimoto to the reference,
             via ProLIF when installed, otherwise ODDT.
  PB_Flags   Number of failed PoseBusters checks (dock config), with the names
             of the failing checks in PB_Failed.

The metric fields are written both into an annotated SDF and into a CSV table
that carries every original pose field (docking scores etc.) along with them.
Mirrors the pose annotations produced by the GNINA web app.

Author: Evert J. Homan, PhD
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import argcomplete
from argcomplete.completers import FilesCompleter
from rich_argparse import RawDescriptionRichHelpFormatter
from tqdm import tqdm

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS, rdShapeHelpers

RDLogger.DisableLog("rdApp.*")

METRICS = ("ref", "mcs", "shape", "plif", "pb")


# --- molecule sanitization -------------------------------------------------
# Ported from the GNINA web app so both tools score identically.

def _sanitize_mol_keep_hs(mol):
    """Sanitize an RDKit mol loaded with sanitize=False, in place. Returns mol.

    Full SanitizeMol can fail on docking output (unusual atom types, charges).
    When it does, fall back to FastFindRings so that ring info is populated and
    Morgan fingerprints / MCS still work.
    """
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.FastFindRings(mol)

    # Some tools (e.g. PyMOL) export delocalized groups outside rings — a
    # zwitterionic carboxylate as C(:O):[O-] — using MDL bond type 4
    # ("aromatic"). RDKit's aromaticity model only covers ring systems, so
    # SanitizeMol leaves those bonds as bare BondType.AROMATIC without ever
    # resolving them into a single/double pair, and the group then hashes
    # differently than its normal Kekule form. Resolve them explicitly.
    stray = [b for b in mol.GetBonds()
             if b.GetBondType() == Chem.BondType.AROMATIC and not b.IsInRing()]
    if stray:
        stray_idx = {b.GetIdx() for b in stray}
        resolved = set()
        # An atom may be the DOUBLE end of at most one stray bond; tracking that
        # across atoms keeps a 3+-atom conjugated chain from getting a cumulated
        # double bond that SanitizeMol would reject.
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
                pool.sort(key=lambda b: b.GetOtherAtom(atom).GetFormalCharge(), reverse=True)
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


def _sanitize_mol(mol):
    """Like _sanitize_mol_keep_hs, but also strips explicit Hs for heavy-atom-only
    comparisons (Morgan FP, shape overlap, MCS).

    SDMolSupplier/MolFromMolBlock silently skip their removeHs step when
    sanitize=False, so explicit Hs in pose SDFs survive parsing even when the
    reference ligand has none — an asymmetry that corrupts every heavy-atom
    comparison. Strip them here instead.
    """
    mol = _sanitize_mol_keep_hs(mol)
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
        # RemoveHs(sanitize=False) drops ring info, so repopulate it before any
        # fingerprint/shape/MCS call hits "RingInfo not initialized".
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            Chem.FastFindRings(mol)
    except Exception:
        pass
    return mol


_STANDARDIZERS: Dict[str, Any] = {}


def _standardize(mol):
    """Neutralize protonation state and collapse to a canonical tautomer so that
    Ref_Sim treats different protomers/tautomers of one compound as identical.

    Order matters: uncharge first (protonated pyridinium -> neutral pyridine),
    then canonicalize tautomers on the neutral form.
    """
    if not _STANDARDIZERS:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _STANDARDIZERS["uncharger"] = rdMolStandardize.Uncharger()
        _STANDARDIZERS["tautomer"] = rdMolStandardize.TautomerEnumerator()
    try:
        mol = _STANDARDIZERS["uncharger"].uncharge(mol)
    except Exception:
        pass
    try:
        mol = _STANDARDIZERS["tautomer"].Canonicalize(mol)
    except Exception:
        pass
    return mol


# --- SDF handling ----------------------------------------------------------
# Poses are carried around as raw text blocks rather than as RDKit mols: that
# preserves every original field byte-for-byte in the annotated SDF, even for
# poses RDKit can only parse with sanitize=False.

_PROP_TAG_RE = re.compile(r"^>\s+<([^>]+)>")


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def _is_counts_line(line: str) -> bool:
    return line.rstrip().endswith(("V2000", "V3000"))


def _trim_leading_blanks(lines: List[str]) -> List[str]:
    """Drop separator blank lines that leaked in front of a record.

    A molblock's counts line is always the 4th, and an *empty title line* is
    legal — blindly stripping a record would shift the counts line up and make
    RDKit reject the whole molecule. So only drop a leading blank when doing so
    puts a counts line back in position 4.
    """
    i = 0
    while i < len(lines) and not lines[i].strip():
        if len(lines) > i + 3 and _is_counts_line(lines[i + 3]):
            break
        i += 1
    return lines[i:]


def read_blocks(path: Path) -> List[str]:
    """Split an SDF (optionally gzipped) into molecule blocks, without the $$$$."""
    blocks: List[str] = []
    current: List[str] = []
    for line in _read_text(path).splitlines():
        if line.startswith("$$$$"):
            if any(l.strip() for l in current):
                blocks.append("\n".join(_trim_leading_blanks(current)).rstrip())
            current = []
        else:
            current.append(line)
    if any(l.strip() for l in current):
        blocks.append("\n".join(_trim_leading_blanks(current)).rstrip())
    return blocks


_ID_FIELDS = ("Ligand_ID", "ID", "Name", "NAME", "Title", "TITLE", "_Name", "id", "name")


def block_name(block: str, idx: int, props: Optional[Dict[str, str]] = None,
               id_field: Optional[str] = None) -> str:
    """Molecule title, else an identifier field, else a generated pose id.

    Docking output often leaves the title line empty and carries the compound
    name in a data field instead.
    """
    lines = block.splitlines()
    title = lines[0].strip() if lines else ""
    if title and id_field is None:
        return title
    if props is None:
        props = block_props(block)
    if id_field is not None:
        value = props.get(id_field, "").strip()
        if value:
            return value
    if title:
        return title
    for field in _ID_FIELDS:
        value = props.get(field, "").strip()
        if value:
            return value
    return f"pose_{idx + 1}"


def block_props(block: str) -> Dict[str, str]:
    """Parse the `> <TAG>` data fields of an SDF block into a dict."""
    props: Dict[str, str] = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        m = _PROP_TAG_RE.match(lines[i])
        if not m:
            i += 1
            continue
        key = m.group(1)
        i += 1
        values = []
        while i < len(lines) and lines[i].strip():
            values.append(lines[i].rstrip())
            i += 1
        props[key] = "\n".join(values)
    return props


def strip_prop(block: str, key: str) -> str:
    """Remove an existing `> <key>` field so re-running never duplicates it."""
    out: List[str] = []
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        m = _PROP_TAG_RE.match(lines[i])
        if m and m.group(1) == key:
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).rstrip()


def set_prop(block: str, key: str, value: str) -> str:
    return f"{strip_prop(block, key)}\n\n> <{key}>\n{value}\n"


def parse_pose(block: str, keep_hs: bool):
    """Parse one pose block into a sanitized RDKit mol, or None."""
    try:
        mol = Chem.MolFromMolBlock(block, removeHs=not keep_hs, sanitize=False)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return _sanitize_mol_keep_hs(mol) if keep_hs else _sanitize_mol(mol)
    except Exception:
        return None


def load_reference(path: Path, keep_hs: bool = False):
    """First molecule of the reference SDF, sanitized the same way as the poses."""
    for block in read_blocks(path):
        mol = parse_pose(block, keep_hs)
        if mol is not None:
            return mol
    return None


def reference_block(path: Path) -> Optional[str]:
    """Raw text block of the first parseable reference molecule."""
    for block in read_blocks(path):
        if Chem.MolFromMolBlock(block, removeHs=False, sanitize=False) is not None:
            return block
    return None


def fmt(value: Optional[float], decimals: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}"


# --- parallel helper -------------------------------------------------------

def chunked(items: Sequence[Any], n_chunks: int) -> List[Tuple[int, List[Any]]]:
    """Split items into at most n_chunks contiguous (offset, slice) pieces."""
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    size = -(-len(items) // n_chunks)
    return [(i, list(items[i:i + size])) for i in range(0, len(items), size)]


def run_chunks(worker, payloads: List[Any], workers: int, total: int,
               desc: str, quiet: bool) -> List[Any]:
    """Map worker over payloads, in order, with a pose-level progress bar."""
    results: List[Any] = [None] * len(payloads)
    bar = tqdm(total=total, desc=desc, unit="pose", disable=quiet)
    if workers <= 1:
        for i, payload in enumerate(payloads):
            results[i] = worker(payload)
            bar.update(len(payload[1]))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, p): i for i, p in enumerate(payloads)}
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                bar.update(len(payloads[i][1]))
    bar.close()
    return results


# --- Ref_Sim ---------------------------------------------------------------

def calc_ref_sim(ref_mol, blocks: List[str], quiet: bool) -> List[Optional[float]]:
    """Morgan ECFP4 Tanimoto to the reference, on standardized heavy-atom mols."""
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(
        _standardize(ref_mol), radius=2, nBits=2048)
    sims: List[Optional[float]] = []
    for block in tqdm(blocks, desc="Ref_Sim", unit="pose", disable=quiet):
        mol = parse_pose(block, keep_hs=False)
        if mol is None:
            sims.append(None)
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                _standardize(mol), radius=2, nBits=2048)
            sims.append(DataStructs.TanimotoSimilarity(ref_fp, fp))
        except Exception:
            sims.append(None)
    return sims


# --- MCS_RMSD --------------------------------------------------------------

def _mcs_chunk(payload: Tuple[Any, List[str]]) -> List[Optional[float]]:
    """Worker: MCS RMSD for one chunk of pose blocks."""
    (ref_block, timeout, align), blocks = payload
    ref_mol = _sanitize_mol(Chem.MolFromMolBlock(ref_block, removeHs=True, sanitize=False))
    out: List[Optional[float]] = []
    for block in blocks:
        out.append(_mcs_rmsd_one(ref_mol, block, timeout, align))
    return out


def _mcs_rmsd_one(ref_mol, block: str, timeout: int, align: bool) -> Optional[float]:
    try:
        pose_mol = parse_pose(block, keep_hs=False)
        if pose_mol is None or not pose_mol.GetNumConformers():
            return None

        mcs = rdFMCS.FindMCS(
            [ref_mol, pose_mol],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            timeout=timeout,
        )
        if mcs.numAtoms < 3:
            return None

        patt = Chem.MolFromSmarts(mcs.smartsString)
        # uniquify=False keeps symmetry-equivalent mappings (e.g. a flipped
        # phenyl), so the reported RMSD is the best over all of them rather than
        # an arbitrary atom numbering artifact.
        ref_matches = ref_mol.GetSubstructMatches(patt, uniquify=False, maxMatches=16)
        pose_matches = pose_mol.GetSubstructMatches(patt, uniquify=False, maxMatches=16)
        if not ref_matches or not pose_matches:
            return None

        if align:
            best = float("inf")
            for ref_match in ref_matches:
                for pose_match in pose_matches:
                    probe = Chem.Mol(pose_mol)
                    try:
                        rmsd = AllChem.AlignMol(
                            probe, ref_mol, atomMap=list(zip(pose_match, ref_match)))
                    except Exception:
                        continue
                    best = min(best, rmsd)
            return None if best == float("inf") else best

        ref_conf = ref_mol.GetConformer()
        pose_conf = pose_mol.GetConformer()
        ref_coords = {m: [list(ref_conf.GetAtomPosition(i)) for i in m] for m in ref_matches}
        best = float("inf")
        for pose_match in pose_matches:
            pose_xyz = [list(pose_conf.GetAtomPosition(i)) for i in pose_match]
            for ref_match in ref_matches:
                ref_xyz = ref_coords[ref_match]
                total = sum(
                    (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
                    for a, b in zip(ref_xyz, pose_xyz)
                )
                best = min(best, (total / len(ref_xyz)) ** 0.5)
        return None if best == float("inf") else best
    except Exception:
        return None


def calc_mcs_rmsd(ref_block: str, blocks: List[str], timeout: int, align: bool,
                  workers: int, quiet: bool) -> List[Optional[float]]:
    payloads = [((ref_block, timeout, align), chunk)
                for _, chunk in chunked(blocks, workers * 4)]
    chunks = run_chunks(_mcs_chunk, payloads, workers, len(blocks), "MCS_RMSD", quiet)
    return [v for chunk in chunks for v in chunk]


# --- Shape_Sim -------------------------------------------------------------

def _o3a_align(probe, ref_mol) -> None:
    """Overlay probe onto ref_mol with Open3DAlign, in place."""
    from rdkit.Chem import rdMolAlign
    ref_props = AllChem.MMFFGetMoleculeProperties(ref_mol)
    probe_props = AllChem.MMFFGetMoleculeProperties(probe)
    if ref_props is not None and probe_props is not None:
        rdMolAlign.GetO3A(probe, ref_mol, probe_props, ref_props).Align()
        return
    # MMFF has no parameters for some atom types; Crippen O3A works off logP/MR
    # contributions instead and covers whatever MMFF cannot type.
    rdMolAlign.GetCrippenO3A(probe, ref_mol).Align()


def calc_shape_sim(ref_mol, blocks: List[str], align: bool,
                   quiet: bool) -> List[Optional[float]]:
    """Shape Tanimoto against the reference. Poses already sit in the binding
    site, so by default the overlap is measured on the docked coordinates;
    --shape-align instead asks how similar the two shapes can be made."""
    if not ref_mol.GetNumConformers():
        return [None] * len(blocks)

    sims: List[Optional[float]] = []
    align_failures = 0
    for block in tqdm(blocks, desc="Shape_Sim", unit="pose", disable=quiet):
        mol = parse_pose(block, keep_hs=False)
        if mol is None or not mol.GetNumConformers():
            sims.append(None)
            continue
        try:
            probe = mol
            if align:
                probe = Chem.Mol(mol)
                try:
                    _o3a_align(probe, ref_mol)
                except Exception:
                    align_failures += 1
                    probe = mol  # fall back to the docked pose
            sims.append(1.0 - rdShapeHelpers.ShapeTanimotoDist(ref_mol, probe))
        except Exception:
            sims.append(None)

    if align_failures:
        print(f"Warning: Open3DAlign failed for {align_failures} pose(s); "
              f"those used the docked coordinates", file=sys.stderr)
    return sims


# --- PLIF_Sim --------------------------------------------------------------

def _plif_prolif(receptor: Path, ref_sdf: Path, blocks: List[str],
                 quiet: bool) -> List[Optional[float]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import MDAnalysis as mda
        import prolif

        universe = mda.Universe(str(receptor))
        prot_ag = universe.select_atoms("protein")
        try:
            protein = prolif.Molecule.from_mda(prot_ag)
        except Exception:
            # Some PDB files carry connectivity RDKit rejects under the strict
            # valence check that NoImplicit=True enforces.
            protein = prolif.Molecule.from_mda(prot_ag, NoImplicit=False)

    # PLIF needs explicit Hs for H-bond donor/acceptor perception, so unlike the
    # other metrics the ligands keep them.
    ref_mol = load_reference(ref_sdf, keep_hs=True)
    if ref_mol is None:
        raise RuntimeError("could not load reference ligand")
    ref_lig = prolif.Molecule.from_rdkit(Chem.AddHs(ref_mol, addCoords=True))

    ligands, valid = [], []
    for i, block in enumerate(tqdm(blocks, desc="PLIF parse", unit="pose", disable=quiet)):
        mol = parse_pose(block, keep_hs=True)
        if mol is None:
            continue
        try:
            ligands.append(prolif.Molecule.from_rdkit(mol))
            valid.append(i)
        except Exception:
            pass

    sims: List[Optional[float]] = [None] * len(blocks)
    if not ligands:
        return sims

    # Reference and poses go through one Fingerprint instance so every bitvector
    # shares the same residue-interaction columns and they stay comparable.
    fp = prolif.Fingerprint()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp.run_from_iterable([ref_lig] + ligands, protein, progress=not quiet)
    bitvectors = fp.to_bitvectors()
    ref_bv = bitvectors[0]
    for i, bv in zip(valid, bitvectors[1:]):
        sims[i] = DataStructs.TanimotoSimilarity(ref_bv, bv)
    return sims


def _plif_oddt(receptor: Path, ref_sdf: Path, blocks: List[str],
               quiet: bool) -> List[Optional[float]]:
    import oddt
    from oddt.fingerprints import InteractionFingerprint, tanimoto

    protein = next(oddt.toolkit.readfile("pdb", str(receptor)))
    protein.protein = True
    ref = next(oddt.toolkit.readfile("sdf", str(ref_sdf)))
    ref_fp = InteractionFingerprint(ref, protein, strict=True)

    sims: List[Optional[float]] = []
    for block in tqdm(blocks, desc="PLIF_Sim", unit="pose", disable=quiet):
        try:
            mol = oddt.toolkit.readstring("sdf", block + "\n$$$$\n")
            fp = InteractionFingerprint(mol, protein, strict=True)
            sims.append(None if fp is None or len(fp) == 0 else float(tanimoto(ref_fp, fp)))
        except Exception:
            sims.append(None)
    return sims


def calc_plif_sim(receptor: Path, ref_sdf: Path, blocks: List[str], backend: str,
                  quiet: bool) -> Tuple[List[Optional[float]], str]:
    """Interaction-fingerprint Tanimoto to the reference. Returns (values, backend)."""
    order = [backend] if backend != "auto" else ["prolif", "oddt"]
    last_error = None
    for name in order:
        try:
            if name == "prolif":
                return _plif_prolif(receptor, ref_sdf, blocks, quiet), "prolif"
            return _plif_oddt(receptor, ref_sdf, blocks, quiet), "oddt"
        except ImportError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"no usable PLIF backend ({last_error})")


# --- PoseBusters -----------------------------------------------------------
# Driven through the `bust` CLI rather than `import posebusters`: the package
# version installed alongside an older RDKit raises on Mol.GetProp(autoConvert=),
# and this repo's own posebusters.py would shadow the import anyway. Going
# through the executable also lets --bust-executable borrow a working
# PoseBusters from another conda env.

# Bookkeeping columns of the report, not pose-quality checks.
_PB_INFRA_COLS = {"file", "molecule", "position", "mol_cond_loaded", "mol_true_loaded"}


def _bust_chunk(exe: str, receptor: Path, blocks: List[str], indices: List[int],
                tmpdir: Path) -> Optional[List[Dict[str, str]]]:
    """Run `bust` over the given poses. Returns one report row per pose, or None
    if the run failed or came back with the wrong number of rows."""
    chunk_sdf = tmpdir / f"chunk_{indices[0]}_{len(indices)}.sdf"
    chunk_csv = chunk_sdf.with_suffix(".csv")
    with open(chunk_sdf, "w") as fh:
        for i in indices:
            fh.write(blocks[i].rstrip() + "\n\n$$$$\n")
    try:
        # bust exits 0 even when it blows up internally, so the row count below
        # is what actually tells us the run was good.
        subprocess.run(
            [exe, str(chunk_sdf), "-p", str(receptor),
             "--outfmt", "csv", "--output", str(chunk_csv)],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(f"PoseBusters executable not found: {exe}")
    finally:
        chunk_sdf.unlink(missing_ok=True)

    if not chunk_csv.exists():
        return None
    with open(chunk_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    chunk_csv.unlink(missing_ok=True)
    return rows if len(rows) == len(indices) else None


def _failed_checks(row: Dict[str, str]) -> List[str]:
    return [k for k, v in row.items()
            if k not in _PB_INFRA_COLS and str(v).strip().lower() == "false"]


def calc_posebusters(receptor: Path, blocks: List[str], exe: str, chunk_size: int,
                     quiet: bool, report_path: Optional[Path], id_field: Optional[str]):
    """Returns (per-pose (fail_count, failed_checks), per-check fail totals).

    A pose that PoseBusters cannot evaluate is isolated by bisection so it does
    not cost the rest of its chunk; those poses come back as (None, []).
    """
    per_pose: List[Tuple[Optional[int], List[str]]] = [(None, [])] * len(blocks)
    totals: Dict[str, int] = {}
    report_rows: List[Dict[str, str]] = []
    bar = tqdm(total=len(blocks), desc="PoseBusters", unit="pose", disable=quiet)

    def process(indices: List[int], tmpdir: Path) -> None:
        rows = _bust_chunk(exe, receptor, blocks, indices, tmpdir)
        if rows is None:
            if len(indices) == 1:
                bar.update(1)
                return
            mid = len(indices) // 2
            process(indices[:mid], tmpdir)
            process(indices[mid:], tmpdir)
            return
        for pos, row in zip(indices, rows):
            failed = _failed_checks(row)
            per_pose[pos] = (len(failed), failed)
            for check in failed:
                totals[check] = totals.get(check, 0) + 1
            if report_path is not None:
                report_rows.append({
                    "pose_index": str(pos),
                    "Name": block_name(blocks[pos], pos, id_field=id_field),
                    **{k: v for k, v in row.items() if k not in ("file", "position")},
                })
        bar.update(len(indices))

    with tempfile.TemporaryDirectory(prefix="pose_metrics_pb_") as tmp:
        tmpdir = Path(tmp)
        for start in range(0, len(blocks), chunk_size):
            process(list(range(start, min(start + chunk_size, len(blocks)))), tmpdir)
    bar.close()

    if report_path is not None and report_rows:
        write_csv(report_path, report_rows)

    return per_pose, totals


# --- output ----------------------------------------------------------------

def write_sdf(path: Path, blocks: List[str]) -> None:
    with open(path, "w") as fh:
        for block in blocks:
            fh.write(block.rstrip("\r\n ") + "\n\n$$$$\n")


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def summarize(label: str, values: List[Optional[float]], decimals: int = 3) -> str:
    valid = [v for v in values if v is not None]
    if not valid:
        return f"  {label:<11} no valid values ({len(values)} poses)"
    return (f"  {label:<11} n={len(valid)}/{len(values)}  "
            f"min={min(valid):.{decimals}f}  "
            f"mean={sum(valid) / len(valid):.{decimals}f}  "
            f"max={max(valid):.{decimals}f}")


# --- CLI -------------------------------------------------------------------

def output_stem(name: str) -> str:
    """Accept either a bare stem or a filename, so both outputs share the stem."""
    p = Path(name)
    if p.suffix.lower() in (".sdf", ".csv"):
        p = p.with_suffix("")
    return str(p)


def parse_metrics(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(METRICS)
    chosen = [m.strip().lower() for m in value.split(",") if m.strip()]
    bad = [m for m in chosen if m not in METRICS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown metric(s): {', '.join(bad)} (choose from {', '.join(METRICS)}, or all)")
    return [m for m in METRICS if m in chosen]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  # Every metric, writing results.sdf + results.csv
  %(prog)s -r receptor.pdb -a ref.sdf -l poses.sdf -o results

  # 2D/3D ligand comparison only — no receptor needed
  %(prog)s -a ref.sdf -l poses.sdf -o results -m ref,mcs,shape

  # PoseBusters only, keeping the full per-check report
  %(prog)s -r receptor.pdb -l poses.sdf -o busted -m pb --pb-report checks.csv

  # Ask how similar the shapes can be, rather than how they overlap as docked
  %(prog)s -a ref.sdf -l poses.sdf -o results -m shape --shape-align
""")

    parser.add_argument(
        "-l", "--ligands", type=Path, required=True, metavar="POSES.sdf",
        help="Docking poses (SDF or SDF.gz), with their score fields",
    ).completer = FilesCompleter(allowednames=(".sdf", ".gz"))
    parser.add_argument(
        "-r", "--receptor", type=Path, default=None, metavar="PROTEIN.pdb",
        help="Protein structure (PDB); required for the plif and pb metrics",
    ).completer = FilesCompleter(allowednames=(".pdb",))
    parser.add_argument(
        "-a", "--reference", type=Path, default=None, metavar="REF.sdf",
        help="Reference ligand (SDF); required for the ref, mcs, shape and plif metrics",
    ).completer = FilesCompleter(allowednames=(".sdf", ".gz"))
    parser.add_argument(
        "-o", "--output", default="pose_metrics", metavar="NAME",
        help="Output stem; writes NAME.sdf and NAME.csv (default: pose_metrics). "
             "A trailing .sdf/.csv is stripped.",
    ).completer = FilesCompleter(allowednames=(".sdf", ".csv"))
    parser.add_argument(
        "-m", "--metrics", type=parse_metrics, default="all", metavar="LIST",
        help=f"Comma-separated subset of {','.join(METRICS)} (default: all)")

    parser.add_argument("--mcs-timeout", type=int, default=2, metavar="SEC",
                        help="Per-pose MCS search timeout (default: 2)")
    parser.add_argument("--mcs-align", action="store_true",
                        help="Superimpose each pose on the reference MCS before "
                             "measuring RMSD (default: in-place, docked coordinates)")
    parser.add_argument("--shape-align", action="store_true",
                        help="Open3DAlign each pose onto the reference before the "
                             "shape comparison (default: in-place)")
    parser.add_argument("--plif-backend", choices=("auto", "prolif", "oddt"), default="auto",
                        help="Interaction-fingerprint engine (default: auto — "
                             "ProLIF if installed, else ODDT)")
    parser.add_argument("--pb-report", type=Path, default=None, metavar="FILE.csv",
                        help="Also write the full per-check PoseBusters table here",
                        ).completer = FilesCompleter(allowednames=(".csv",))
    parser.add_argument("--bust-executable", default="bust", metavar="PATH",
                        help="PoseBusters executable (default: bust on PATH). Point "
                             "this at another conda env's bin/bust if the local "
                             "PoseBusters and RDKit versions disagree.")
    parser.add_argument("--pb-chunk-size", type=int, default=200, metavar="N",
                        help="Poses per `bust` invocation (default: 200)")
    parser.add_argument("--id-field", default=None, metavar="TAG",
                        help="SDF field to use as the pose Name in the CSV "
                             "(default: molecule title, else Ligand_ID/ID/Name)")
    parser.add_argument("-j", "--workers", type=int, default=min(8, os.cpu_count() or 1),
                        metavar="N", help="Worker processes for MCS_RMSD "
                             "(PoseBusters parallelizes internally)")
    parser.add_argument("--no-sdf", action="store_true", help="Skip the annotated SDF output")
    parser.add_argument("--no-csv", action="store_true", help="Skip the CSV table")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bars")

    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    if isinstance(args.metrics, str):
        args.metrics = parse_metrics(args.metrics)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    metrics = args.metrics

    if not args.ligands.exists():
        print(f"Error: poses file not found: {args.ligands}", file=sys.stderr)
        return 1

    needs_ref = [m for m in metrics if m in ("ref", "mcs", "shape", "plif")]
    if needs_ref and args.reference is None:
        print(f"Error: --reference is required for: {', '.join(needs_ref)}", file=sys.stderr)
        return 1
    if args.reference is not None and not args.reference.exists():
        print(f"Error: reference file not found: {args.reference}", file=sys.stderr)
        return 1

    needs_receptor = [m for m in metrics if m in ("plif", "pb")]
    if needs_receptor and args.receptor is None:
        print(f"Error: --receptor is required for: {', '.join(needs_receptor)}", file=sys.stderr)
        return 1
    if args.receptor is not None and not args.receptor.exists():
        print(f"Error: receptor file not found: {args.receptor}", file=sys.stderr)
        return 1

    blocks = read_blocks(args.ligands)
    if not blocks:
        print(f"Error: no molecules found in {args.ligands}", file=sys.stderr)
        return 1
    print(f"Loaded {len(blocks)} poses from {args.ligands}")

    ref_mol = ref_block = None
    if needs_ref:
        ref_block = reference_block(args.reference)
        ref_mol = load_reference(args.reference)
        if ref_mol is None or ref_block is None:
            print(f"Error: could not read a reference ligand from {args.reference}",
                  file=sys.stderr)
            return 1

    workers = max(1, args.workers)
    columns: Dict[str, List[str]] = {}
    summary: List[str] = []

    if "ref" in metrics:
        values = calc_ref_sim(ref_mol, blocks, args.quiet)
        columns["Ref_Sim"] = [fmt(v) for v in values]
        summary.append(summarize("Ref_Sim", values))

    if "mcs" in metrics:
        values = calc_mcs_rmsd(ref_block, blocks, args.mcs_timeout, args.mcs_align,
                               workers, args.quiet)
        columns["MCS_RMSD"] = [fmt(v) for v in values]
        summary.append(summarize("MCS_RMSD", values))

    if "shape" in metrics:
        values = calc_shape_sim(ref_mol, blocks, args.shape_align, args.quiet)
        columns["Shape_Sim"] = [fmt(v) for v in values]
        summary.append(summarize("Shape_Sim", values))

    if "plif" in metrics:
        # Both backends want the reference as a file; materialize the parsed block
        # so a gzipped or multi-molecule reference works too.
        with tempfile.NamedTemporaryFile("w", suffix=".sdf", delete=False) as tmp:
            tmp.write(ref_block.rstrip() + "\n\n$$$$\n")
            ref_tmp = Path(tmp.name)
        try:
            values, backend = calc_plif_sim(args.receptor, ref_tmp, blocks,
                                            args.plif_backend, args.quiet)
            columns["PLIF_Sim"] = [fmt(v) for v in values]
            summary.append(summarize("PLIF_Sim", values) + f"  [{backend}]")
        except Exception as exc:
            print(f"Warning: PLIF_Sim skipped — {exc}", file=sys.stderr)
        finally:
            ref_tmp.unlink(missing_ok=True)

    pb_totals: Dict[str, int] = {}
    if "pb" in metrics:
        try:
            per_pose, pb_totals = calc_posebusters(
                args.receptor, blocks, args.bust_executable, max(1, args.pb_chunk_size),
                args.quiet, args.pb_report, args.id_field)
            columns["PB_Flags"] = ["N/A" if c is None else str(c) for c, _ in per_pose]
            columns["PB_Failed"] = [";".join(f) for _, f in per_pose]
            counts = [c for c, _ in per_pose if c is not None]
            clean = sum(1 for c in counts if c == 0)
            summary.append(f"  {'PB_Flags':<11} n={len(counts)}/{len(blocks)}  "
                           f"clean={clean}  flagged={len(counts) - clean}")
            if not counts:
                print(f"Warning: '{args.bust_executable}' evaluated no poses — check "
                      f"that PoseBusters runs in this environment, or pass "
                      f"--bust-executable /path/to/env/bin/bust", file=sys.stderr)
        except Exception as exc:
            print(f"Warning: PoseBusters skipped — {exc}", file=sys.stderr)

    stem = output_stem(args.output)

    if not args.no_sdf:
        annotated = []
        for i, block in enumerate(blocks):
            for key, values in columns.items():
                block = set_prop(block, key, values[i])
            annotated.append(block)
        write_sdf(Path(f"{stem}.sdf"), annotated)
        print(f"Wrote {len(annotated)} annotated poses → {stem}.sdf")

    if not args.no_csv:
        rows = []
        for i, block in enumerate(blocks):
            props = block_props(block)
            row = {"Name": block_name(block, i, props, args.id_field)}
            row.update({k: v for k, v in props.items() if k not in columns})
            row.update({key: values[i] for key, values in columns.items()})
            rows.append(row)
        write_csv(Path(f"{stem}.csv"), rows)
        print(f"Wrote {len(rows)} rows → {stem}.csv")

    if summary:
        print("\nSummary:")
        for line in summary:
            print(line)
    if pb_totals:
        print("\nPoseBusters failures by check:")
        for check, n in sorted(pb_totals.items(), key=lambda kv: -kv[1]):
            print(f"  {check:<40} {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
