#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Ligand Filter — remove molecules with unwanted structural features

Reads molecules from a SMILES (.smi / .csv / .tsv) or SDF file and writes
only the molecules that pass all enabled filters to the output file.
The output format matches the input format unless --output is given with an
explicit extension (.smi or .sdf).

Workflow
--------
1. Read input molecules (SMILES or SDF)
2. Preprocessing (optional, applied in order before filtering):
   --strip      strip salts / small fragments, keep the largest fragment
   --neutralize neutralize formal charges (e.g. carboxylate → acid,
                ammonium → amine) using RDKit MolStandardize Uncharger
   --unique     deduplicate on canonical SMILES; for duplicates keep the entry
                with the lexicographically smallest identifier
3. Apply each enabled filter in order; the first failure short-circuits
4. Write passing molecules to output; log reason for each rejection

Inputs
------
  SMILES file : first column = SMILES, optional second column = name
                (comma- or tab-separated, lines starting with '#' skipped)
  SDF file    : standard V2000/V3000 SD file

Outputs
-------
  Filtered SMILES / SDF file (same format as input by default)
  Summary printed to stdout

Dependencies
------------
  rdkit          pip install rdkit   (molecule I/O and SMARTS matching)

Written by Claude Sonnet 4.6, 2026-02-27
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from rich_argparse import RawDescriptionRichHelpFormatter as _HelpFmt
    _RICH = True
except ImportError:
    _HelpFmt = argparse.RawDescriptionHelpFormatter
    _RICH = False

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    _ARGCOMPLETE = True
except ImportError:
    _ARGCOMPLETE = False

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, QED
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _RDKIT = True
except ImportError:
    _RDKIT = False

# Default filter-file locations (same directory as this script)
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PAINS   = _SCRIPT_DIR / 'PAINS.txt'
_DEFAULT_REOS    = _SCRIPT_DIR / 'REOS.txt'
_DEFAULT_CUSTOM  = _SCRIPT_DIR / 'custom_filters.txt'


# ─── Logging helpers ──────────────────────────────────────────────────────────

def _ok(msg: str):   print(f"  ✓ {msg}")
def _info(msg: str): print(f"    {msg}")
def _warn(msg: str): print(f"  ⚠ {msg}")

def _fatal(msg: str):
    print(f"  ✗ {msg}", file=sys.stderr)
    sys.exit(1)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ─── Molecule I/O ─────────────────────────────────────────────────────────────

def _iter_smiles(path: Path) -> Iterator[Tuple[Chem.Mol, str]]:
    """Yield (mol, name) from a SMILES file (comma- or tab-separated)."""
    RDLogger.DisableLog('rdApp.*')
    with path.open(encoding='utf-8', errors='replace') as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', '\t').split('\t')
            smi = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else f"mol_{lineno}"
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                _warn(f"Line {lineno}: could not parse SMILES — skipped")
                continue
            mol.SetProp('_Name', name)
            yield mol, name


def _iter_sdf(path: Path) -> Iterator[Tuple[Chem.Mol, str]]:
    """Yield (mol, name) from an SDF file."""
    RDLogger.DisableLog('rdApp.*')
    sup = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for idx, mol in enumerate(sup):
        if mol is None:
            _warn(f"SDF entry {idx + 1}: could not parse — skipped")
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"mol_{idx + 1}"
        yield mol, name


def _iter_molecules(path: Path) -> Iterator[Tuple[Chem.Mol, str]]:
    """Auto-detect format from extension and yield (mol, name) pairs."""
    if path.suffix.lower() == '.sdf':
        yield from _iter_sdf(path)
    else:
        yield from _iter_smiles(path)


def _write_smiles(mol: Chem.Mol, name: str, fh) -> None:
    smi = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    fh.write(f"{smi}\t{name}\n")


def _make_writer(out_path: Path):
    """Return (writer_fn, close_fn) for the given output path."""
    if out_path.suffix.lower() == '.sdf':
        w = Chem.SDWriter(str(out_path))
        def _write(mol, name): w.write(mol)
        def _close():          w.close()
    else:
        fh = out_path.open('w', encoding='utf-8')
        def _write(mol, name): _write_smiles(mol, name, fh)
        def _close():          fh.close()
    return _write, _close


# ─── Preprocessing ────────────────────────────────────────────────────────────

def _largest_fragment(mol: Chem.Mol) -> Tuple[Chem.Mol, bool]:
    """Return (largest_fragment, was_stripped).

    Splits on disconnected fragments and returns the one with the most heavy
    atoms.  If the molecule is already a single fragment, returns it unchanged.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return mol, False
    largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    return largest, True


_uncharger = None  # lazy-initialised so it's only created when --neutralize is used

def _neutralize_mol(mol: Chem.Mol) -> Tuple[Chem.Mol, bool]:
    """Neutralize formal charges using RDKit's Uncharger.

    Returns (neutralized_mol, was_changed).  Quaternary nitrogens and other
    centres that cannot be neutralized without removing atoms are left intact.
    """
    global _uncharger
    if _uncharger is None:
        _uncharger = rdMolStandardize.Uncharger()
    uncharged = _uncharger.uncharge(mol)
    changed = Chem.MolToSmiles(uncharged) != Chem.MolToSmiles(mol)
    return uncharged, changed


def _preprocess(molecules: Iterator[Tuple[Chem.Mol, str]],
                do_strip: bool,
                do_neutralize: bool,
                do_unique: bool,
                ) -> Tuple[List[Tuple[Chem.Mol, str]], int, int, int]:
    """Apply salt stripping, neutralization, and/or deduplication (in that order).

    Returns (processed_list, n_stripped, n_neutralized, n_duplicates).
    n_stripped    = molecules where at least one fragment was removed
    n_neutralized = molecules where at least one formal charge was neutralized
    n_duplicates  = extra entries removed during deduplication
    """
    n_stripped = 0
    n_neutralized = 0
    n_duplicates = 0

    # seen maps canonical SMILES → (mol, name) for the representative entry
    seen: Dict[str, Tuple[Chem.Mol, str]] = {}
    result: List[Tuple[Chem.Mol, str]] = []

    for mol, name in molecules:
        # 1. Salt stripping
        if do_strip:
            mol, stripped = _largest_fragment(mol)
            if stripped:
                n_stripped += 1

        # 2. Neutralization
        if do_neutralize:
            mol, changed = _neutralize_mol(mol)
            if changed:
                n_neutralized += 1

        if do_unique:
            canon = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canon in seen:
                # Keep the entry with the lexicographically smallest identifier
                _, existing_name = seen[canon]
                if name < existing_name:
                    seen[canon] = (mol, name)
                n_duplicates += 1
            else:
                seen[canon] = (mol, name)
        else:
            result.append((mol, name))

    if do_unique:
        result = list(seen.values())

    return result, n_stripped, n_neutralized, n_duplicates


# ─── Range parsing ────────────────────────────────────────────────────────────

def _parse_range(s: str, name: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse a range string into (lo, hi), either of which may be None (unbounded).

    Accepted formats:
      '200:500'  →  200 ≤ x ≤ 500
      '200:'     →  x ≥ 200
      ':500'     →  x ≤ 500
      '300'      →  x == 300  (exact value, lo == hi)
    """
    if ':' in s:
        lo_s, hi_s = s.split(':', 1)
        try:
            lo = float(lo_s) if lo_s.strip() else None
            hi = float(hi_s) if hi_s.strip() else None
        except ValueError:
            _fatal(f"--{name}: could not parse range '{s}' — expected format MIN:MAX")
    else:
        try:
            lo = hi = float(s)
        except ValueError:
            _fatal(f"--{name}: could not parse value '{s}'")
    return lo, hi


# ─── Filter loaders ───────────────────────────────────────────────────────────

def _load_pains(path: Path) -> List[Tuple[str, Chem.Mol]]:
    """Load PAINS patterns from a tab-separated file (name\\tSMARTS).

    Returns a list of (name, query_mol) tuples.  Patterns that cannot be
    compiled are warned and skipped.
    """
    patterns = []
    with path.open(encoding='utf-8', errors='replace') as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                _warn(f"PAINS line {lineno}: expected name\\tSMARTS — skipped")
                continue
            name, smarts = parts[0].strip(), parts[1].strip()
            q = Chem.MolFromSmarts(smarts)
            if q is None:
                _warn(f"PAINS line {lineno}: could not compile SMARTS '{smarts}' — skipped")
                continue
            patterns.append((name, q))
    return patterns


def _load_reos(path: Path) -> List[Tuple[str, int, str, Chem.Mol]]:
    """Load REOS rules from a tab-separated file (SMARTS\\tmax_count\\tdescription).

    max_count = 0  → reject if any match found
    max_count = N  → reject if more than N matches found

    Returns a list of (smarts_str, max_count, description, query_mol).
    """
    rules = []
    with path.open(encoding='utf-8', errors='replace') as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                _warn(f"REOS line {lineno}: expected SMARTS\\tmax_count\\tdescription — skipped")
                continue
            smarts = parts[0].strip()
            desc   = parts[2].strip().strip('"')
            try:
                max_count = int(parts[1].strip())
            except ValueError:
                _warn(f"REOS line {lineno}: non-integer max_count — skipped")
                continue
            q = Chem.MolFromSmarts(smarts)
            if q is None:
                _warn(f"REOS line {lineno}: could not compile SMARTS '{smarts}' — skipped")
                continue
            rules.append((smarts, max_count, desc, q))
    return rules


# ─── Filters ──────────────────────────────────────────────────────────────────
# Each filter is a callable:  filter(mol, args) -> Optional[str]
#   returns None  if the molecule PASSES
#   returns a str (reason) if the molecule FAILS
#
# Add new filters here as named functions, then register them in
# _build_filter_pipeline() below.
# ──────────────────────────────────────────────────────────────────────────────

def _make_pains_filter(patterns: List[Tuple[str, Chem.Mol]]):
    """Return a PAINS filter function closed over the compiled patterns."""
    def _filt(mol, _args):
        for name, q in patterns:
            if mol.HasSubstructMatch(q):
                return f"PAINS: {name}"
        return None
    return _filt


def _make_reos_filter(rules: List[Tuple[str, int, str, Chem.Mol]]):
    """Return a REOS filter function closed over the compiled rules."""
    def _filt(mol, _args):
        for _smarts, max_count, desc, q in rules:
            n = len(mol.GetSubstructMatches(q))
            if n > max_count:
                return f"REOS: {desc}"
        return None
    return _filt


def _make_custom_filter(rules: List[Tuple[str, int, str, Chem.Mol]]):
    """Return a custom filter function; reuses the REOS rule structure."""
    def _filt(mol, _args):
        for _smarts, max_count, desc, q in rules:
            n = len(mol.GetSubstructMatches(q))
            if n > max_count:
                return f"Custom: {desc}"
        return None
    return _filt


def _make_mw_filter(lo: Optional[float], hi: Optional[float]):
    """Return a molecular-weight filter (average MW via RDKit Descriptors)."""
    def _filt(mol, _args):
        mw = Descriptors.MolWt(mol)
        if lo is not None and mw < lo:
            return f"MW {mw:.1f} < {lo}"
        if hi is not None and mw > hi:
            return f"MW {mw:.1f} > {hi}"
        return None
    return _filt


def _make_logp_filter(lo: Optional[float], hi: Optional[float]):
    """Return a Wildman-Crippen logP filter."""
    def _filt(mol, _args):
        lp = Crippen.MolLogP(mol)
        if lo is not None and lp < lo:
            return f"LogP {lp:.2f} < {lo}"
        if hi is not None and lp > hi:
            return f"LogP {lp:.2f} > {hi}"
        return None
    return _filt


def _make_lipinski_filter():
    """Return a Lipinski Rule-of-Five filter.

    Rejects molecules that violate more than one of:
      MW  ≤ 500
      HBD ≤ 5   (H-bond donors,    NH + OH)
      HBA ≤ 10  (H-bond acceptors, N + O)
      logP ≤ 5

    One violation is permitted (the classic Ro5 definition allows one
    exception for molecules that are substrates of active transporters).
    Pass --lipinski-strict to require all four rules.
    """
    def _filt(mol, args):
        violations = []
        mw  = Descriptors.MolWt(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        lp  = Crippen.MolLogP(mol)
        if mw  > 500: violations.append(f"MW {mw:.1f}>500")
        if hbd > 5:   violations.append(f"HBD {hbd}>5")
        if hba > 10:  violations.append(f"HBA {hba}>10")
        if lp  > 5:   violations.append(f"logP {lp:.2f}>5")
        limit = 0 if getattr(args, 'lipinski_strict', False) else 1
        if len(violations) > limit:
            return "Lipinski: " + ", ".join(violations)
        return None
    return _filt


def _make_ro3_filter():
    """Return an Astex Rule-of-Three filter for fragment screening.

    Rejects molecules that violate any of:
      MW       ≤ 300
      logP     ≤ 3
      HBD      ≤ 3  (H-bond donors)
      HBA      ≤ 3  (H-bond acceptors)
      RotBonds ≤ 3  (rotatable bonds)
    """
    def _filt(mol, _args):
        violations = []
        mw   = Descriptors.MolWt(mol)
        lp   = Crippen.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if mw   > 300: violations.append(f"MW {mw:.1f}>300")
        if lp   > 3:   violations.append(f"logP {lp:.2f}>3")
        if hbd  > 3:   violations.append(f"HBD {hbd}>3")
        if hba  > 3:   violations.append(f"HBA {hba}>3")
        if rotb > 3:   violations.append(f"RotBonds {rotb}>3")
        if violations:
            return "Ro3: " + ", ".join(violations)
        return None
    return _filt


def _make_qed_filter(lo: Optional[float], hi: Optional[float]):
    """Return a QED (Quantitative Estimate of Drug-likeness) filter.

    QED ranges from 0 (least drug-like) to 1 (most drug-like).
    Typical drug-like threshold: QED ≥ 0.5.
    """
    def _filt(mol, _args):
        score = QED.qed(mol)
        if lo is not None and score < lo:
            return f"QED {score:.3f} < {lo}"
        if hi is not None and score > hi:
            return f"QED {score:.3f} > {hi}"
        return None
    return _filt


def _make_chiral_filter(lo: Optional[float], hi: Optional[float]):
    """Return a chiral-centre count filter (specified + unspecified stereocenters)."""
    def _filt(mol, _args):
        n = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        if lo is not None and n < lo:
            return f"Chiral {n} < {int(lo)}"
        if hi is not None and n > hi:
            return f"Chiral {n} > {int(hi)}"
        return None
    return _filt


def _make_tpsa_filter(lo: Optional[float], hi: Optional[float]):
    """Return a topological polar surface area filter (Å²)."""
    def _filt(mol, _args):
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        if lo is not None and tpsa < lo:
            return f"TPSA {tpsa:.1f} < {lo}"
        if hi is not None and tpsa > hi:
            return f"TPSA {tpsa:.1f} > {hi}"
        return None
    return _filt


def _make_hba_filter(lo: Optional[float], hi: Optional[float]):
    """Return an H-bond acceptor count filter."""
    def _filt(mol, _args):
        n = rdMolDescriptors.CalcNumHBA(mol)
        if lo is not None and n < lo:
            return f"HBA {n} < {int(lo)}"
        if hi is not None and n > hi:
            return f"HBA {n} > {int(hi)}"
        return None
    return _filt


def _make_hbd_filter(lo: Optional[float], hi: Optional[float]):
    """Return an H-bond donor count filter."""
    def _filt(mol, _args):
        n = rdMolDescriptors.CalcNumHBD(mol)
        if lo is not None and n < lo:
            return f"HBD {n} < {int(lo)}"
        if hi is not None and n > hi:
            return f"HBD {n} > {int(hi)}"
        return None
    return _filt


def _make_rb_filter(lo: Optional[float], hi: Optional[float]):
    """Return a rotatable bond count filter."""
    def _filt(mol, _args):
        n = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if lo is not None and n < lo:
            return f"RotBonds {n} < {int(lo)}"
        if hi is not None and n > hi:
            return f"RotBonds {n} > {int(hi)}"
        return None
    return _filt


def _build_filter_pipeline(args, pains_patterns=None, reos_rules=None,
                           custom_rules=None, mw_range=None, logp_range=None,
                           lipinski=False, ro3=False, qed_range=None,
                           hba_range=None, hbd_range=None, rb_range=None,
                           tpsa_range=None, chiral_range=None) -> list:
    """Return an ordered list of (label, filter_fn) tuples to apply."""
    pipeline = []
    if pains_patterns is not None:
        pipeline.append(('PAINS',   _make_pains_filter(pains_patterns)))
    if reos_rules is not None:
        pipeline.append(('REOS',    _make_reos_filter(reos_rules)))
    if custom_rules is not None:
        pipeline.append(('Custom',  _make_custom_filter(custom_rules)))
    if mw_range is not None:
        pipeline.append(('MW',      _make_mw_filter(*mw_range)))
    if logp_range is not None:
        pipeline.append(('LogP',    _make_logp_filter(*logp_range)))
    if lipinski:
        pipeline.append(('Lipinski', _make_lipinski_filter()))
    if ro3:
        pipeline.append(('Ro3',      _make_ro3_filter()))
    if qed_range is not None:
        pipeline.append(('QED',      _make_qed_filter(*qed_range)))
    if hba_range is not None:
        pipeline.append(('HBA',      _make_hba_filter(*hba_range)))
    if hbd_range is not None:
        pipeline.append(('HBD',      _make_hbd_filter(*hbd_range)))
    if rb_range is not None:
        pipeline.append(('RotBonds', _make_rb_filter(*rb_range)))
    if tpsa_range is not None:
        pipeline.append(('TPSA',     _make_tpsa_filter(*tpsa_range)))
    if chiral_range is not None:
        pipeline.append(('Chiral',   _make_chiral_filter(*chiral_range)))
    return pipeline


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not _RDKIT:
        _fatal("RDKit is required:  conda install -c conda-forge rdkit")

    p = argparse.ArgumentParser(
        prog='ligfilter.py',
        description=__doc__,
        formatter_class=_HelpFmt,
    )

    io = p.add_argument_group('Input / Output')
    io.add_argument('input', metavar='FILE',
                    help='Input SMILES (.smi/.csv/.tsv) or SDF file')
    io.add_argument('-o', '--output', metavar='FILE',
                    help='Output file (default: <input>_filtered.<ext>)')

    pre = p.add_argument_group('Preprocessing')
    pre.add_argument('--strip', action='store_true',
                     help='Strip salts and small fragments; keep the largest '
                          'fragment by heavy-atom count')
    pre.add_argument('--neutralize', action='store_true',
                     help='Neutralize formal charges where possible '
                          '(e.g. carboxylate → carboxylic acid, ammonium → amine) '
                          'using RDKit MolStandardize.  Applied after --strip, '
                          'before --unique.  Quaternary N and other centres that '
                          'cannot be neutralized without atom removal are left intact.')
    pre.add_argument('--unique', action='store_true',
                     help='Deduplicate on canonical SMILES; for duplicates '
                          'keep the entry with the lexicographically smallest '
                          'identifier')

    filt = p.add_argument_group('Filters')
    filt.add_argument('--pains', action='store_true',
                      help=f'Reject PAINS (Pan-Assay INterference compoundS); '
                           f'uses {_DEFAULT_PAINS.name} from the script directory')
    filt.add_argument('--pains-file', metavar='FILE', default=None,
                      help='Custom PAINS file (implies --pains; '
                           'format: name<TAB>SMARTS)')
    filt.add_argument('--reos', action='store_true',
                      help=f'Reject REOS (Rapid Elimination Of Swill) unwanted '
                           f'functional groups; uses {_DEFAULT_REOS.name}')
    filt.add_argument('--reos-file', metavar='FILE', default=None,
                      help='Custom REOS file (implies --reos; '
                           'format: SMARTS<TAB>max_count<TAB>description)')
    filt.add_argument('--custom', action='store_true',
                      help=f'Apply custom SMARTS filters from '
                           f'{_DEFAULT_CUSTOM.name} in the script directory '
                           f'(same format as REOS: SMARTS<TAB>max_count<TAB>description)')
    filt.add_argument('--custom-file', metavar='FILE', default=None,
                      help='Custom filter file (implies --custom; '
                           'format: SMARTS<TAB>max_count<TAB>description)')

    prop = p.add_argument_group('Property ranges')
    prop.add_argument('--mw', metavar='RANGE', default=None,
                      help='Molecular weight range (average MW).  '
                           'Format: MIN:MAX, MIN:, :MAX, or exact value.  '
                           'E.g. --mw 150:500  --mw :600  --mw 250:')
    prop.add_argument('--logp', metavar='RANGE', default=None,
                      help='Wildman-Crippen logP range.  '
                           'Format: MIN:MAX, MIN:, :MAX, or exact value.  '
                           'E.g. --logp -2:5  --logp :4.5')
    prop.add_argument('--lipinski', action='store_true',
                      help='Reject molecules that violate more than one '
                           'Lipinski Rule of Five (MW≤500, HBD≤5, HBA≤10, '
                           'logP≤5).  One violation is permitted by default.')
    prop.add_argument('--lipinski-strict', action='store_true',
                      help='Require all four Lipinski rules to pass '
                           '(zero violations allowed; implies --lipinski)')
    prop.add_argument('--qed', metavar='RANGE', default=None,
                      help='Quantitative Estimate of Drug-likeness (0–1, '
                           'higher = more drug-like).  '
                           'Format: MIN:MAX, MIN:, :MAX, or exact value.  '
                           'E.g. --qed 0.5:  --qed 0.4:0.9')
    prop.add_argument('--chiral', metavar='RANGE', default=None,
                      help='Chiral centre count range (specified + unspecified).  '
                           'E.g. --chiral :3  --chiral 0:2')
    prop.add_argument('--tpsa', metavar='RANGE', default=None,
                      help='Topological polar surface area range (Å²).  '
                           'E.g. --tpsa :140  --tpsa 40:130')
    prop.add_argument('--hba', metavar='RANGE', default=None,
                      help='H-bond acceptor count range.  '
                           'E.g. --hba :10  --hba 1:8')
    prop.add_argument('--hbd', metavar='RANGE', default=None,
                      help='H-bond donor count range.  '
                           'E.g. --hbd :5  --hbd 1:3')
    prop.add_argument('--rb', metavar='RANGE', default=None,
                      help='Rotatable bond count range.  '
                           'E.g. --rb :10  --rb 2:8')
    prop.add_argument('--ro3', action='store_true',
                      help='Astex Rule of Three for fragment screening: '
                           'MW≤300, logP≤3, HBD≤3, HBA≤3, RotBonds≤3 '
                           '(all rules must pass)')

    if _ARGCOMPLETE:
        argcomplete.autocomplete(p)

    args = p.parse_args()

    t0 = time.time()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        _fatal(f"Input file not found: {in_path}")

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = in_path.with_name(in_path.stem + '_filtered' + in_path.suffix)

    # ── Config ────────────────────────────────────────────────────────────────
    bar = '─' * 60
    print(bar)
    print("  ligfilter.py — Ligand structural filter")
    print(bar)
    _info(f"Input:         {in_path.name}")
    _info(f"Output:        {out_path.name}")
    _info(f"Strip salts:   {'yes' if args.strip else 'no'}")
    _info(f"Neutralize:    {'yes' if args.neutralize else 'no'}")
    _info(f"Deduplicate:   {'yes' if args.unique else 'no'}")
    _info(f"MW range:      {args.mw if args.mw else 'any'}")
    _info(f"LogP range:    {args.logp if args.logp else 'any'}")
    _info(f"QED range:     {args.qed if args.qed else 'any'}")
    _info(f"HBA range:     {args.hba if args.hba else 'any'}")
    _info(f"HBD range:     {args.hbd if args.hbd else 'any'}")
    _info(f"RotBonds range:{args.rb   if args.rb   else 'any'}")
    _info(f"TPSA range:    {args.tpsa   if args.tpsa   else 'any'}")
    _info(f"Chiral range:  {args.chiral if args.chiral else 'any'}")
    lip_mode = ('strict' if args.lipinski_strict else 'on (1 violation allowed)') if (args.lipinski or args.lipinski_strict) else 'off'
    _info(f"Lipinski Ro5:  {lip_mode}")
    _info(f"Rule of Three: {'on' if args.ro3 else 'off'}")
    print(bar)

    # ── Load filter data ──────────────────────────────────────────────────────
    pains_patterns = None
    reos_rules     = None

    if args.pains or args.pains_file:
        pains_path = Path(args.pains_file).resolve() if args.pains_file else _DEFAULT_PAINS
        if not pains_path.exists():
            _fatal(f"PAINS file not found: {pains_path}")
        pains_patterns = _load_pains(pains_path)
        _ok(f"PAINS:  {len(pains_patterns)} patterns loaded  ({pains_path.name})")

    if args.reos or args.reos_file:
        reos_path = Path(args.reos_file).resolve() if args.reos_file else _DEFAULT_REOS
        if not reos_path.exists():
            _fatal(f"REOS file not found: {reos_path}")
        reos_rules = _load_reos(reos_path)
        _ok(f"REOS:   {len(reos_rules)} rules loaded  ({reos_path.name})")

    custom_rules = None
    if args.custom or args.custom_file:
        custom_path = Path(args.custom_file).resolve() if args.custom_file else _DEFAULT_CUSTOM
        if not custom_path.exists():
            _fatal(f"Custom filter file not found: {custom_path}")
        custom_rules = _load_reos(custom_path)   # same format as REOS
        _ok(f"Custom: {len(custom_rules)} rules loaded  ({custom_path.name})")

    mw_range   = _parse_range(args.mw,   'mw')   if args.mw   else None
    logp_range = _parse_range(args.logp, 'logp') if args.logp else None
    qed_range  = _parse_range(args.qed,  'qed')  if args.qed  else None
    hba_range  = _parse_range(args.hba,  'hba')  if args.hba  else None
    hbd_range  = _parse_range(args.hbd,  'hbd')  if args.hbd  else None
    rb_range   = _parse_range(args.rb,   'rb')   if args.rb   else None
    tpsa_range   = _parse_range(args.tpsa,   'tpsa')   if args.tpsa   else None
    chiral_range = _parse_range(args.chiral, 'chiral') if args.chiral else None

    pipeline = _build_filter_pipeline(args,
                                      pains_patterns=pains_patterns,
                                      reos_rules=reos_rules,
                                      custom_rules=custom_rules,
                                      mw_range=mw_range,
                                      logp_range=logp_range,
                                      lipinski=args.lipinski or args.lipinski_strict,
                                      ro3=args.ro3,
                                      qed_range=qed_range,
                                      hba_range=hba_range,
                                      hbd_range=hbd_range,
                                      rb_range=rb_range,
                                      tpsa_range=tpsa_range,
                                      chiral_range=chiral_range)
    if not pipeline and not args.strip and not args.neutralize and not args.unique:
        _warn("No preprocessing or filters enabled — all molecules will pass.")
    elif pipeline:
        _info(f"Active filters ({len(pipeline)}): "
              + ', '.join(label for label, _ in pipeline))
    print(bar)

    # ── Read & preprocess ─────────────────────────────────────────────────────
    raw_stream = _iter_molecules(in_path)

    if args.strip or args.neutralize or args.unique:
        molecules, n_stripped, n_neutralized, n_duplicates = _preprocess(
            raw_stream, do_strip=args.strip, do_neutralize=args.neutralize,
            do_unique=args.unique)
        n_read = len(molecules) + n_duplicates
    else:
        molecules = list(raw_stream)
        n_read = len(molecules)
        n_stripped = n_neutralized = n_duplicates = 0

    # ── Filter ────────────────────────────────────────────────────────────────
    n_pass = n_fail = 0
    rejection_counts: dict = {}

    write_mol, close_out = _make_writer(out_path)

    for mol, name in molecules:
        reason = None
        for _label, filt_fn in pipeline:
            reason = filt_fn(mol, args)
            if reason:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                break
        if reason is None:
            write_mol(mol, name)
            n_pass += 1
        else:
            n_fail += 1

    close_out()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(bar)
    print("  [SUMMARY]")
    _info(f"Molecules read:    {n_read}")
    if args.strip:
        _info(f"Salts stripped:    {n_stripped}")
    if args.neutralize:
        _info(f"Neutralized:       {n_neutralized}")
    if args.unique:
        _info(f"Duplicates removed:{n_duplicates}")
    _info(f"Passed filters:    {n_pass}")
    _info(f"Rejected:          {n_fail}")
    if rejection_counts:
        _info("Rejection reasons:")
        for reason, count in sorted(rejection_counts.items(),
                                    key=lambda x: -x[1]):
            _info(f"  {count:>6}  {reason}")
    _info(f"Time:              {format_time(elapsed)}")
    print(bar)


if __name__ == '__main__':
    main()
