#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Molecule Preparation Script for Docking

This script takes molecules as input from either an SDF or a SMILES file and prepares them for docking by:
- Stripping salts and keeping the largest fragment
- Enumerating stereoisomers for unspecified chiral centers (optional)
- Protonating molecules at pH 7.4 using OpenBabel
- Adding explicit hydrogens
- Generating 3D conformations
- Minimizing geometry using the MMFF94s force field
- Preserving specified molecule identifier properties throughout processing
- Writing the processed molecules to an output SDF file

The script supports parallel processing with user-defined CPU count for faster preparation of large molecule libraries.
It reports total and average processing time, formatted for easy reading.

SMILES input files must have the SMILES string in the first column, followed optionally by an identifier (second column).
The file can be comma-separated or tab-separated.

Written by Perplexity 4.0.0, 2025-08-19
Updated by Claude, 2026-01-24: Added argparse and stereoisomer enumeration
Updated by Claude, 2026-02-28: Batch obabel calls, per-isomer task batching, streaming writer, default ncpus=all
"""

import argparse
import concurrent.futures
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time

try:
    from rich_argparse import RawDescriptionRichHelpFormatter as _HelpFmt
except ImportError:
    _HelpFmt = argparse.RawDescriptionHelpFormatter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    _ARGCOMPLETE = True
except ImportError:
    _ARGCOMPLETE = False

# Prefer the system (apt) obabel 3.1.1+ over the conda 3.1.0 build.
# The conda 3.1.0 phmodel.txt tetrazole TRANSFORM produces an invalid mol
# block (explicit H retained alongside N⁻ formal charge) that RDKit cannot
# parse, causing a silent fallback to the neutral protonation state.
# The apt 3.1.1 correctly removes the H and sets the formal charge.
_OBABEL_CANDIDATES = ['/usr/local/bin/obabel', '/usr/bin/obabel', shutil.which('obabel')]
_OBABEL = next((p for p in _OBABEL_CANDIDATES if p and os.path.isfile(p)), 'obabel')

_XTBBIN = shutil.which('xtb')

# Number of isomers per batch submitted to the process pool.
# Embed+minimize dominates, so equal-sized batches give good load balance.
_BATCH_SIZE = 50

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdmolfiles import SDWriter
from rdkit.Geometry import rdGeometry as _Geometry


_RING_TEMPLATES = None  # lazy-initialised once per worker process


def _init_ring_templates():
    """Build 3D chair-geometry templates for common 6-membered drug-like rings.

    Called once per worker process on first use.  Each template is a
    (query_mol, [Point3D, ...]) pair: query_mol is used for substructure
    matching and the Point3D list holds the corresponding ideal chair
    coordinates (one per heavy atom in the query).

    Templates cover the rings most commonly found in drug-like molecules:
    cyclohexane, piperidine, piperazine, morpholine, thiomorpholine,
    tetrahydropyran, tetrahydrothiopyran, piperidin-2-one, piperazin-2-one.
    """
    global _RING_TEMPLATES

    _RING_SMILES = [
        'C1CCCCC1',    # cyclohexane
        'C1CCNCC1',    # piperidine
        'C1CNCCN1',    # piperazine
        'C1COCCN1',    # morpholine
        'C1CSCCN1',    # thiomorpholine
        'C1CCOCC1',    # tetrahydropyran
        'C1CCSCC1',    # tetrahydrothiopyran
        'O=C1CCCCN1',  # piperidin-2-one (delta-lactam)
        'O=C1CNCCN1',  # piperazin-2-one
    ]

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d
    params.useSmallRingTorsions = True

    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)

    templates = []
    for smi in _RING_SMILES:
        query = Chem.MolFromSmiles(smi)
        if query is None:
            continue
        mol_h = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(mol_h, params) < 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant='MMFF94s')
        mol_3d = Chem.RemoveHs(mol_h)
        conf = mol_3d.GetConformer()
        coords = [
            _Geometry.Point3D(conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z)
            for i in range(mol_3d.GetNumAtoms())
        ]
        templates.append((query, coords))

    logger.setLevel(RDLogger.WARNING)
    _RING_TEMPLATES = templates


def _get_ring_coord_map(mol):
    """Return a coordMap dict seeding common 6-membered rings to chair templates.

    Matches the ring template library against mol (which should have explicit
    Hs via AddHs).  For each non-overlapping match, the heavy-atom indices in
    mol are mapped to the corresponding ideal chair coordinates.  Overlapping
    matches (e.g. fused rings sharing atoms) are skipped.

    Returns {} if no templates match (acyclic molecules, aromatic-only, etc.).
    """
    if _RING_TEMPLATES is None:
        _init_ring_templates()
    if not _RING_TEMPLATES:
        return {}

    used_atoms = set()
    coord_map = {}
    for query, coords in _RING_TEMPLATES:
        for match in mol.GetSubstructMatches(query):
            if any(idx in used_atoms for idx in match):
                continue  # skip overlapping matches (e.g. fused bicyclics)
            for i, mol_idx in enumerate(match):
                coord_map[mol_idx] = coords[i]
            used_atoms.update(match)
    return coord_map


def _parse_col_spec(s: str):
    """Parse a column specifier: positive integer string → 0-based int; else column name str."""
    try:
        n = int(s)
        if n < 1:
            print(f"ERROR: column index must be ≥ 1, got {n}", file=sys.stderr)
            sys.exit(1)
        return n - 1  # convert 1-based user input to 0-based
    except ValueError:
        return s  # column header name


def obabel_protonate_batch(smiles_list, ph=7.4):
    """Batch-protonate SMILES at the given pH using a single obabel subprocess.

    Uses SMILES output (-osmi) so that stereochemistry is preserved directly
    in the SMILES string — no 0D mol block, no atom-index re-attachment needed.

    Each SMILES is tagged with a synthetic name ``__idx_N`` so the output
    lines can be mapped back to their original positions.

    Returns list[Mol | None] in the same order as smiles_list.
    None entries indicate protonation failures for those positions.
    On subprocess failure (missing binary, timeout) returns all-None list.
    """
    if not smiles_list:
        return []

    stdin_data = "".join(f"{smi}\t__idx_{i}\n" for i, smi in enumerate(smiles_list))
    timeout = 30 + 5 * len(smiles_list)

    try:
        result = subprocess.run(
            [_OBABEL, '-ismi', '-osmi', f'-p{ph}'],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [None] * len(smiles_list)

    if not result.stdout.strip():
        return [None] * len(smiles_list)

    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    output = [None] * len(smiles_list)
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        smi, name = parts[0], parts[1]
        if not name.startswith('__idx_'):
            continue
        try:
            idx = int(name[6:])
        except ValueError:
            continue
        if not (0 <= idx < len(output)):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            output[idx] = mol
    logger.setLevel(RDLogger.WARNING)
    return output


def strip_salts_keep_largest(mol):
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumAtoms())


def enumerate_stereoisomers(mol, max_isomers=32):
    """
    Enumerate all stereoisomers for unspecified chiral centers.

    Args:
        mol: RDKit molecule with potentially unspecified stereocenters
        max_isomers: Maximum number of isomers to generate (default 32)

    Returns:
        List of RDKit molecules representing all unique stereoisomers
    """
    opts = StereoEnumerationOptions(
        tryEmbedding=False,
        unique=True,
        onlyUnassigned=True,
        maxIsomers=max_isomers
    )
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    return isomers if isomers else [mol]


def get_stereo_suffix(mol, index):
    """Generate a suffix indicating stereochemistry (e.g., '_R,S' or '_isomer_1')."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    stereo_labels = [
        atom.GetProp('_CIPCode')
        for atom in mol.GetAtoms()
        if atom.HasProp('_CIPCode')
    ]
    return f"_{''.join(stereo_labels)}" if stereo_labels else f"_isomer_{index}"


def _xtb_refine(mol, xtb_level, env):
    """Run xTB geometry optimisation on mol (with explicit Hs) in implicit water.

    The single conformer's coordinates are updated in-place on success.
    On any failure (missing binary, crash, atom-count mismatch) the MMFF94s
    geometry is silently kept.

    Args:
        mol:       RDKit mol with explicit Hs and exactly one conformer.
        xtb_level: 'gfnff', 'gfn1', or 'gfn2'.
        env:       subprocess environment dict (should have OMP_NUM_THREADS=1).
    """
    chrg = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    level_args = ['--gfnff'] if xtb_level == 'gfnff' else ['--gfn', xtb_level[3:]]

    with tempfile.TemporaryDirectory() as td:
        sdf_in = os.path.join(td, 'mol.sdf')
        w = Chem.SDWriter(sdf_in)
        w.write(mol)
        w.close()

        try:
            subprocess.run(
                [_XTBBIN, 'mol.sdf', '--opt', '--alpb', 'water']
                + level_args + ['--chrg', str(chrg)],
                cwd=td, capture_output=True, text=True, timeout=300, env=env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return

        opt_sdf = os.path.join(td, 'xtbopt.sdf')
        if not os.path.exists(opt_sdf):
            return

        logger = RDLogger.logger()
        logger.setLevel(RDLogger.CRITICAL)
        suppl = Chem.SDMolSupplier(opt_sdf, removeHs=False)
        opt_mol = next(iter(suppl), None)
        logger.setLevel(RDLogger.WARNING)

        if opt_mol is None or opt_mol.GetNumAtoms() != mol.GetNumAtoms():
            return

        # Update conformer coordinates in-place; all mol properties are kept.
        conf = mol.GetConformer()
        opt_conf = opt_mol.GetConformer()
        for j in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(j, opt_conf.GetAtomPosition(j))


def _prune_by_rmsd(mol, cids_by_energy, threshold):
    """Greedy RMSD pruning: always keep the lowest-energy conformer, then each
    subsequent one only if its RMSD from all already-kept conformers is >= threshold.

    Args:
        mol:             RDKit mol with multiple conformers.
        cids_by_energy:  Conformer IDs sorted lowest-energy first.
        threshold:       Minimum RMSD (Å) required to keep a conformer.

    Returns:
        List of kept conformer IDs (subset of cids_by_energy).
    """
    kept = [cids_by_energy[0]]
    for cid in cids_by_energy[1:]:
        if all(AllChem.GetConformerRMS(mol, k, cid, prealigned=False) >= threshold
               for k in kept):
            kept.append(cid)
    return kept


def _prepare_isomer_batch(batch):
    """Worker: prepare a batch of isomers with one shared obabel call.

    Args:
        batch: list of (mol_binary, base_name, assigned_name,
                        ph, id_col, num_confs, xtb_level,
                        keep_confs, energy_window, rmsd_threshold)
               keep_confs: 'best', 'all', or int N.
               xtb_level is None to skip xTB, or 'gfnff'/'gfn1'/'gfn2'.

    Returns:
        list of (mol_binary_or_None, base_name, assigned_name)
        mol_binary is None when 3D embedding failed.
        In multi-conf mode (keep_confs != 'best') each isomer may produce
        multiple entries, one per kept conformer.
    """
    if not batch:
        return []

    ph             = batch[0][3]
    num_confs      = batch[0][5]
    xtb_level      = batch[0][6]
    keep_confs     = batch[0][7]
    energy_window  = batch[0][8]
    rmsd_threshold = batch[0][9]

    xtb_env = None
    if xtb_level:
        xtb_env = {**os.environ, 'OMP_NUM_THREADS': '1'}

    # Step 1: deserialise all mols and build SMILES list
    mols_in = [Chem.Mol(item[0]) for item in batch]
    smiles_list = [Chem.MolToSmiles(m, isomericSmiles=True) for m in mols_in]

    # Step 2: single obabel subprocess for the whole batch
    protonated = obabel_protonate_batch(smiles_list, ph)

    # Pickle flags that preserve mol-level and private (e.g. _Name) properties.
    # ToBinary() defaults to NoProps, which silently strips all SD tags.
    _PICKLE_PROPS = (Chem.PropertyPickleOptions.MolProps |
                     Chem.PropertyPickleOptions.PrivateProps)

    # ETKDGv3 + useSmallRingTorsions — CSD-derived torsion prefs for 3-8-membered
    # rings; this flag is False by default in ETKDGv3 but is critical for obtaining
    # chair (rather than twisted-boat) conformations of piperazines and cyclohexanes.
    _etkdg = AllChem.ETKDGv3()
    _etkdg.randomSeed = 0xf00d
    _etkdg.useSmallRingTorsions = True

    # Step 3: embed and minimize each isomer
    results = []
    for i, (mol_binary, base_name, assigned_name, ph, id_col, num_confs, _xtb_level,
            _keep_confs, _energy_window, _rmsd_threshold) in enumerate(batch):
        obabel_failed = protonated[i] is None
        if obabel_failed:
            print(
                f"  [WARNING] obabel protonation failed for {assigned_name or '(unknown)'}; "
                "using input protonation state",
                file=sys.stderr,
            )
            prot = mols_in[i]
        else:
            prot = protonated[i]

        # Set name and identity tags.
        prot.SetProp('_Name', assigned_name)
        if base_name:
            prot.SetProp(id_col, base_name)
            if assigned_name != base_name:
                prot.SetProp(f"{id_col}_stereo", assigned_name)

        prot = Chem.AddHs(prot)

        # Fallback chain for 3D embedding:
        #   1. ETKDGv3 + chair templates (coordMap) — for molecules with common
        #      6-membered rings; seeds DG from pre-computed chair coordinates to
        #      avoid boat/twist-boat ring geometries
        #   2. ETKDGv3 standard                     — general case / no template
        #   3. ETKDGv3 + randomCoords=True           — bypasses DG for difficult
        #      ring systems (macrocycles, bridged polycyclics)
        coord_map = _get_ring_coord_map(prot)

        if coord_map:
            prot.RemoveAllConformers()
            cids = []
            for _ci in range(num_confs):
                _p = AllChem.ETKDGv3()
                _p.randomSeed = 0xf00d + _ci
                _p.useSmallRingTorsions = True
                try:
                    cid = AllChem.EmbedMolecule(prot, params=_p, clearConfs=False,
                                                coordMap=coord_map)
                except (RuntimeError, TypeError):
                    cid = -1
                if cid >= 0:
                    cids.append(cid)
        else:
            cids = []

        if not cids:
            try:
                cids = list(AllChem.EmbedMultipleConfs(prot, numConfs=num_confs, params=_etkdg))
            except RuntimeError:
                cids = []

        if not cids:
            _etkdg_rc = AllChem.ETKDGv3()
            _etkdg_rc.randomSeed = 0xf00d
            _etkdg_rc.useSmallRingTorsions = True
            _etkdg_rc.useRandomCoords = True
            try:
                cids = list(AllChem.EmbedMultipleConfs(prot, numConfs=num_confs, params=_etkdg_rc))
            except RuntimeError:
                cids = []

        if not cids:
            # Final fallback: strip all stereo constraints and embed without them.
            # Catches enumerated isomers whose stereo configuration is geometrically
            # impossible (e.g. bridgehead stereocenters, strained ring fusions).
            prot_ns = Chem.RWMol(prot)
            for atom in prot_ns.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            for bond in prot_ns.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREONONE)
            try:
                cids = AllChem.EmbedMultipleConfs(prot_ns, numConfs=num_confs, params=_etkdg)
            except RuntimeError:
                cids = []
            if cids:
                prot = prot_ns.GetMol()

        if not cids:
            label = assigned_name or smiles_list[i]
            print(
                f"  [WARNING] 3D embedding failed for {label}; skipping",
                file=sys.stderr,
            )
            results.append((None, base_name, label))
            continue

        # Use MMFF94s for fully-parametrised molecules (better accuracy for organics).
        # Fall back to UFF when MMFF lacks parameters — most commonly boron, silicon,
        # and other heteroatoms common in fragment/boron-based drug libraries.
        if AllChem.MMFFHasAllMoleculeParams(prot):
            try:
                ff_results = AllChem.MMFFOptimizeMoleculeConfs(prot, mmffVariant='MMFF94s')
            except RuntimeError:
                ff_results = [(0, 0.0)] * len(cids)
        else:
            try:
                ff_results = AllChem.UFFOptimizeMoleculeConfs(prot)
            except RuntimeError:
                ff_results = [(0, 0.0)] * len(cids)

        # ff_results: list of (not_converged, energy) in conformer order
        energies = [e for _, e in ff_results]

        if keep_confs == 'best':
            best_cid = cids[min(range(len(energies)), key=lambda k: energies[k])]

            # Discard all but the lowest-energy conformer
            for cid in cids:
                if cid != best_cid:
                    prot.RemoveConformer(cid)

            # Optional xTB post-minimisation in implicit water
            if xtb_level and xtb_env:
                _xtb_refine(prot, xtb_level, xtb_env)

            results.append((prot.ToBinary(_PICKLE_PROPS), base_name, assigned_name))

        else:
            min_energy = min(energies)
            sorted_pairs = sorted(zip(cids, energies), key=lambda x: x[1])

            # Energy window filter (lowest-energy conformer is always kept)
            window_cids = [cid for cid, e in sorted_pairs if e - min_energy <= energy_window]

            # Greedy RMSD pruning — ensures conformational diversity
            kept_cids = _prune_by_rmsd(prot, window_cids, rmsd_threshold)

            # Cap at N if keep_confs is an integer
            if isinstance(keep_confs, int):
                kept_cids = kept_cids[:keep_confs]

            # One result entry per kept conformer; _confN suffix added to name
            for conf_idx, cid in enumerate(kept_cids):
                single = Chem.RWMol(prot)
                for other_id in [c.GetId() for c in single.GetConformers() if c.GetId() != cid]:
                    single.RemoveConformer(other_id)
                single = single.GetMol()
                single.SetProp('_Name', f"{assigned_name}_conf{conf_idx + 1}")
                results.append((single.ToBinary(_PICKLE_PROPS), base_name, assigned_name))

    return results


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minutes"
    hours = minutes / 60
    return f"{hours:.2f} hours"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare molecules for docking from SDF or SMILES files.",
        formatter_class=_HelpFmt,
        epilog="""
Examples:
  # Docking preparation — single low-energy conformer (default):
  %(prog)s -i molecules.sdf -o prepared.sdf
  %(prog)s -i molecules.sdf -o prepared.sdf --mode docking
  %(prog)s -i molecules.sdf -o prepared.sdf --mode docking --num-confs 10

  # Pharmit conformer database — diverse low-energy conformers:
  %(prog)s -i molecules.sdf -o pharmit.sdf --mode pharmit
  %(prog)s -i molecules.sdf -o pharmit.sdf --mode pharmit --num-confs 100
  %(prog)s -i molecules.sdf -o pharmit.sdf --mode pharmit --energy-window 3.0
  %(prog)s -i molecules.sdf -o pharmit.sdf --mode pharmit --rmsd-threshold 1.0

  # Override individual settings (--mode sets defaults, flags override):
  %(prog)s -i molecules.sdf -o prepared.sdf --mode pharmit --keep-confs 5
  %(prog)s -i molecules.sdf -o prepared.sdf --keep-confs all --num-confs 50

  # Other options:
  %(prog)s -i molecules.smi -o prepared.sdf --id-col "Compound_ID"
  %(prog)s -i molecules.sdf -o prepared.sdf -n 8 --no-enumerate-stereo
  %(prog)s -i molecules.sdf -o prepared.sdf --max-isomers 16
  %(prog)s -i compounds.csv -o prepared.sdf --smiles-col smiles --id-col name --mode pharmit
        """
    )

    _input_arg = parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file path (SDF or SMILES format)")
    if _ARGCOMPLETE:
        _input_arg.completer = FilesCompleter(allowednames=(".sdf", ".smi", ".smiles", ".csv", ".txt"))

    _output_arg = parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output SDF file path")
    if _ARGCOMPLETE:
        _output_arg.completer = FilesCompleter(allowednames=(".sdf",))

    parser.add_argument(
        "--mode",
        choices=["docking", "pharmit"],
        default="docking",
        help="Preparation mode (default: docking). "
             "'docking': 10 conformers generated, single lowest-energy one kept — for GNINA and similar programs. "
             "'pharmit': 50 conformers generated, diverse low-energy set kept — for pharmacophore databases. "
             "All conformer settings (--num-confs, --keep-confs, --energy-window, --rmsd-threshold) "
             "can be overridden individually regardless of mode."
    )

    parser.add_argument(
        "--id-col",
        default=None,
        metavar="COL",
        help="Column containing the molecule ID: 1-based integer index or column header name. "
             "For SMILES/CSV: default = auto-detect (first non-SMILES column). "
             "For SDF: default = 'Structure ID' property."
    )

    parser.add_argument(
        "-n", "--ncpus",
        type=int,
        default=os.cpu_count(),
        help=f"Number of CPUs for parallel processing (default: {os.cpu_count()} — all available CPUs)"
    )

    parser.add_argument(
        "--no-enumerate-stereo",
        action="store_true",
        help="Disable stereoisomer enumeration for unspecified chiral centers (enumeration is ON by default)"
    )

    parser.add_argument(
        "--max-isomers",
        type=int,
        default=32,
        help="Maximum number of stereoisomers to generate per molecule (default: 32)"
    )

    parser.add_argument(
        "--num-confs",
        type=int,
        default=None,
        help="Number of ETKDGv3 conformers to generate per isomer "
             "(default: 10 for docking, 50 for pharmit). "
             "Higher values improve ring-conformation sampling at the cost of more CPU time."
    )

    parser.add_argument(
        "--keep-confs",
        default=None,
        metavar="N",
        help="How many conformers to keep per isomer: "
             "'best' keeps only the lowest-energy one (default for docking); "
             "'all' keeps every conformer passing the energy-window and RMSD filters (default for pharmit); "
             "or an integer N keeps at most N conformers. "
             "Multiple conformers are written as separate SDF records with a _confN name suffix."
    )

    parser.add_argument(
        "--energy-window",
        type=float,
        default=None,
        metavar="KCAL",
        help="Energy window in kcal/mol: discard conformers with energy > (minimum + KCAL) "
             "(default: 5.0). Ignored when --keep-confs best."
    )

    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=None,
        metavar="A",
        help="RMSD pruning threshold in Angstroms: discard a conformer if it is within A of "
             "an already-kept lower-energy conformer (default: 0.5). Ignored when --keep-confs best."
    )

    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for protonation with OpenBabel (default: 7.4)"
    )

    parser.add_argument(
        "--smiles-col",
        default=None,
        metavar="COL",
        help="Column containing SMILES: 1-based integer index or column header name "
             "(default: auto-detect from common names, then first column)"
    )

    parser.add_argument(
        "--xtb",
        action="store_true",
        help="Post-minimise each molecule with xTB in implicit water (ALPB) after MMFF94s. "
             "Requires xTB to be installed and on PATH."
    )

    parser.add_argument(
        "--xtb-level",
        default="gfnff",
        choices=["gfnff", "gfn1", "gfn2"],
        help="xTB method for post-minimisation (default: gfnff). "
             "gfnff is fastest (~0.1 s/mol); gfn2 is most accurate but slower (~10 s/mol)."
    )

    parser.add_argument(
        "--failed-log",
        default=None,
        metavar="FILE",
        help="Write failed molecule IDs to this file"
    )

    if _ARGCOMPLETE:
        argcomplete.autocomplete(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input
    output_path = args.output
    id_col = args.id_col
    n_cpus = max(1, args.ncpus)
    enumerate_stereo = not args.no_enumerate_stereo
    max_isomers = args.max_isomers
    ph = args.ph
    xtb_level = args.xtb_level if args.xtb else None

    # Apply mode defaults; individual flags override.
    _mode_defaults = {
        'docking': dict(num_confs=10, keep_confs='best', energy_window=5.0, rmsd_threshold=0.5),
        'pharmit': dict(num_confs=50, keep_confs='all',  energy_window=5.0, rmsd_threshold=0.5),
    }
    _d = _mode_defaults[args.mode]
    num_confs      = max(1, args.num_confs      if args.num_confs      is not None else _d['num_confs'])
    energy_window  =        args.energy_window  if args.energy_window  is not None else _d['energy_window']
    rmsd_threshold =        args.rmsd_threshold if args.rmsd_threshold is not None else _d['rmsd_threshold']

    keep_confs_raw = args.keep_confs if args.keep_confs is not None else _d['keep_confs']
    if keep_confs_raw in ('best', 'all'):
        keep_confs = keep_confs_raw
    else:
        try:
            keep_confs = int(keep_confs_raw)
            if keep_confs < 1:
                raise ValueError
        except ValueError:
            print(f"ERROR: --keep-confs must be 'best', 'all', or a positive integer "
                  f"(got '{keep_confs_raw}')", file=sys.stderr)
            return

    if xtb_level and not _XTBBIN:
        print("ERROR: --xtb requested but 'xtb' binary not found on PATH. "
              "Install via: conda install -c conda-forge xtb", file=sys.stderr)
        return

    if not os.path.isfile(input_path):
        print(f"ERROR: Input file '{input_path}' does not exist.")
        return

    ext = os.path.splitext(input_path)[1].lower()

    # Resolve column specs (1-based int or header name, like ligfilter.py).
    # Done before the format branch so both SDF and SMILES paths can use them.
    smiles_col_spec = _parse_col_spec(args.smiles_col) if args.smiles_col else None
    id_col_spec     = _parse_col_spec(args.id_col)     if args.id_col     else None

    # For SDF, fall back to 'Structure ID' when --id-col is not given.
    if id_col_spec is None and ext == '.sdf':
        id_col = 'Structure ID'
    elif isinstance(id_col_spec, str):
        id_col = id_col_spec
    else:
        id_col = 'Structure ID'  # placeholder; overwritten below for SMILES input

    # Build a lazy mol iterator — SDF supplier is already lazy.
    # For SMILES/CSV, column detection is done eagerly on the header line,
    # then molecules are yielded one at a time via a nested generator.
    if ext == '.sdf':
        suppl = Chem.SDMolSupplier(input_path, removeHs=False)
        mol_iter = (mol for mol in suppl if mol is not None)
    else:
        with open(input_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            print("ERROR: Input file is empty.")
            return

        sep = '\t' if '\t' in lines[0] else ','

        # ── Header detection ──────────────────────────────────────────────────
        need_header = isinstance(smiles_col_spec, str) or isinstance(id_col_spec, str)
        _rdlog = RDLogger.logger()
        _rdlog.setLevel(RDLogger.CRITICAL)
        first_cell = lines[0].split(sep)[0].strip()
        is_header = need_header or (Chem.MolFromSmiles(first_cell) is None)
        _rdlog.setLevel(RDLogger.WARNING)

        if is_header:
            col_names = [c.strip() for c in lines[0].split(sep)]
            data_lines = lines[1:]
        else:
            col_names = []
            data_lines = lines

        # ── Resolve SMILES column ─────────────────────────────────────────────
        if smiles_col_spec is None:
            if col_names:
                _smiles_names = {"smiles", "smi", "canonical_smiles", "smiles_string"}
                smi_col_idx = next((i for i, c in enumerate(col_names)
                                    if c.lower() in _smiles_names), 0)
            else:
                smi_col_idx = 0
        elif isinstance(smiles_col_spec, int):
            smi_col_idx = smiles_col_spec
        else:
            if smiles_col_spec not in col_names:
                print(f"ERROR: --smiles-col '{smiles_col_spec}' not found in header: {col_names}",
                      file=sys.stderr)
                return
            smi_col_idx = col_names.index(smiles_col_spec)

        # ── Resolve ID column ─────────────────────────────────────────────────
        if id_col_spec is None:
            if col_names:
                _id_names = {"name", "id", "compound_id", "molecule_name", "structure id"}
                id_col_idx = next((i for i, c in enumerate(col_names)
                                   if i != smi_col_idx and c.lower() in _id_names), None)
                if id_col_idx is None:
                    id_col_idx = next((i for i in range(len(col_names)) if i != smi_col_idx), None)
            else:
                id_col_idx = next((i for i in range(2) if i != smi_col_idx), None)
        elif isinstance(id_col_spec, int):
            id_col_idx = id_col_spec
        else:
            if id_col_spec not in col_names:
                print(f"ERROR: --id-col '{id_col_spec}' not found in header: {col_names}",
                      file=sys.stderr)
                return
            id_col_idx = col_names.index(id_col_spec)

        # ── Determine SDF property name for the molecule ID ───────────────────
        if col_names and id_col_idx is not None and id_col_idx < len(col_names):
            id_col = col_names[id_col_idx]
        elif isinstance(id_col_spec, str):
            id_col = id_col_spec
        # else: id_col keeps its current value ('Structure ID' placeholder)

        smi_label = col_names[smi_col_idx] if col_names and smi_col_idx < len(col_names) else str(smi_col_idx + 1)
        id_label  = col_names[id_col_idx]  if col_names and id_col_idx is not None and id_col_idx < len(col_names) else (str(id_col_idx + 1) if id_col_idx is not None else 'none')
        print(f"[INFO] SMILES col: '{smi_label}', ID col: '{id_label}'")

        def _smi_mol_iter():
            for line in data_lines:
                parts = [p.strip() for p in line.split(sep)]
                if smi_col_idx >= len(parts):
                    continue
                smi = parts[smi_col_idx]
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    if id_col_idx is not None and id_col_idx < len(parts):
                        mol.SetProp(id_col, parts[id_col_idx])
                    yield mol
                else:
                    print(f"[WARNING] SMILES parse error, skipping: {line}")

        mol_iter = _smi_mol_iter()

    print(f"[INFO] obabel binary: {_OBABEL}")
    _conf_info = (f" (energy window {energy_window} kcal/mol, RMSD threshold {rmsd_threshold} Å)"
                  if keep_confs != 'best' else "")
    print(f"[INFO] mode: {args.mode} | conformers: {num_confs} generated, keep: {keep_confs}{_conf_info}")
    if xtb_level:
        if keep_confs != 'best':
            print(f"[WARNING] --xtb is ignored in multi-conformer mode (keep-confs={keep_confs!r}); "
                  "xTB is only applied when --keep-confs best.", file=sys.stderr)
            xtb_level = None
        else:
            print(f"[INFO] xTB post-minimisation enabled ({xtb_level}, ALPB water) using {_XTBBIN}")
    if enumerate_stereo:
        print(f"Stereoisomer enumeration enabled (max {max_isomers} isomers per molecule)")

    # ── Phase 1: salt strip + stereo enumerate → flat isomer task list ────────
    # Done in main so all workers get equal-sized, independent tasks.
    isomer_tasks = []
    input_count = 0
    for mol in mol_iter:
        input_count += 1
        # Capture name BEFORE salt stripping — GetMolFrags creates new mol
        # objects that do not inherit string properties.
        base_name = mol.GetProp(id_col) if mol.HasProp(id_col) else ""
        mol = strip_salts_keep_largest(mol)
        if mol is None:
            continue
        isomers = enumerate_stereoisomers(mol, max_isomers) if enumerate_stereo else [mol]
        for idx, isomer in enumerate(isomers):
            if len(isomers) > 1 and base_name:
                assigned_name = f"{base_name}{get_stereo_suffix(isomer, idx + 1)}"
            else:
                assigned_name = base_name
            isomer_tasks.append((isomer.ToBinary(), base_name, assigned_name, ph, id_col,
                                  num_confs, xtb_level, keep_confs, energy_window, rmsd_threshold))

    total_isomers = len(isomer_tasks)
    print(f"Processing {input_count} molecules → {total_isomers} isomers using {n_cpus} CPU(s), {num_confs} conformer(s) each...")

    # ── Phase 2: chunk → submit → stream results to writer ───────────────────
    batches = [isomer_tasks[i:i + _BATCH_SIZE] for i in range(0, total_isomers, _BATCH_SIZE)]
    del isomer_tasks  # free binary mol data before spawning workers

    start_time = time.time()
    total_output = 0
    failed_count = 0
    failed_names = []

    # Use spawn context to avoid fork+RDKit deadlocks.  When the main process
    # forks workers, RDKit's C++ internals (thread pool, lazy-init ring
    # templates) may have mutex state that deadlocks the child.  Spawn starts
    # each worker completely fresh so there is no inherited lock state.
    _mp_ctx = multiprocessing.get_context('spawn')
    writer = SDWriter(output_path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus,
                                                mp_context=_mp_ctx) as executor:
        futures = [executor.submit(_prepare_isomer_batch, b) for b in batches]
        del batches

        for future in tqdm(futures, desc="Preparing molecules", unit="batch"):
            for mol_b, base_name, assigned_name in future.result():
                if mol_b:
                    writer.write(Chem.Mol(mol_b))
                    total_output += 1
                else:
                    failed_count += 1
                    failed_names.append(assigned_name)

    writer.close()
    elapsed_time = time.time() - start_time

    if args.failed_log and failed_names:
        with open(args.failed_log, 'w') as f:
            for name in failed_names:
                f.write(f"{name or '(unknown)'}\n")
        print(f"Failed molecule IDs written to {args.failed_log}")

    print(f"\nFinished writing {total_output} prepared molecules to {output_path}")
    print(f"  (from {input_count} input molecules, {failed_count} failed)")
    print(f"Total processing time: {format_time(elapsed_time)}")

    if input_count > 0:
        print(f"Average processing time per input structure: {elapsed_time / input_count:.3f} seconds")


if __name__ == "__main__":
    main()
