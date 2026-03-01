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
from rich_argparse import RawDescriptionRichHelpFormatter
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import time
from tqdm import tqdm

import argcomplete
from argcomplete.completers import FilesCompleter

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


def obabel_protonate_batch(smiles_list, ph=7.4):
    """Batch-protonate SMILES at the given pH using a single obabel subprocess.

    Each SMILES is tagged with a synthetic name ``__idx_N`` so the parsed SDF
    records can be mapped back to their original positions.

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
            [_OBABEL, '-ismi', '-osdf', f'-p{ph}'],
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
    supplier = Chem.SDMolSupplier()
    supplier.SetData(result.stdout, removeHs=False)
    logger.setLevel(RDLogger.WARNING)

    output = [None] * len(smiles_list)
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
        if name.startswith('__idx_'):
            try:
                idx = int(name[6:])
                if 0 <= idx < len(output):
                    output[idx] = mol
            except ValueError:
                pass
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


def _prepare_isomer_batch(batch):
    """Worker: prepare a batch of isomers with one shared obabel call.

    Args:
        batch: list of (mol_binary, base_name, assigned_name, props_dict,
                        ph, id_col, num_confs, xtb_level)
               xtb_level is None to skip xTB, or 'gfnff'/'gfn1'/'gfn2'.

    Returns:
        list of (mol_binary_or_None, base_name, assigned_name)
        mol_binary is None when 3D embedding failed.
    """
    if not batch:
        return []

    ph        = batch[0][4]
    num_confs = batch[0][6]
    xtb_level = batch[0][7]

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
    for i, (mol_binary, base_name, assigned_name, props_dict, ph, id_col, num_confs, _xtb_level) in enumerate(batch):
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

        # Strip flat obabel coords (all-zero 0D block) before re-embedding.
        prot = Chem.RemoveHs(prot)

        # obabel outputs a 0D mol block (all coords 0,0,0).  The stereo parity
        # bits ARE written to the atom table but RDKit cannot reconstruct chiral
        # tags from them without real coordinates, so they are silently dropped.
        # Re-attach the tags from the canonical SMILES we sent to obabel; obabel
        # preserves atom ordering from SMILES input, so index-based copy is safe.
        # Skip when obabel failed and we fell back to the original mol (stereo OK).
        if not obabel_failed:
            mol_canon = Chem.MolFromSmiles(smiles_list[i])
            if mol_canon is not None:
                rw = Chem.RWMol(prot)
                for atom in mol_canon.GetAtoms():
                    ct = atom.GetChiralTag()
                    if ct != Chem.ChiralType.CHI_UNSPECIFIED:
                        idx = atom.GetIdx()
                        if idx < rw.GetNumAtoms():
                            rw.GetAtomWithIdx(idx).SetChiralTag(ct)
                prot = rw.GetMol()

        # Restore SD properties and set name tags after all mol-object surgery.
        for k, v in props_dict.items():
            prot.SetProp(k, v)
        prot.SetProp('_Name', assigned_name)
        if base_name:
            prot.SetProp(id_col, base_name)
            if assigned_name != base_name:
                prot.SetProp(f"{id_col}_stereo", assigned_name)

        prot = Chem.AddHs(prot)

        # Generate num_confs conformers with ETKDGv3, minimize all, keep the
        # lowest-energy one.  Multi-conformer sampling avoids the twisted-boat
        # ring minima that a single fixed-seed embedding can get trapped in.
        #
        # Fallback chain for complex ring systems (macrocycles, bridged polycyclics)
        # where distance-geometry initialisation fails:
        #   1. ETKDGv3 + useSmallRingTorsions  (normal path)
        #   2. ETKDGv3 + randomCoords=True      (bypasses DG initialisation)
        try:
            cids = AllChem.EmbedMultipleConfs(prot, numConfs=num_confs, params=_etkdg)
        except RuntimeError:
            cids = []

        if not cids:
            _etkdg_rc = AllChem.ETKDGv3()
            _etkdg_rc.randomSeed = 0xf00d
            _etkdg_rc.useSmallRingTorsions = True
            _etkdg_rc.useRandomCoords = True
            try:
                cids = AllChem.EmbedMultipleConfs(prot, numConfs=num_confs, params=_etkdg_rc)
            except RuntimeError:
                cids = []

        if not cids:
            label = assigned_name or smiles_list[i]
            print(
                f"  [WARNING] 3D embedding failed for {label}; skipping",
                file=sys.stderr,
            )
            results.append((None, base_name, label))
            continue

        try:
            ff_results = AllChem.MMFFOptimizeMoleculeConfs(prot, mmffVariant='MMFF94s')
        except RuntimeError:
            # BFGS linearSearch failure on a specific conformer; keep unminimised best
            ff_results = [(0, 0.0)] * len(cids)

        # ff_results: list of (not_converged, energy) in conformer order
        energies = [e for _, e in ff_results]
        best_cid = cids[min(range(len(energies)), key=lambda k: energies[k])]

        # Discard all but the lowest-energy conformer
        for cid in cids:
            if cid != best_cid:
                prot.RemoveConformer(cid)

        # Optional xTB post-minimisation in implicit water
        if xtb_level and xtb_env:
            _xtb_refine(prot, xtb_level, xtb_env)

        results.append((prot.ToBinary(_PICKLE_PROPS), base_name, assigned_name))

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
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i molecules.sdf -o prepared.sdf
  %(prog)s -i molecules.smi -o prepared.sdf --id-col "Compound_ID"
  %(prog)s -i molecules.sdf -o prepared.sdf -n 8 --no-enumerate-stereo
  %(prog)s -i molecules.sdf -o prepared.sdf --max-isomers 16
  %(prog)s -i compounds.csv -o prepared.sdf --smiles-col smiles --id-col name
  %(prog)s -i compounds.csv -o prepared.sdf --smiles-col 2
        """
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file path (SDF or SMILES format)"
    ).completer = FilesCompleter(allowednames=(".sdf", ".smi", ".smiles", ".csv", ".txt"))

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output SDF file path"
    ).completer = FilesCompleter(allowednames=(".sdf",))

    parser.add_argument(
        "--id-col",
        default="Structure ID",
        help="Identifier property name to preserve (default: 'Structure ID')"
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
        default=3,
        help="Number of ETKDGv3 conformers to generate per isomer; the lowest-energy one is kept (default: 3). "
             "Higher values improve ring-conformation sampling at the cost of proportionally more CPU time."
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
        help="SMILES column name or 0-based index in delimited files (default: auto-detect)"
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
    num_confs = max(1, args.num_confs)
    ph = args.ph
    xtb_level = args.xtb_level if args.xtb else None

    if xtb_level and not _XTBBIN:
        print("ERROR: --xtb requested but 'xtb' binary not found on PATH. "
              "Install via: conda install -c conda-forge xtb", file=sys.stderr)
        return

    if not os.path.isfile(input_path):
        print(f"ERROR: Input file '{input_path}' does not exist.")
        return

    ext = os.path.splitext(input_path)[1].lower()

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
        header = [c.strip() for c in lines[0].split(sep)]

        # Detect header by trying to parse the first cell as SMILES
        smi_col_idx = 0
        id_col_idx = 1 if len(header) > 1 else None
        _rdlog = RDLogger.logger()
        _rdlog.setLevel(RDLogger.CRITICAL)
        is_header = Chem.MolFromSmiles(header[0]) is None
        _rdlog.setLevel(RDLogger.WARNING)

        if is_header:
            smiles_names = {"smiles", "smi", "canonical_smiles", "smiles_string"}
            for i, col in enumerate(header):
                if col.lower() in smiles_names:
                    smi_col_idx = i
                    break
            id_names = {id_col.lower(), "name", "id", "compound_id", "molecule_name"}
            id_col_idx = None
            for i, col in enumerate(header):
                if i != smi_col_idx and col.lower() in id_names:
                    id_col_idx = i
                    break
            data_lines = lines[1:]
        else:
            data_lines = lines

        # Apply --smiles-col override (name or 0-based index)
        if args.smiles_col is not None:
            sc = args.smiles_col
            if sc.isdigit():
                smi_col_idx = int(sc)
            elif is_header:
                col_lower = [c.lower() for c in header]
                if sc.lower() in col_lower:
                    smi_col_idx = col_lower.index(sc.lower())
                else:
                    print(f"ERROR: --smiles-col '{sc}' not found in header: {header}")
                    return
            else:
                print(f"ERROR: --smiles-col '{sc}' is not a valid index and file has no header row.")
                return

        smi_label = header[smi_col_idx] if is_header else str(smi_col_idx)
        id_label = header[id_col_idx] if (is_header and id_col_idx is not None) else str(id_col_idx) if id_col_idx is not None else "none"
        if is_header or args.smiles_col is not None:
            print(f"[INFO] SMILES col: '{smi_label}', ID col: '{id_label}'")

        def _smi_mol_iter():
            for line in data_lines:
                parts = [p.strip() for p in line.split(sep)]
                if smi_col_idx >= len(parts):
                    continue
                smi = parts[smi_col_idx]
                name = parts[id_col_idx] if id_col_idx is not None and id_col_idx < len(parts) else None
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    if name:
                        mol.SetProp(id_col, name)
                    yield mol
                else:
                    print(f"SMILES parse error for line: {line}")

        mol_iter = _smi_mol_iter()

    print(f"[INFO] obabel binary: {_OBABEL}")
    if xtb_level:
        print(f"[INFO] xTB post-minimisation enabled ({xtb_level}, ALPB water) using {_XTBBIN}")
    if enumerate_stereo:
        print(f"Stereoisomer enumeration enabled (max {max_isomers} isomers per molecule)")

    # ── Phase 1: salt strip + stereo enumerate → flat isomer task list ────────
    # Done in main so all workers get equal-sized, independent tasks.
    isomer_tasks = []
    input_count = 0
    for mol in mol_iter:
        input_count += 1
        mol = strip_salts_keep_largest(mol)
        if mol is None:
            continue
        base_name = mol.GetProp(id_col) if mol.HasProp(id_col) else ""
        props = {p: mol.GetProp(p) for p in mol.GetPropNames()}
        isomers = enumerate_stereoisomers(mol, max_isomers) if enumerate_stereo else [mol]
        for idx, isomer in enumerate(isomers):
            if len(isomers) > 1 and base_name:
                assigned_name = f"{base_name}{get_stereo_suffix(isomer, idx + 1)}"
            else:
                assigned_name = base_name
            isomer_tasks.append((isomer.ToBinary(), base_name, assigned_name, props, ph, id_col, num_confs, xtb_level))

    total_isomers = len(isomer_tasks)
    print(f"Processing {input_count} molecules → {total_isomers} isomers using {n_cpus} CPU(s), {num_confs} conformer(s) each...")

    # ── Phase 2: chunk → submit → stream results to writer ───────────────────
    batches = [isomer_tasks[i:i + _BATCH_SIZE] for i in range(0, total_isomers, _BATCH_SIZE)]
    del isomer_tasks  # free binary mol data before spawning workers

    start_time = time.time()
    total_output = 0
    failed_count = 0
    failed_names = []

    writer = SDWriter(output_path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
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
