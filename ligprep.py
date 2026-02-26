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
- Minimizing geometry using the UFF force field
- Preserving specified molecule identifier properties throughout processing
- Writing the processed molecules to an output SDF file

The script supports parallel processing with user-defined CPU count for faster preparation of large molecule libraries.
It reports total and average processing time, formatted for easy reading.

SMILES input files must have the SMILES string in the first column, followed optionally by an identifier (second column).
The file can be comma-separated or tab-separated.

Written by Perplexity 4.0.0, 2025-08-19
Updated by Claude, 2026-01-24: Added argparse and stereoisomer enumeration
"""

import argparse
from rich_argparse import RawDescriptionRichHelpFormatter
import concurrent.futures
import os
import time
from tqdm import tqdm

import argcomplete
from argcomplete.completers import FilesCompleter

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdmolfiles import SDWriter
from openbabel import openbabel


def rdkit_to_openbabel(rdkit_mol):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol", "mol")
    mol_block = Chem.MolToMolBlock(rdkit_mol)
    obmol = openbabel.OBMol()
    obConversion.ReadString(obmol, mol_block)
    return obmol, obConversion


def openbabel_protonate(obmol, obConversion, pH=7.4):
    obmol.CorrectForPH(pH)
    return obConversion.WriteString(obmol)


def openbabel_to_rdkit(mol_block):
    return Chem.MolFromMolBlock(mol_block, sanitize=True, removeHs=False)


def strip_salts_keep_largest(mol):
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
    return largest_frag


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
    
    # If no unspecified stereocenters, return the original molecule
    if len(isomers) == 0:
        return [mol]
    
    return isomers


def get_stereo_suffix(mol, index):
    """Generate a suffix indicating stereochemistry (e.g., '_R,S' or '_isomer_1')."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    stereo_labels = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            stereo_labels.append(atom.GetProp('_CIPCode'))
    
    if stereo_labels:
        return f"_{''.join(stereo_labels)}"
    else:
        return f"_isomer_{index}"


def prepare_molecule_worker(args):
    mol, id_col, cached_id, enumerate_stereo, max_isomers, ph = args
    
    mol = strip_salts_keep_largest(mol)
    if mol is None:
        return [], cached_id

    props = {prop: mol.GetProp(prop) for prop in mol.GetPropNames()}

    # Enumerate stereoisomers if requested
    if enumerate_stereo:
        isomers = enumerate_stereoisomers(mol, max_isomers)
    else:
        isomers = [mol]
    
    prepared_mols = []
    
    for idx, isomer in enumerate(isomers):
        # Restore properties to isomer
        for k, v in props.items():
            isomer.SetProp(k, v)
        
        # Protonate using OpenBabel
        obmol, obConversion = rdkit_to_openbabel(isomer)
        protonated_mol_block = openbabel_protonate(obmol, obConversion, pH=ph)
        protonated_mol = openbabel_to_rdkit(protonated_mol_block)
        
        if protonated_mol is None:
            continue

        # Restore properties after protonation
        for k, v in props.items():
            protonated_mol.SetProp(k, v)

        protonated_mol = Chem.AddHs(protonated_mol)
        
        if AllChem.EmbedMolecule(protonated_mol, randomSeed=0xf00d) != 0:
            continue

        AllChem.UFFOptimizeMolecule(protonated_mol)
        
        # Add stereochemistry info to name if multiple isomers
        if len(isomers) > 1 and cached_id:
            stereo_suffix = get_stereo_suffix(isomer, idx + 1)
            protonated_mol.SetProp(f"{id_col}_stereo", f"{cached_id}{stereo_suffix}")
        
        prepared_mols.append(protonated_mol)
    
    return prepared_mols, cached_id


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
        default=1,
        help="Number of CPUs for parallel processing (default: 1)"
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
    ph = args.ph

    if not os.path.isfile(input_path):
        print(f"ERROR: Input file '{input_path}' does not exist.")
        return

    ext = os.path.splitext(input_path)[1].lower()
    mols = []
    
    if ext == '.sdf':
        suppl = Chem.SDMolSupplier(input_path, removeHs=False)
        mols = [mol for mol in suppl if mol is not None]
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
        is_header = Chem.MolFromSmiles(header[0]) is None

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
                mols.append(mol)
            else:
                print(f"SMILES parse error for line: {line}")

    print(f"Processing {len(mols)} molecules using {n_cpus} CPU(s)...")
    if enumerate_stereo:
        print(f"Stereoisomer enumeration enabled (max {max_isomers} isomers per molecule)")

    worker_args = [
        (mol, id_col, mol.GetProp(id_col) if mol.HasProp(id_col) else "", enumerate_stereo, max_isomers, ph)
        for mol in mols
    ]

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
        results = list(tqdm(executor.map(prepare_molecule_worker, worker_args),
                            total=len(worker_args), desc="Preparing molecules", unit="mol"))

    elapsed_time = time.time() - start_time

    writer = SDWriter(output_path)
    total_output = 0
    failed_count = 0

    for prepared_list, cached_id in results:
        if prepared_list:
            for prepared in prepared_list:
                if cached_id:
                    prepared.SetProp(id_col, cached_id)
                writer.write(prepared)
                total_output += 1
        else:
            failed_count += 1
            print(f"Warning: Could not prepare molecule {cached_id or '(unknown)'}")

    writer.close()

    if args.failed_log and failed_count > 0:
        with open(args.failed_log, 'w') as f:
            for prepared_list, cached_id in results:
                if not prepared_list:
                    f.write(f"{cached_id or '(unknown)'}\n")
        print(f"Failed molecule IDs written to {args.failed_log}")

    print(f"\nFinished writing {total_output} prepared molecules to {output_path}")
    if enumerate_stereo:
        print(f"  (from {len(mols)} input molecules, {failed_count} failed)")
    print(f"Total processing time: {format_time(elapsed_time)}")

    if len(mols) > 0:
        time_per_structure = elapsed_time / len(mols)
        print(f"Average processing time per input structure: {time_per_structure:.3f} seconds")


if __name__ == "__main__":
    main()
