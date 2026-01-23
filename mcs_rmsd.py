#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
MCS RMSD Calculator - CLI Tool

Calculates Maximum Common Substructure (MCS) RMSD between docking poses 
and a reference ligand using RDKit.

Author: Converted from KNIME script
"""

import argparse
from rich_argparse import RawDescriptionRichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter
import sys
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, PandasTools


def get_heavy_atom_indices(mol):
    """Get indices of heavy (non-hydrogen) atoms."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def calculate_mcs(ref_mol, pose_mol):
    """Calculate Maximum Common Substructure between two molecules."""
    mcs_result = rdFMCS.FindMCS(
        [ref_mol, pose_mol], 
        atomCompare=rdFMCS.AtomCompare.CompareElements
    )
    mcs_smarts = mcs_result.smartsString
    return Chem.MolFromSmarts(mcs_smarts) if mcs_smarts else None


def get_mcs_match_indices(mol, mcs_mol):
    """Get atom indices that match the MCS pattern."""
    match = mol.GetSubstructMatch(mcs_mol)
    return list(match) if match else []


def calculate_rmsd(ref_mol, pose_mol, ref_indices, pose_indices):
    """Calculate RMSD between matched atoms after alignment."""
    ref_conf = ref_mol.GetConformer()
    pose_conf = pose_mol.GetConformer()
    
    # Get coordinates before alignment
    ref_coords = [ref_conf.GetAtomPosition(i) for i in ref_indices]
    pose_coords = [pose_conf.GetAtomPosition(i) for i in pose_indices]
    
    # Align molecules
    AllChem.AlignMol(
        pose_mol, 
        ref_mol, 
        atomMap=list(zip(pose_indices, ref_indices))
    )
    
    # Calculate RMSD
    diff_sq = [(ref_coords[i] - pose_coords[i]).LengthSq() 
               for i in range(len(ref_coords))]
    rmsd = (sum(diff_sq) / len(diff_sq)) ** 0.5
    
    return rmsd


def process_mcs_rmsd(ref_mol, poses_df, mol_column='Molecule', id_column='ID'):
    """
    Calculate MCS RMSD for all poses against reference molecule.
    
    Args:
        ref_mol: Reference RDKit Mol object
        poses_df: DataFrame containing docking poses
        mol_column: Name of column containing Mol objects
        id_column: Name of ID column (optional)
    
    Returns:
        DataFrame with MCS RMSD results
    """
    results = []
    
    for idx, row in tqdm(poses_df.iterrows(), total=len(poses_df), desc="Calculating MCS RMSD", unit="pose"):
        # Start with all existing columns
        result_row = {}
        
        # Get pose molecule
        pose_mol = row.get(mol_column, None)
        
        # Copy all non-molecule columns from input
        for col in poses_df.columns:
            if col != mol_column:
                result_row[col] = row[col]
        
        # Ensure we have a Pose_ID (either from data or generated)
        if 'Pose_ID' not in result_row or pd.isna(result_row.get('Pose_ID')):
            result_row['Pose_ID'] = f"pose_{idx + 1}"
        
        # Calculate RMSD
        if pose_mol is None or ref_mol is None:
            rmsd = None
        else:
            mcs_mol = calculate_mcs(ref_mol, pose_mol)
            
            if mcs_mol is None:
                rmsd = None
            else:
                ref_match = get_mcs_match_indices(ref_mol, mcs_mol)
                pose_match = get_mcs_match_indices(pose_mol, mcs_mol)
                
                # Filter for heavy atoms only
                ref_heavy = [i for i in ref_match 
                           if ref_mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                pose_heavy = [i for i in pose_match 
                            if pose_mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                
                if len(ref_heavy) != len(pose_heavy) or len(ref_heavy) == 0:
                    rmsd = None
                else:
                    rmsd = calculate_rmsd(ref_mol, pose_mol, ref_heavy, pose_heavy)
        
        # Add RMSD to result
        result_row['MCS_RMSD'] = rmsd
        result_row[mol_column] = pose_mol
        
        results.append(result_row)
    
    output_df = pd.DataFrame(results)
    
    # Handle ID column type if it exists
    if id_column in output_df.columns:
        if output_df[id_column].isnull().any():
            output_df[id_column] = output_df[id_column].astype('Int32')
        else:
            try:
                output_df[id_column] = output_df[id_column].astype('int32')
            except (ValueError, TypeError):
                pass  # Keep as is if conversion fails
    
    return output_df


def load_molecules_from_file(filepath, mol_column='Molecule', sanitize=True):
    """
    Load molecules from various file formats.
    
    Supports: SDF, CSV (with SMILES), pickle (pandas with RDKit Mol objects)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.sdf':
        # Try with sanitization first
        suppl = Chem.SDMolSupplier(str(filepath), sanitize=sanitize, removeHs=False)
        mols = [mol for mol in suppl if mol is not None]
        
        # If no molecules loaded and sanitization was on, try without sanitization
        if not mols and sanitize:
            print(f"Warning: Sanitization failed, retrying without sanitization...")
            suppl = Chem.SDMolSupplier(str(filepath), sanitize=False, removeHs=False)
            mols = [mol for mol in suppl if mol is not None]
        
        if not mols:
            raise ValueError(f"No valid molecules found in {filepath}")
        
        # Create DataFrame with properties
        data = []
        for idx, mol in enumerate(mols):
            mol_data = {mol_column: mol}
            
            # Add properties from SDF
            props = mol.GetPropsAsDict()
            for prop in props:
                mol_data[prop] = mol.GetProp(prop)
            
            # Try to get or generate an identifier
            # Common property names for identifiers in SDF files
            id_found = False
            for id_prop in ['_Name', 'ID', 'Name', 'id', 'name', 'TITLE', 'Title']:
                if id_prop in props:
                    if 'Pose_ID' not in mol_data:
                        mol_data['Pose_ID'] = props[id_prop]
                    id_found = True
                    break
            
            # Also try molecule name
            if not id_found and mol.HasProp('_Name'):
                mol_data['Pose_ID'] = mol.GetProp('_Name')
                id_found = True
            
            # If no ID found, generate one based on position
            if not id_found:
                mol_data['Pose_ID'] = f"pose_{idx + 1}"
            
            data.append(mol_data)
        
        return pd.DataFrame(data)
    
    elif suffix == '.csv':
        # Load CSV - expect SMILES column
        df = pd.read_csv(filepath)
        
        # Try to find SMILES column
        smiles_col = None
        for col in ['SMILES', 'smiles', 'Smiles', 'SMARTS', 'smarts']:
            if col in df.columns:
                smiles_col = col
                break
        
        if smiles_col is None:
            raise ValueError("CSV file must contain a SMILES column")
        
        # Convert SMILES to Mol objects
        PandasTools.AddMoleculeColumnToFrame(df, smiles_col, mol_column)
        return df
    
    elif suffix == '.pkl' or suffix == '.pickle':
        # Load pickle file (assumes DataFrame with Mol objects)
        df = pd.read_pickle(filepath)
        return df
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}. "
                       f"Supported formats: .sdf, .csv, .pkl")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate MCS RMSD between docking poses and reference ligand',
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  # Basic usage with SDF files
  %(prog)s -r reference.sdf -p poses.sdf -o results.csv
  
  # Skip sanitization for problematic structures
  %(prog)s -r reference.sdf -p poses.sdf -o results.csv --no-sanitize
  
  # Specify custom column names
  %(prog)s -r ref.sdf -p poses.sdf -o out.csv --mol-column ROMol --id-column mol_id
  
  # Output as SDF
  %(prog)s -r reference.sdf -p poses.sdf -o results.sdf

Supported input formats: .sdf, .csv (with SMILES), .pkl (pandas DataFrame)
Supported output formats: .csv, .sdf, .pkl
        """
    )
    
    parser.add_argument(
        '-r', '--reference',
        required=True,
        help='Reference ligand file (SDF, CSV with SMILES, or pickle)'
    ).completer = FilesCompleter(allowednames=(".sdf", ".csv", ".pkl", ".pickle"))

    parser.add_argument(
        '-p', '--poses',
        required=True,
        help='Docking poses file (SDF, CSV with SMILES, or pickle)'
    ).completer = FilesCompleter(allowednames=(".sdf", ".csv", ".pkl", ".pickle"))

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file path (CSV, SDF, or pickle)'
    ).completer = FilesCompleter(allowednames=(".csv", ".sdf", ".pkl", ".pickle"))
    
    parser.add_argument(
        '--mol-column',
        default='Molecule',
        help='Name of molecule column in DataFrames (default: Molecule)'
    )
    
    parser.add_argument(
        '--id-column',
        default='ID',
        help='Name of ID column (default: ID)'
    )
    
    parser.add_argument(
        '--no-sanitize',
        action='store_true',
        help='Skip molecule sanitization (use for problematic structures)'
    )
    
    parser.add_argument(
        '--rmsd-cutoff',
        type=float,
        default=None,
        metavar='Å',
        help='Only output poses with MCS RMSD below this cutoff (Angstroms)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    try:
        # Load reference ligand
        if args.verbose:
            print(f"Loading reference ligand from: {args.reference}")
        
        ref_df = load_molecules_from_file(
            args.reference, 
            args.mol_column,
            sanitize=not args.no_sanitize
        )
        
        if ref_df.empty:
            raise ValueError("Reference ligand file is empty")
        
        ref_mol = ref_df[args.mol_column].iloc[0]
        
        if ref_mol is None:
            raise ValueError("Reference molecule could not be loaded")
        
        if args.verbose:
            print(f"Reference ligand loaded successfully")
            print(f"  Formula: {Chem.rdMolDescriptors.CalcMolFormula(ref_mol)}")
            print(f"  Heavy atoms: {ref_mol.GetNumHeavyAtoms()}")
        
        # Load docking poses
        if args.verbose:
            print(f"\nLoading docking poses from: {args.poses}")
        
        poses_df = load_molecules_from_file(
            args.poses, 
            args.mol_column,
            sanitize=not args.no_sanitize
        )
        
        if poses_df.empty:
            raise ValueError("Docking poses file is empty")
        
        num_poses = len(poses_df)
        if args.verbose:
            print(f"Loaded {num_poses} docking poses")
            print(f"Available columns: {', '.join(poses_df.columns)}")
            if 'Pose_ID' in poses_df.columns:
                print(f"Pose identifiers found in 'Pose_ID' column")
        
        # Calculate MCS RMSD
        if args.verbose:
            print(f"\nCalculating MCS RMSD...")
        
        results_df = process_mcs_rmsd(
            ref_mol, 
            poses_df,
            mol_column=args.mol_column,
            id_column=args.id_column
        )
        
        # Apply RMSD cutoff filter
        if args.rmsd_cutoff is not None:
            before = len(results_df)
            results_df = results_df[results_df['MCS_RMSD'] <= args.rmsd_cutoff]
            print(f"RMSD cutoff {args.rmsd_cutoff} Å: {len(results_df)}/{before} poses retained")

        # Report statistics (always shown)
        valid_rmsds = results_df['MCS_RMSD'].dropna()
        if len(valid_rmsds) > 0:
            print(f"\nRMSD Statistics:")
            print(f"  Valid calculations: {len(valid_rmsds)}/{num_poses}")
            print(f"  Mean RMSD: {valid_rmsds.mean():.3f} Å")
            print(f"  Min RMSD:  {valid_rmsds.min():.3f} Å")
            print(f"  Max RMSD:  {valid_rmsds.max():.3f} Å")
        else:
            print(f"\nWarning: No valid RMSD calculations")

        # Save results
        output_path = Path(args.output)
        output_suffix = output_path.suffix.lower()
        
        if args.verbose:
            print(f"\nSaving results to: {args.output}")
        
        if output_suffix == '.csv':
            # For CSV, drop the Mol column and save
            output_df = results_df.drop(columns=[args.mol_column], errors='ignore')
            output_df.to_csv(output_path, index=False)
        
        elif output_suffix == '.sdf':
            # Save as SDF
            PandasTools.WriteSDF(
                results_df, 
                str(output_path), 
                molColName=args.mol_column,
                properties=list(results_df.columns)
            )
        
        elif output_suffix in ['.pkl', '.pickle']:
            # Save as pickle
            results_df.to_pickle(output_path)
        
        else:
            raise ValueError(f"Unsupported output format: {output_suffix}. "
                           f"Use .csv, .sdf, or .pkl")
        
        if args.verbose:
            print(f"Results saved successfully!")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())