# PYTHON_ARGCOMPLETE_OK
"""
PoseBusters CLI Wrapper for Counting Failed Pose Quality Checks

This script wraps the PoseBusters command-line tool to evaluate docking poses
against a protein structure.

Requirements:
- RDKit installed with Python bindings
- PoseBusters CLI installed and accessible in PATH
- Properly formatted PDB and SDF input files

Usage:
    python posebusters_cli.py -p protein.pdb -l poses.sdf -i Ligand_ID -o results.tsv -v
"""

import argparse
from rich_argparse import ArgumentDefaultsRichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter
import logging
from tqdm import tqdm
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def validate_inputs(protein_path: Path, poses_path: Path) -> None:
    """Validate that input files exist and have correct extensions."""
    if not protein_path.exists():
        raise FileNotFoundError(f"Protein file not found: {protein_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_path}")
    
    if protein_path.suffix.lower() != ".pdb":
        logger.warning(f"Protein file does not have .pdb extension: {protein_path}")
    if poses_path.suffix.lower() != ".sdf":
        logger.warning(f"Poses file does not have .sdf extension: {poses_path}")


def load_poses(
    poses_path: Path,
    id_col: str
) -> pd.DataFrame:
    """
    Load docking poses from SDF file into a DataFrame.
    
    Args:
        poses_path: Path to the SDF file containing docking poses
        id_col: Name of the property to use as ligand identifier
        
    Returns:
        DataFrame with 'mol' and id_col columns
    """
    suppl = Chem.SDMolSupplier(str(poses_path), removeHs=False)
    poses = []
    
    for idx, mol in tqdm(enumerate(suppl), desc="Loading poses", unit="mol"):
        if mol is None:
            logger.warning(f"Failed to load molecule at index {idx}")
            continue
            
        # Try to get ID from specified property, fall back to _Name
        props = mol.GetPropsAsDict()
        ligand_id = props.get(id_col)
        
        if ligand_id is None and mol.HasProp("_Name"):
            ligand_id = mol.GetProp("_Name")
            
        if ligand_id is None:
            ligand_id = f"mol_{idx}"
            logger.warning(f"No identifier found for molecule {idx}, using '{ligand_id}'")
            
        poses.append({"mol": mol, id_col: str(ligand_id).strip()})
    
    if not poses:
        raise ValueError(f"No valid molecules loaded from {poses_path}")
        
    logger.info(f"Loaded {len(poses)} molecules from {poses_path}")
    return pd.DataFrame(poses)


def write_poses_to_sdf(
    poses_df: pd.DataFrame,
    output_path: Path,
    mol_col: str = "mol",
    id_col: str = "Ligand_ID"
) -> None:
    """
    Write molecules to an SDF file, setting _Name property to ligand ID.
    
    Args:
        poses_df: DataFrame containing molecules and identifiers
        output_path: Path to write the SDF file
        mol_col: Name of the column containing RDKit mol objects
        id_col: Name of the column containing ligand identifiers
    """
    with Chem.SDWriter(str(output_path)) as writer:
        for _, row in poses_df.iterrows():
            mol = row[mol_col]
            ligand_id = str(row[id_col]).strip()
            mol.SetProp("_Name", ligand_id)
            writer.write(mol)


def parse_posebusters_output(csv_path: Path) -> dict[str, int]:
    """
    Parse PoseBusters CSV output and count failed criteria per molecule.
    
    Args:
        csv_path: Path to the PoseBusters CSV output file
        
    Returns:
        Dictionary mapping molecule name to number of failed checks
    """
    df = pd.read_csv(csv_path)
    
    # Columns to exclude from quality check counting
    exclude_cols = {"file", "molecule", "mol_pred_loaded", "mol_cond_loaded"}
    criteria_cols = [c for c in df.columns if c not in exclude_cols]
    
    if not criteria_cols:
        raise ValueError("No quality criteria columns found in PoseBusters output")
    
    logger.info(f"Evaluating {len(criteria_cols)} quality criteria")
    
    # Count False values (failed checks) per row
    df["fail_count"] = df[criteria_cols].apply(
        lambda row: sum(str(v).strip().lower() == "false" for v in row),
        axis=1
    )
    
    return dict(zip(df["molecule"].astype(str), df["fail_count"]))


def run_posebusters(
    poses_sdf: Path,
    protein_pdb: Path,
    output_csv: Path,
    bust_executable: str = "bust"
) -> None:
    """
    Run the PoseBusters CLI tool.
    
    Args:
        poses_sdf: Path to the poses SDF file
        protein_pdb: Path to the protein PDB file
        output_csv: Path to write CSV output
        bust_executable: Name or path of the bust executable
    """
    cmd = [
        bust_executable,
        str(poses_sdf),
        "-p", str(protein_pdb),
        "--outfmt", "csv",
        "--output", str(output_csv)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        logger.debug(f"stdout: {result.stdout}")
    if result.stderr:
        logger.debug(f"stderr: {result.stderr}")
        
    if result.returncode != 0:
        raise RuntimeError(
            f"PoseBusters failed with return code {result.returncode}\n"
            f"stderr: {result.stderr}"
        )


def run_analysis(
    protein_path: Path,
    poses_path: Path,
    id_col: str,
    output_path: Path,
    bust_executable: str = "bust",
    full_report_path: Optional[Path] = None
) -> None:
    """
    Run the complete PoseBusters analysis pipeline.
    
    Args:
        protein_path: Path to the protein PDB file
        poses_path: Path to the docking poses SDF file
        id_col: Name of the ligand identifier column
        output_path: Path to write output TSV
        bust_executable: Name or path of the bust executable
    """
    validate_inputs(protein_path, poses_path)
    
    # Load poses
    poses_df = load_poses(poses_path, id_col)
    
    # Create temporary files for PoseBusters
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_poses = tmpdir / "poses.sdf"
        tmp_output = tmpdir / "bust_output.csv"
        
        # Write poses with _Name set to ID
        write_poses_to_sdf(poses_df, tmp_poses, mol_col="mol", id_col=id_col)
        
        # Run PoseBusters
        run_posebusters(tmp_poses, protein_path, tmp_output, bust_executable)
        
        # Parse results
        fail_counts = parse_posebusters_output(tmp_output)
        logger.info(f"Parsed results for {len(fail_counts)} molecules")

        # Save full report if requested
        if full_report_path is not None:
            import shutil
            shutil.copy(tmp_output, full_report_path)
            logger.info(f"Full report saved to {full_report_path}")
    
    # Map fail counts back to poses
    poses_df["posebusters_fail_count"] = poses_df[id_col].apply(
        lambda x: fail_counts.get(str(x).strip(), pd.NA)
    )
    poses_df["posebusters_fail_count"] = poses_df["posebusters_fail_count"].astype("Int64")
    
    # Write output
    poses_df[[id_col, "posebusters_fail_count"]].to_csv(
        output_path, sep="\t", index=False
    )
    logger.info(f"Results written to {output_path}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PoseBusters quality checks on docking poses",
        formatter_class=ArgumentDefaultsRichHelpFormatter
    )
    
    parser.add_argument(
        "-p", "--protein",
        type=Path,
        required=True,
        help="Path to the protein structure file (PDB format)"
    ).completer = FilesCompleter(allowednames=(".pdb",))
    parser.add_argument(
        "-l", "--ligands",
        type=Path,
        required=True,
        help="Path to the docking poses file (SDF format)"
    ).completer = FilesCompleter(allowednames=(".sdf",))
    parser.add_argument(
        "-i", "--id-column",
        type=str,
        default="_Name",
        help="Name of the property identifying each ligand/pose"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to write the output TSV file"
    ).completer = FilesCompleter(allowednames=(".tsv",))
    parser.add_argument(
        "--bust-executable",
        type=str,
        default="bust",
        help="Name or path of the PoseBusters executable"
    )
    parser.add_argument(
        "--full-report",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save the full PoseBusters CSV report (all criteria columns) to this path"
    ).completer = FilesCompleter(allowednames=(".csv",))

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug output"
    )

    argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        run_analysis(
            protein_path=args.protein,
            poses_path=args.ligands,
            id_col=args.id_column,
            output_path=args.output,
            bust_executable=args.bust_executable,
            full_report_path=args.full_report
        )
        return 0
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except ValueError as e:
        logger.error(str(e))
        return 2
    except RuntimeError as e:
        logger.error(str(e))
        return 3
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())