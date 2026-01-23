#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import os
import math
import argparse
from rich_argparse import RichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter
import csv
from rdkit import Chem

def get_property_value(mol, property_name, ascending=True):
    try:
        return float(mol.GetProp(property_name))
    except:
        # Treat missing/invalid values as worst
        return float('inf') if ascending else float('-inf')

def read_molecules(input_file):
    """Read molecules from SDF or SMILES file."""
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext == '.sdf':
        suppl = Chem.SDMolSupplier(input_file, sanitize=False, removeHs=False)
        mols = [m for m in suppl if m is not None]
        return mols, 'sdf'
    
    elif ext in ('.smi', '.smiles', '.csv', '.tsv'):
        mols = []
        
        # Detect delimiter
        delimiter = '\t' if ext == '.tsv' else None  # None = auto-detect for csv
        
        with open(input_file, 'r') as f:
            # Peek at first line to check for header
            first_line = f.readline().strip()
            f.seek(0)
            
            # Try to detect if it's a CSV/TSV with header
            if ext in ('.csv', '.tsv'):
                if delimiter is None:
                    dialect = csv.Sniffer().sniff(first_line)
                    delimiter = dialect.delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter)
                
                # Find SMILES column
                smiles_col = None
                for col in reader.fieldnames:
                    if col.lower() in ('smiles', 'smi', 'canonical_smiles', 'molecule'):
                        smiles_col = col
                        break
                
                if smiles_col is None:
                    # Assume first column is SMILES
                    smiles_col = reader.fieldnames[0]
                
                for row in reader:
                    smiles = row[smiles_col]
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is not None:
                        # Store all other columns as properties
                        for key, value in row.items():
                            if key != smiles_col:
                                mol.SetProp(key, str(value))
                        # Also store SMILES as property
                        mol.SetProp('SMILES', smiles)
                        mols.append(mol)
            
            else:
                # Simple .smi/.smiles format: SMILES [name] or just SMILES per line
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    smiles = parts[0]
                    name = parts[1] if len(parts) > 1 else ''
                    
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is not None:
                        mol.SetProp('SMILES', smiles)
                        if name:
                            mol.SetProp('_Name', name)
                        mols.append(mol)
        
        return mols, 'smiles'
    
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .sdf, .smi, .smiles, .csv, or .tsv")

def write_molecules(mols, output_file, input_format, header=True):
    """Write molecules to SDF or SMILES file based on output extension."""
    ext = os.path.splitext(output_file)[1].lower()
    
    if ext == '.sdf':
        writer = Chem.SDWriter(output_file)
        for mol in mols:
            writer.write(mol)
        writer.close()
    
    elif ext in ('.smi', '.smiles'):
        with open(output_file, 'w') as f:
            for mol in mols:
                try:
                    smiles = mol.GetProp('SMILES')
                except:
                    smiles = Chem.MolToSmiles(mol)
                
                try:
                    name = mol.GetProp('_Name')
                    f.write(f"{smiles}\t{name}\n")
                except:
                    f.write(f"{smiles}\n")
    
    elif ext in ('.csv', '.tsv'):
        delimiter = '\t' if ext == '.tsv' else ','
        
        # Collect all property names
        all_props = set()
        for mol in mols:
            all_props.update(mol.GetPropsAsDict().keys())
        
        # Ensure SMILES is first
        prop_order = ['SMILES'] if 'SMILES' in all_props else []
        prop_order += sorted(p for p in all_props if p != 'SMILES')
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=prop_order, delimiter=delimiter)
            if header:
                writer.writeheader()
            
            for mol in mols:
                row = mol.GetPropsAsDict()
                if 'SMILES' not in row:
                    row['SMILES'] = Chem.MolToSmiles(mol)
                writer.writerow({k: row.get(k, '') for k in prop_order})
    
    else:
        raise ValueError(f"Unsupported output format: {ext}. Use .sdf, .smi, .smiles, .csv, or .tsv")

def filter_top_ligands(input_file, output_file, property_name, ascending=True,
                       top_percent=None, top_count=None, cutoff=None, header=True):
    
    mols, input_format = read_molecules(input_file)

    if not mols:
        print("[ERROR] No valid molecules found in input file.")
        return

    # Sort by specified property
    mols.sort(key=lambda m: get_property_value(m, property_name, ascending), reverse=not ascending)

    # Determine which molecules to select
    if cutoff is not None:
        if ascending:
            top_mols = [m for m in mols if get_property_value(m, property_name, ascending) <= cutoff]
            selection_desc = f"ligands with {property_name} <= {cutoff}"
        else:
            top_mols = [m for m in mols if get_property_value(m, property_name, ascending) >= cutoff]
            selection_desc = f"ligands with {property_name} >= {cutoff}"
        num_to_select = len(top_mols)
    elif top_count is not None:
        num_to_select = min(top_count, len(mols))
        top_mols = mols[:num_to_select]
        selection_desc = f"top {num_to_select} ligands"
    elif top_percent is not None:
        num_to_select = max(1, math.ceil(len(mols) * top_percent / 100))
        top_mols = mols[:num_to_select]
        selection_desc = f"top {top_percent}% ({num_to_select} ligands)"
    else:
        print("[ERROR] Must specify either --percent, --count, or --cutoff.")
        return

    if not top_mols:
        print(f"[WARNING] No molecules met the selection criteria.")
        return

    sort_order = "ascending (lowest = best)" if ascending else "descending (highest = best)"
    
    write_molecules(top_mols, output_file, input_format, header=header)

    print(f"Input format: {input_format.upper()}")
    print(f"Total molecules read: {len(mols)}")
    print(f"Sorted by: {property_name} ({sort_order})")
    print(f"Selected: {selection_desc} ({num_to_select} molecules)")
    print(f"Filtered ligands written to: {output_file}")

def list_properties(input_file):
    """List all available properties in the input file."""
    mols, input_format = read_molecules(input_file)
    
    all_props = set()
    numeric_props = set()
    
    for mol in mols:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        all_props.update(props.keys())
        
        for key, value in props.items():
            try:
                float(value)
                numeric_props.add(key)
            except (ValueError, TypeError):
                pass
    
    print(f"Input format: {input_format.upper()}")
    print(f"Total molecules: {len(mols)}")
    print(f"\nNumeric properties (can be used for sorting):")
    for prop in sorted(numeric_props):
        print(f"  - {prop}")
    
    non_numeric = all_props - numeric_props
    if non_numeric:
        print(f"\nNon-numeric properties:")
        for prop in sorted(non_numeric):
            print(f"  - {prop}")

def main():
    parser = argparse.ArgumentParser(
        description="Filter top ligands from an SDF or SMILES file by a specified property.",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input filename (.sdf, .smi, .smiles, .csv, or .tsv)"
    ).completer = FilesCompleter(allowednames=(".sdf", ".smi", ".smiles", ".csv", ".tsv"))
    parser.add_argument(
        "-o", "--output",
        help="Output filename (.sdf, .smi, .smiles, .csv, or .tsv)"
    ).completer = FilesCompleter(allowednames=(".sdf", ".smi", ".smiles", ".csv", ".tsv"))
    parser.add_argument(
        "-s", "--sort-by",
        default="minimizedAffinity",
        help="Property name to sort by (default: minimizedAffinity)"
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending (highest = best). Default is ascending (lowest = best)."
    )
    parser.add_argument(
        "--list-properties",
        action="store_true",
        help="List all available properties in the input file and exit"
    )
    
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit header row in CSV/TSV output"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-p", "--percent",
        type=float,
        help="Select top X percent of ligands (e.g., 10 for top 10%%)"
    )
    group.add_argument(
        "-n", "--count",
        type=int,
        help="Select top N ligands by count"
    )
    group.add_argument(
        "-c", "--cutoff",
        type=float,
        help="Select all ligands below (ascending) or above (descending) this cutoff value"
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    # If listing properties, do that and exit
    if args.list_properties:
        list_properties(args.input)
        return

    # For filtering, we need output and a selection method
    if not args.output:
        parser.error("--output is required when filtering (not using --list-properties)")
    if args.percent is None and args.count is None and args.cutoff is None:
        parser.error("Must specify either --percent, --count, or --cutoff when filtering")

    # Validate output file
    if args.output == args.input:
        parser.error("Output filename must be different from input filename.")

    # Validate percent/count values
    if args.percent is not None and not (0 < args.percent <= 100):
        parser.error("Percent must be between 0 and 100.")
    if args.count is not None and args.count <= 0:
        parser.error("Count must be a positive integer.")

    filter_top_ligands(
        args.input,
        args.output,
        property_name=args.sort_by,
        ascending=not args.descending,
        top_percent=args.percent,
        top_count=args.count,
        cutoff=args.cutoff,
        header=not args.no_header
    )

if __name__ == "__main__":
    main()
