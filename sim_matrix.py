#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
SMILES Similarity Matrix Generator

Generates a high-resolution similarity matrix plot for a set of SMILES strings
using molecular fingerprints and Tanimoto similarity.

Requirements:
    pip install rdkit matplotlib numpy seaborn

Usage:
    # From command line with a SMILES file:
    python smiles_similarity_matrix.py -i smiles.txt -o matrix.png
    python smiles_similarity_matrix.py -i smiles.csv --smiles-col SMILES --label-col Name
    python smiles_similarity_matrix.py -i smiles.txt -o matrix.png --dpi 600 --figsize 20 20
    python smiles_similarity_matrix.py -i smiles.txt -o matrix.png --font-size 14
    
    # Or import and use the functions directly:
    from smiles_similarity_matrix import plot_similarity_matrix
    plot_similarity_matrix(smiles_list, labels=None, output_file="similarity.png")
"""

import argparse
from rich_argparse import RawDescriptionRichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
from typing import Optional
from pathlib import Path

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')

# Custom colormap: black -> purple -> red -> orange -> light yellow
SIMILARITY_CMAP = LinearSegmentedColormap.from_list(
    'similarity',
    ['#000000', '#4B0082', '#8B0000', '#FF4500', '#FFA500', '#FFFFE0'],
    N=256
)


def smiles_to_fingerprint(smiles: str, fp_type: str = 'morgan', radius: int = 2, n_bits: int = 2048):
    """
    Convert a SMILES string to a molecular fingerprint.

    Args:
        smiles: SMILES string representation of a molecule
        fp_type: Fingerprint type ('morgan', 'maccs', or 'rdkit')
        radius: Radius for Morgan fingerprint (default: 2, equivalent to ECFP4)
        n_bits: Number of bits in the fingerprint

    Returns:
        Fingerprint or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if fp_type == 'morgan':
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    elif fp_type == 'maccs':
        return AllChem.GetMACCSKeysFingerprint(mol)
    elif fp_type == 'rdkit':
        return Chem.RDKFingerprint(mol, fpSize=n_bits)
    raise ValueError(f"Unknown fingerprint type: {fp_type}")


def calculate_similarity_matrix(smiles_list: list[str], fp_type: str = 'morgan', radius: int = 2, n_bits: int = 2048, verbose: bool = True) -> tuple[np.ndarray, list[int]]:
    """
    Calculate pairwise Tanimoto similarity matrix for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        radius: Radius for Morgan fingerprint
        n_bits: Number of bits in fingerprint
        verbose: Print progress for large datasets
        
    Returns:
        Tuple of (similarity matrix, list of valid indices)
    """
    fingerprints = []
    valid_indices = []
    
    n_total = len(smiles_list)
    if verbose and n_total > 100:
        print(f"Generating fingerprints for {n_total} molecules...")
    
    for i, smiles in enumerate(smiles_list):
        fp = smiles_to_fingerprint(smiles, fp_type, radius, n_bits)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)
        else:
            print(f"Warning: Invalid SMILES at index {i}: {smiles[:50]}...")
    
    n = len(fingerprints)
    if verbose:
        print(f"Valid molecules: {n}/{n_total}")
        if n > 100:
            print(f"Calculating {n*n:,} pairwise similarities...")
    
    similarity_matrix = np.zeros((n, n))
    
    # Use bulk similarity calculation for better performance
    for i in range(n):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints)
        similarity_matrix[i, :] = similarities
        
        # Progress indicator for large matrices
        if verbose and n > 100 and (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{n} rows ({100*(i+1)/n:.1f}%)")
    
    if verbose and n > 100:
        print("Similarity calculation complete.")
    
    return similarity_matrix, valid_indices


def plot_similarity_matrix(
    smiles_list: list[str],
    labels: Optional[list[str]] = None,
    output_file: str = "similarity_matrix.png",
    output_format: Optional[str] = None,  # 'png' or 'svg', auto-detected if None
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    cmap = None,  # Uses custom black-purple-red-orange-yellow gradient
    title: str = "Molecular Similarity Matrix (Tanimoto)",
    annot: bool = None,  # Auto-determined based on matrix size
    annot_fontsize: int = 8,
    show_plot: bool = False,
    fp_type: str = 'morgan',
    radius: int = 2,
    n_bits: int = 2048,
    show_labels: bool = None,  # Auto-determined based on matrix size
    tick_interval: int = None,  # Show labels at this interval for large matrices
    xlabel: str = "Molecules",
    ylabel: str = "Molecules",
    font_size: int = None,  # Base font size; overrides auto-scaling when set
    verbose: bool = True
) -> np.ndarray:
    """
    Generate and save a high-resolution similarity matrix heatmap.
    
    Args:
        smiles_list: List of SMILES strings
        labels: Optional labels for molecules (defaults to indices or truncated SMILES)
        output_file: Path to save the output image
        output_format: Output format ('png' or 'svg'). Auto-detected from filename if None
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch (300 for high quality, ignored for SVG)
        cmap: Colormap for the heatmap
        title: Plot title
        annot: Whether to annotate cells with values
        annot_fontsize: Font size for annotations
        show_plot: Whether to display the plot interactively
        radius: Morgan fingerprint radius
        n_bits: Number of fingerprint bits
        
        show_labels: Whether to show axis labels (auto-determined if None)
        font_size: Base font size in points. When set, overrides the auto-scaled
            font sizes for axis labels, colorbar, tick labels, and annotations.
            When None (default), sizes are auto-scaled based on figure dimensions.
        verbose: Print progress messages
        
    Returns:
        The similarity matrix as a numpy array
    """
    # Calculate similarity matrix
    similarity_matrix, valid_indices = calculate_similarity_matrix(
        smiles_list, fp_type, radius, n_bits, verbose=verbose
    )
    
    n = len(similarity_matrix)
    
    # Generate labels if not provided
    if labels is None:
        labels = [f"Mol {i+1}" for i in valid_indices]
    else:
        labels = [labels[i] for i in valid_indices]
    
    # Use custom colormap if none specified
    if cmap is None:
        cmap = SIMILARITY_CMAP
    
    # Auto-determine annotation and label settings based on matrix size
    if annot is None:
        annot = n <= 20
    if show_labels is None:
        show_labels = n <= 100
    
    if n > 15 and annot:
        annot_fontsize = max(4, 8 - (n - 15) // 5)
    
    # Auto-scale figure size for large matrices (capped to avoid memory issues)
    if n > 50:
        scale = min(3, max(1, n / 50))  # Cap at 3x to avoid memory issues
        figsize = (int(figsize[0] * scale), int(figsize[1] * scale))
        if verbose:
            print(f"Auto-scaled figure size to {figsize} for {n} molecules")
    
    if verbose:
        print(f"Generating plot...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine tick labels based on settings
    if tick_interval is not None:
        # Show labels at regular intervals (numeric indices)
        x_tick_labels = [str(i) if i % tick_interval == 0 else '' for i in range(n)]
        y_tick_labels = [str(i) if i % tick_interval == 0 else '' for i in range(n)]
    elif show_labels:
        x_tick_labels = labels
        y_tick_labels = labels
    else:
        x_tick_labels = False
        y_tick_labels = False
    
    # Create heatmap with horizontal colorbar above
    heatmap = sns.heatmap(
        similarity_matrix,
        xticklabels=x_tick_labels,
        yticklabels=y_tick_labels,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=annot,
        fmt=".2f",
        annot_kws={"fontsize": annot_fontsize},
        square=True,
        linewidths=0.5 if n <= 50 else 0,  # Remove lines for large matrices
        linecolor='white',
        cbar=False,  # We'll add colorbar manually
        ax=ax
    )
    
    # Add horizontal colorbar above the heatmap, centered with label on left
    plt.tight_layout()
    pos = ax.get_position()
    
    # Scale colorbar dimensions based on figure size (for large matrices)
    fig_height = fig.get_size_inches()[1]
    scale_factor = max(1, fig_height / 10)  # Base scale on 10-inch figure
    
    cbar_height = 0.06 / scale_factor  # Doubled again from 0.03
    cbar_pad = 0.03 / scale_factor
    legend_fontsize = max(14, int(14 * scale_factor))  # For Similarity label
    tick_fontsize = max(12, int(12 * scale_factor))    # For colorbar ticks
    axes_label_fontsize = max(16, int(16 * scale_factor))  # For "Molecules" labels

    # Override auto-scaled font sizes if a base font_size is specified
    if font_size is not None:
        legend_fontsize = font_size
        tick_fontsize = font_size
        axes_label_fontsize = font_size
        annot_fontsize = font_size
    
    # Calculate centered position for label + colorbar together
    cbar_width = pos.width * 0.4
    label_width = 0.06  # Approximate width for "Similarity" text
    total_width = label_width + cbar_width
    start_x = pos.x0 + (pos.width - total_width) / 2  # Center the combo
    
    cbar_ax = fig.add_axes([start_x + label_width,  # Colorbar starts after label
                            pos.y1 + cbar_pad,
                            cbar_width,
                            cbar_height])
    cbar = fig.colorbar(ax.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0.0', '0.5', '1.0'])
    cbar_ax.xaxis.set_ticks_position('top')  # Ticks above colorbar
    cbar_ax.tick_params(labelsize=tick_fontsize)
    # Ensure tick labels are horizontal and centered
    plt.setp(cbar_ax.xaxis.get_ticklabels(), rotation=0, ha='center')
    
    # Add 'Similarity' label to the left of the colorbar (aligned with bottom of colorbar)
    fig.text(start_x - 0.02, pos.y1 + cbar_pad, 'Similarity', 
             fontsize=legend_fontsize, va='bottom', ha='left')
    
    # Style adjustments (no title)
    ax.set_xlabel(xlabel, fontsize=axes_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axes_label_fontsize)
    
    # Rotate x-axis labels for readability (only main axes, not colorbar)
    if show_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=max(4, 10 - n // 20))
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=max(4, 10 - n // 20))
    elif tick_interval is not None:
        # For interval ticks, use larger readable font and no rotation
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=tick_fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=tick_fontsize)
    
    # Determine output format
    if output_format is None:
        # Auto-detect from filename
        suffix = Path(output_file).suffix.lower()
        if suffix == '.svg':
            output_format = 'svg'
        elif suffix == '.pdf':
            output_format = 'pdf'
        else:
            output_format = 'png'
    
    # Ensure correct file extension
    output_path = Path(output_file)
    expected_suffix = f'.{output_format}'
    if output_path.suffix.lower() != expected_suffix:
        output_file = str(output_path.with_suffix(expected_suffix))
    
    # Save figure
    if output_format == 'svg':
        plt.savefig(output_file, format='svg', bbox_inches='tight', facecolor='white')
    elif output_format == 'pdf':
        plt.savefig(output_file, format='pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    else:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    print(f"Saved {output_format.upper()} plot to: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return similarity_matrix


def load_smiles_file(filepath: str, smiles_col: str = None, label_col: str = None, delimiter: str = None) -> tuple[list[str], Optional[list[str]]]:
    """
    Load SMILES from a file. Supports:
    - Plain text files (one SMILES per line)
    - CSV/TSV files with headers
    
    Args:
        filepath: Path to the input file
        smiles_col: Column name for SMILES (for CSV/TSV with headers)
        label_col: Column name for labels (optional)
        delimiter: Column delimiter (auto-detected if None)
        
    Returns:
        Tuple of (smiles_list, labels or None)
    """
    path = Path(filepath)
    
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError(f"Empty file: {filepath}")
    
    # Auto-detect delimiter
    if delimiter is None:
        first_line = lines[0]
        if '\t' in first_line:
            delimiter = '\t'
        elif ',' in first_line:
            delimiter = ','
        else:
            delimiter = None  # Plain text, one SMILES per line
    
    # Check if first line looks like a header
    has_header = False
    if delimiter and smiles_col:
        has_header = True
    elif delimiter:
        # Heuristic: if first line contains common header words
        first_lower = lines[0].lower()
        has_header = any(word in first_lower for word in ['smiles', 'molecule', 'name', 'id', 'compound'])
    
    smiles_list = []
    labels = None
    
    if delimiter and has_header:
        # CSV/TSV with header
        header = lines[0].split(delimiter)
        header_lower = [h.lower().strip() for h in header]
        
        # Find SMILES column
        if smiles_col:
            try:
                smiles_idx = header_lower.index(smiles_col.lower())
            except ValueError:
                raise ValueError(f"Column '{smiles_col}' not found. Available: {header}")
        else:
            # Try common SMILES column names
            for name in ['smiles', 'smi', 'canonical_smiles', 'molecule']:
                if name in header_lower:
                    smiles_idx = header_lower.index(name)
                    break
            else:
                smiles_idx = 0  # Default to first column
        
        # Find label column
        label_idx = None
        if label_col:
            try:
                label_idx = header_lower.index(label_col.lower())
                labels = []
            except ValueError:
                print(f"Warning: Label column '{label_col}' not found, skipping labels")
        
        # Parse data rows
        for line in lines[1:]:
            parts = line.split(delimiter)
            if len(parts) > smiles_idx:
                smiles_list.append(parts[smiles_idx].strip())
                if label_idx is not None and len(parts) > label_idx:
                    labels.append(parts[label_idx].strip())
    
    elif delimiter:
        # CSV/TSV without header - assume first column is SMILES
        for line in lines:
            parts = line.split(delimiter)
            smiles_list.append(parts[0].strip())
    
    else:
        # Plain text, one SMILES per line
        smiles_list = lines
    
    return smiles_list, labels


def load_smiles_from_sdf(filepath: str, label_prop: Optional[str] = None) -> tuple[list[str], Optional[list[str]]]:
    """Load SMILES from an SDF file, optionally extracting a label property."""
    suppl = Chem.SDMolSupplier(filepath)
    smiles_list = []
    labels = [] if label_prop else None
    for mol in suppl:
        if mol is None:
            continue
        smiles_list.append(Chem.MolToSmiles(mol))
        if label_prop is not None:
            labels.append(mol.GetProp(label_prop) if mol.HasProp(label_prop) else '')
    return smiles_list, labels


def save_matrix(matrix: np.ndarray, output_file: str, labels: Optional[list[str]] = None):
    """Save the similarity matrix to a CSV file."""
    csv_path = Path(output_file).with_suffix('.csv')
    
    if labels:
        header = ',' + ','.join(labels)
        with open(csv_path, 'w') as f:
            f.write(header + '\n')
            for i, row in enumerate(matrix):
                f.write(labels[i] + ',' + ','.join(f'{v:.4f}' for v in row) + '\n')
    else:
        np.savetxt(csv_path, matrix, delimiter=',', fmt='%.4f')
    
    print(f"Saved similarity matrix to: {csv_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate a high-resolution similarity matrix plot for SMILES strings.',
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  # Simple text file (one SMILES per line):
  python smiles_similarity_matrix.py -i molecules.txt -o matrix.png

  # CSV file with headers:
  python smiles_similarity_matrix.py -i data.csv --smiles-col SMILES --label-col Name

  # Publication-ready figure (single column, 600 DPI):
  python smiles_similarity_matrix.py -i molecules.csv -o figure1.png --figsize 7 7 --dpi 600

  # PDF output for manuscripts:
  python smiles_similarity_matrix.py -i molecules.csv -o figure1.pdf --dpi 600

  # Large dataset (500+ molecules):
  python smiles_similarity_matrix.py -i large_dataset.csv -o matrix.png --no-labels --save-matrix
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input file containing SMILES (txt, csv, or tsv)'
                        ).completer = FilesCompleter(allowednames=(".txt", ".csv", ".tsv", ".smi", ".smiles"))
    parser.add_argument('-o', '--output', default='similarity_matrix.png',
                        help='Output image file (default: similarity_matrix.png)'
                        ).completer = FilesCompleter(allowednames=(".png", ".svg", ".pdf"))
    parser.add_argument('--format', choices=['png', 'svg', 'pdf'], default=None,
                        help='Output format (png, svg, or pdf). Auto-detected from output filename if not specified. PNG recommended for manuscripts.')
    parser.add_argument('--smiles-col', default=None,
                        help='Column name for SMILES in CSV/TSV (auto-detected if not specified)')
    parser.add_argument('--label-col', default=None,
                        help='Column name for molecule labels in CSV/TSV')
    parser.add_argument('--delimiter', default=None,
                        help='Column delimiter (auto-detected if not specified)')
    
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output resolution in DPI (default: 300). Note: 600 DPI with large matrices (500+) may cause memory issues.')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 10],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches (default: 12 10)')
    parser.add_argument('--title', default='Molecular Similarity Matrix (Tanimoto)',
                        help='Plot title')
    
    parser.add_argument('--fp-type', default='morgan',
                        choices=['morgan', 'maccs', 'rdkit'],
                        help='Fingerprint type (default: morgan)')
    parser.add_argument('--radius', type=int, default=2,
                        help='Morgan fingerprint radius (default: 2, i.e., ECFP4)')
    parser.add_argument('--nbits', type=int, default=2048,
                        help='Fingerprint bit size (default: 2048)')
    
    parser.add_argument('--no-labels', action='store_true',
                        help='Hide axis labels (recommended for >100 molecules)')
    parser.add_argument('--no-annot', action='store_true',
                        help='Hide cell value annotations')
    parser.add_argument('--show-labels', action='store_true',
                        help='Force showing axis labels even for large matrices')
    parser.add_argument('--show-annot', action='store_true',
                        help='Force showing annotations even for large matrices')
    parser.add_argument('--tick-interval', type=int, default=None,
                        help='Show axis tick labels at this interval (e.g., 50 shows every 50th molecule). Useful for large matrices.')
    parser.add_argument('--xlabel', default='Molecules',
                        help='Label for x-axis (default: Molecules)')
    parser.add_argument('--ylabel', default='Molecules',
                        help='Label for y-axis (default: Molecules)')
    parser.add_argument('--font-size', type=int, default=None,
                        help='Base font size in points for all text elements (axis labels, '
                             'colorbar, tick labels, annotations). Overrides auto-scaling. '
                             'Recommended range: 8–24 (default: auto)')
    
    parser.add_argument('--save-matrix', action='store_true',
                        help='Also save the similarity matrix as CSV')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')
    
    argcomplete.autocomplete(parser)
    return parser.parse_args()


def main():
    """Main entry point for CLI usage."""
    args = parse_args()
    
    # Load SMILES from file
    if not args.quiet:
        print(f"Loading SMILES from: {args.input}")

    if args.input.lower().endswith('.sdf'):
        smiles_list, labels = load_smiles_from_sdf(args.input, label_prop=args.label_col)
    else:
        smiles_list, labels = load_smiles_file(
            args.input,
            smiles_col=args.smiles_col,
            label_col=args.label_col,
            delimiter=args.delimiter
        )
    
    if not args.quiet:
        print(f"Loaded {len(smiles_list)} SMILES strings")
    
    # Determine annotation/label settings
    annot = None  # Auto-determine
    show_labels = None  # Auto-determine
    
    if args.no_annot:
        annot = False
    elif args.show_annot:
        annot = True
        
    if args.no_labels:
        show_labels = False
    elif args.show_labels:
        show_labels = True
    
    # Generate the plot
    similarity_matrix = plot_similarity_matrix(
        smiles_list=smiles_list,
        labels=labels,
        output_file=args.output,
        output_format=args.format,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        title=args.title,
        annot=annot,
        show_plot=False,
        fp_type=args.fp_type,
        radius=args.radius,
        n_bits=args.nbits,
        show_labels=show_labels,
        tick_interval=args.tick_interval,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        font_size=args.font_size,
        verbose=not args.quiet
    )
    
    # Optionally save the matrix as CSV
    if args.save_matrix:
        save_matrix(similarity_matrix, args.output, labels)
    
    if not args.quiet:
        print(f"\nDone! Output saved to: {args.output}")


if __name__ == "__main__":
    main()