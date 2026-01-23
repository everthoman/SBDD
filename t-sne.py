#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Generate publication-quality 2D t-SNE plots from SMILES strings.

This script converts SMILES to molecular fingerprints and visualizes
chemical space using t-SNE dimensionality reduction.

Requirements:
    pip install rdkit numpy scikit-learn matplotlib pandas seaborn

Usage:
    python tsne_smiles.py --input smiles.csv --output tsne_plot.png
    
    Or import and use programmatically:
        from tsne_smiles import plot_tsne_from_smiles
        fig = plot_tsne_from_smiles(smiles_list)
"""

import argparse
from rich_argparse import RawDescriptionRichHelpFormatter
import argcomplete
from argcomplete.completers import FilesCompleter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
import warnings

# Suppress RDKit warnings for cleaner output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Suppress sklearn data conversion warning for jaccard metric
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings('ignore', category=DataConversionWarning)


def smiles_to_fingerprints(smiles_list, fp_type='morgan', radius=2, n_bits=2048):
    """
    Convert SMILES strings to molecular fingerprints.
    
    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings
    fp_type : str
        Fingerprint type: 'morgan' (ECFP-like), 'maccs', or 'rdkit'
    radius : int
        Radius for Morgan fingerprints (default 2 = ECFP4)
    n_bits : int
        Number of bits for fingerprint (Morgan and RDKit only)
    
    Returns
    -------
    fingerprints : np.ndarray
        Array of fingerprints (n_molecules, n_bits)
    valid_indices : list
        Indices of valid SMILES that were successfully parsed
    valid_smiles : list
        List of valid SMILES strings
    """
    fingerprints = []
    valid_indices = []
    valid_smiles = []
    
    for i, smi in enumerate(smiles_list):
        # Skip NaN, None, or non-string values
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            continue
        if not isinstance(smi, str):
            try:
                smi = str(smi)
            except:
                warnings.warn(f"Could not convert SMILES at index {i} to string")
                continue
        
        smi = smi.strip()
        if not smi:  # Skip empty strings
            continue
            
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            warnings.warn(f"Could not parse SMILES at index {i}: {smi}")
            continue
        
        if fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        elif fp_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        arr = np.zeros((len(fp),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
        valid_indices.append(i)
        valid_smiles.append(smi)
    
    return np.array(fingerprints), valid_indices, valid_smiles


def compute_tsne(fingerprints, perplexity=30, n_iter=1000, random_state=42, 
                 learning_rate='auto', init='pca'):
    """
    Compute t-SNE embedding from fingerprints.
    
    Parameters
    ----------
    fingerprints : np.ndarray
        Fingerprint array (n_molecules, n_bits)
    perplexity : float
        t-SNE perplexity (typically 5-50, should be < n_samples)
    n_iter : int
        Number of iterations
    random_state : int
        Random seed for reproducibility
    learning_rate : str or float
        Learning rate for t-SNE
    init : str
        Initialization method ('pca' or 'random')
    
    Returns
    -------
    embedding : np.ndarray
        2D t-SNE coordinates (n_molecules, 2)
    """
    # Adjust perplexity if necessary
    n_samples = fingerprints.shape[0]
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 4)
        warnings.warn(f"Perplexity adjusted to {perplexity} (must be < n_samples)")
    
    # Handle sklearn version compatibility (n_iter vs max_iter)
    import sklearn
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    iter_param = 'max_iter' if sklearn_version >= (1, 5) else 'n_iter'
    
    tsne_params = {
        'n_components': 2,
        'perplexity': perplexity,
        iter_param: n_iter,
        'random_state': random_state,
        'learning_rate': learning_rate,
        'init': init,
        'metric': 'jaccard'  # Tanimoto-like distance for binary fingerprints
    }
    
    tsne = TSNE(**tsne_params)
    embedding = tsne.fit_transform(fingerprints)
    return embedding


def plot_tsne_from_smiles(
    smiles_list,
    labels=None,
    colors=None,
    color_values=None,
    cmap='viridis',
    fp_type='morgan',
    radius=2,
    n_bits=2048,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    figsize=(8, 8),
    dpi=300,
    point_size=50,
    alpha=0.7,
    edgecolor='white',
    linewidth=0.5,
    title=None,
    xlabel='t-SNE 1',
    ylabel='t-SNE 2',
    colorbar_label=None,
    legend_title=None,
    show_legend=True,
    font_family='DejaVu Sans',
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    style='seaborn-v0_8-whitegrid',
    # Highlight options
    highlight_category=None,
    highlight_color='#008080',
    highlight_marker='*',
    highlight_size=1.5,
    background_color='#d3d3d3',
    background_marker='o',
    background_alpha=0.5,
    # Axis limits
    xlim=None,
    ylim=None
):
    """
    Generate a publication-quality t-SNE plot from SMILES strings.
    
    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings
    labels : list, optional
        Categorical labels for coloring points
    colors : list, optional
        Direct color specification for each point
    color_values : list, optional
        Continuous values for coloring (uses colormap)
    cmap : str
        Matplotlib colormap name
    fp_type : str
        Fingerprint type: 'morgan', 'maccs', or 'rdkit'
    radius : int
        Morgan fingerprint radius
    n_bits : int
        Fingerprint bit length
    perplexity : float
        t-SNE perplexity parameter
    n_iter : int
        Number of t-SNE iterations
    random_state : int
        Random seed
    figsize : tuple
        Figure size in inches
    dpi : int
        Resolution for saved figures
    point_size : float
        Marker size
    alpha : float
        Point transparency
    edgecolor : str
        Marker edge color
    linewidth : float
        Marker edge width
    title : str, optional
        Plot title
    xlabel, ylabel : str
        Axis labels
    colorbar_label : str, optional
        Label for colorbar (continuous coloring)
    legend_title : str, optional
        Title for legend (categorical coloring)
    show_legend : bool
        Whether to show legend
    font_family : str
        Font family for text
    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize : int
        Font sizes
    style : str
        Matplotlib style
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    embedding : np.ndarray
        The t-SNE coordinates
    valid_indices : list
        Indices of successfully processed molecules
    """
    # Set up plotting style with version compatibility
    available_styles = plt.style.available
    
    # Try different style names (changed across matplotlib versions)
    style_options = [
        style,
        'seaborn-v0_8-whitegrid',
        'seaborn-whitegrid',
        'seaborn-v0_8-white',
        'seaborn-white',
        'seaborn',
        'ggplot',
        'default'
    ]
    
    style_set = False
    for s in style_options:
        if s in available_styles:
            plt.style.use(s)
            style_set = True
            break
    
    if not style_set:
        # Fall back to manual styling
        pass
    
    plt.rcParams['font.family'] = font_family
    plt.rcParams['axes.linewidth'] = 1.0
    
    # Convert SMILES to fingerprints
    print("Converting SMILES to fingerprints...")
    fingerprints, valid_indices, valid_smiles = smiles_to_fingerprints(
        smiles_list, fp_type=fp_type, radius=radius, n_bits=n_bits
    )
    print(f"Successfully processed {len(valid_indices)}/{len(smiles_list)} molecules")
    
    if len(fingerprints) < 5:
        raise ValueError("Need at least 5 valid molecules for t-SNE")
    
    # Compute t-SNE
    print("Computing t-SNE embedding...")
    embedding = compute_tsne(
        fingerprints, 
        perplexity=perplexity, 
        n_iter=n_iter, 
        random_state=random_state
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Determine coloring scheme
    if highlight_category is not None and labels is not None:
        # Highlight mode: one category highlighted, rest in background
        valid_labels = [labels[i] for i in valid_indices]
        highlight_mask = np.array([l == highlight_category for l in valid_labels])
        background_mask = ~highlight_mask
        
        # Debug output
        n_highlighted = highlight_mask.sum()
        n_background = background_mask.sum()
        print(f"Highlighting '{highlight_category}': {n_highlighted} points highlighted, {n_background} in background")
        if n_highlighted == 0:
            unique_labels = sorted(set(valid_labels))
            print(f"  Available categories: {unique_labels}")
        
        # highlight_size is a multiplier (e.g., 1.5 = 1.5x point_size)
        hl_size = point_size * highlight_size
        
        # Plot highlighted points first (for legend order), but with high zorder (to appear on top)
        if highlight_mask.any():
            ax.scatter(
                embedding[highlight_mask, 0],
                embedding[highlight_mask, 1],
                c=highlight_color,
                s=hl_size,
                alpha=alpha,
                edgecolor='white',
                linewidth=linewidth,
                marker=highlight_marker,
                label=highlight_category,
                zorder=10  # Ensure on top
            )
        
        # Plot background points second (for legend order), with lower zorder
        if background_mask.any():
            ax.scatter(
                embedding[background_mask, 0],
                embedding[background_mask, 1],
                c=background_color,
                s=point_size,
                alpha=background_alpha,
                edgecolor='white',
                linewidth=linewidth,
                marker=background_marker,
                label='Other targets',
                zorder=1
            )
        
        if show_legend:
            legend = ax.legend(
                fontsize=legend_fontsize,
                frameon=False,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.12),
                ncol=2
            )
    
    elif labels is not None:
        # Categorical coloring
        valid_labels = [labels[i] for i in valid_indices]
        unique_labels = sorted(set(valid_labels))
        color_map = plt.cm.get_cmap('tab10' if len(unique_labels) <= 10 else 'tab20')
        label_colors = {label: color_map(i / len(unique_labels)) 
                       for i, label in enumerate(unique_labels)}
        
        for label in unique_labels:
            mask = np.array([l == label for l in valid_labels])
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[label_colors[label]],
                s=point_size,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label=label
            )
        
        if show_legend:
            legend = ax.legend(
                fontsize=legend_fontsize,
                frameon=False,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.12),
                ncol=min(len(unique_labels), 5)
            )
    
    elif color_values is not None:
        # Continuous coloring
        valid_values = [color_values[i] for i in valid_indices]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=valid_values,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=label_fontsize)
    
    elif colors is not None:
        # Direct color specification
        valid_colors = [colors[i] for i in valid_indices]
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=valid_colors,
            s=point_size,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth
        )
    
    else:
        # Default single color
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c='#1f77b4',
            s=point_size,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth
        )
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='medium')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=15)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Hide top and right spines, show bottom and left as black axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)
    
    # Ensure grid is shown and includes edges
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.5, color='#cccccc')
    
    # Add border lines at the exact axis limits to complete the box
    if xlim is not None and ylim is not None:
        ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[0]], color='#cccccc', linewidth=0.5, zorder=0)  # bottom
        ax.plot([xlim[0], xlim[1]], [ylim[1], ylim[1]], color='#cccccc', linewidth=0.5, zorder=0)  # top
        ax.plot([xlim[0], xlim[0]], [ylim[0], ylim[1]], color='#cccccc', linewidth=0.5, zorder=0)  # left
        ax.plot([xlim[1], xlim[1]], [ylim[0], ylim[1]], color='#cccccc', linewidth=0.5, zorder=0)  # right
    
    # Equal aspect ratio for t-SNE
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    return fig, ax, embedding, valid_indices


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate t-SNE plot from SMILES',
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="""
Examples:
  python tsne_smiles.py --input molecules.csv --output tsne.png
  python tsne_smiles.py --input data.csv --smiles-col SMILES --label-col Activity
  python tsne_smiles.py --input data.csv --color-col pIC50 --cmap plasma
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input CSV/TSV or SDF file'
                        ).completer = FilesCompleter(allowednames=(".csv", ".tsv", ".txt", ".sdf"))
    parser.add_argument('--output', '-o', default='tsne_plot.png', help='Output file'
                        ).completer = FilesCompleter(allowednames=(".png", ".svg", ".pdf"))
    parser.add_argument('--smiles-col', default='SMILES', help='SMILES column name')
    parser.add_argument('--label-col', help='Column for categorical coloring')
    parser.add_argument('--color-col', help='Column for continuous coloring')
    parser.add_argument('--fp-type', default='morgan', choices=['morgan', 'maccs', 'rdkit'])
    parser.add_argument('--radius', type=int, default=2, help='Morgan fingerprint radius (default: 2)')
    parser.add_argument('--n-bits', type=int, default=2048, help='Fingerprint bit length (default: 2048)')
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--n-iter', type=int, default=1000)
    parser.add_argument('--cmap', default='viridis', help='Colormap for continuous values')
    parser.add_argument('--title', help='Plot title')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8, 8])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--point-size', type=float, default=50)
    parser.add_argument('--font-size', type=float, default=None,
                        help='Base font size; sets title/label/tick/legend sizes relative to this value. '
                             'Overrides individual font size defaults (title: base+2, label: base, '
                             'tick: base-2, legend: base). Default uses script defaults (12).')
    
    # Highlight options
    parser.add_argument('--highlight', help='Category value to highlight (requires --label-col)')
    parser.add_argument('--highlight-color', default='#008080', help='Color for highlighted points (default: dark teal)')
    parser.add_argument('--highlight-marker', default='*', help='Marker for highlighted points (default: star)')
    parser.add_argument('--highlight-size', type=float, default=1.5, help='Size multiplier for highlighted points (default: 1.5x point-size)')
    parser.add_argument('--background-color', default='#d3d3d3', help='Color for non-highlighted points (default: light grey)')
    parser.add_argument('--background-marker', default='o', help='Marker for non-highlighted points (default: circle)')
    parser.add_argument('--background-alpha', type=float, default=0.5, help='Alpha for non-highlighted points')
    parser.add_argument('--xlim', type=float, nargs=2, help='X-axis limits (e.g., --xlim -80 80)')
    parser.add_argument('--ylim', type=float, nargs=2, help='Y-axis limits (e.g., --ylim -80 80)')
    parser.add_argument('--no-save-coordinates', action='store_true',
                        help='Do not save t-SNE coordinates to a CSV file')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Resolve font sizes: if --font-size given, derive all sizes from it
    if args.font_size is not None:
        title_fontsize  = args.font_size + 2
        label_fontsize  = args.font_size
        tick_fontsize   = args.font_size - 2
        legend_fontsize = args.font_size
    else:
        title_fontsize  = 14
        label_fontsize  = 12
        tick_fontsize   = 10
        legend_fontsize = 10

    # Load data
    import pandas as pd

    input_ext = args.input.rsplit('.', 1)[-1].lower()

    if input_ext == 'sdf':
        # Load from SDF: generate SMILES from each molecule
        mols_sdf = [m for m in Chem.SDMolSupplier(args.input, removeHs=False) if m is not None]
        if not mols_sdf:
            raise ValueError(f"No valid molecules found in {args.input}")
        rows = []
        for mol in mols_sdf:
            row = {args.smiles_col: Chem.MolToSmiles(mol)}
            row.update(mol.GetPropsAsDict())
            rows.append(row)
        df = pd.DataFrame(rows)
        print(f"Loaded {len(df)} molecules from SDF with columns: {list(df.columns)}")
    else:
        # CSV/TSV: detect delimiter
        with open(args.input, 'r') as f:
            first_line = f.readline()
        if '\t' in first_line:
            sep = '\t'
        elif ',' in first_line:
            sep = ','
        elif ';' in first_line:
            sep = ';'
        else:
            sep = r'\s+'
        df = pd.read_csv(args.input, sep=sep, engine='python')
        df.columns = df.columns.str.strip().str.rstrip(',')
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    if args.smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{args.smiles_col}' not found in {args.input}")

    smiles_list = df[args.smiles_col].tolist()
    labels = None
    if args.label_col:
        # Strip trailing commas from label values
        labels = [str(l).strip().rstrip(',') if pd.notna(l) else l for l in df[args.label_col]]
    color_values = df[args.color_col].tolist() if args.color_col else None
    
    # Generate plot
    fig, ax, embedding, valid_indices = plot_tsne_from_smiles(
        smiles_list,
        labels=labels,
        color_values=color_values,
        cmap=args.cmap,
        fp_type=args.fp_type,
        radius=args.radius,
        n_bits=args.n_bits,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        point_size=args.point_size,
        title=args.title,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        colorbar_label=args.color_col,
        legend_title=args.label_col,
        # Highlight options
        highlight_category=args.highlight,
        highlight_color=args.highlight_color,
        highlight_marker=args.highlight_marker,
        highlight_size=args.highlight_size,
        background_color=args.background_color,
        background_marker=args.background_marker,
        background_alpha=args.background_alpha,
        # Axis limits
        xlim=args.xlim,
        ylim=args.ylim
    )
    
    # Save
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {args.output}")
    
    # Save coordinates (unless suppressed)
    if not args.no_save_coordinates:
        coord_file = args.output.rsplit('.', 1)[0] + '_coordinates.csv'
        coord_df = pd.DataFrame({
            'original_index': valid_indices,
            'SMILES': [smiles_list[i] for i in valid_indices],
            'tSNE_1': embedding[:, 0],
            'tSNE_2': embedding[:, 1]
        })
        coord_df.to_csv(coord_file, index=False)
        print(f"Saved coordinates to {coord_file}")


# Run CLI
if __name__ == '__main__':
    main()