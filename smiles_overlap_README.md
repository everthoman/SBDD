# smiles_overlap.py — Structural overlap between two SMILES files

Calculates the exact structural overlap between two SMILES files using RDKit. Each molecule is sanitized and reduced to a canonical form before comparison, so overlap reflects identical chemical structures rather than identical SMILES strings (e.g. `CCO` and `OCC` are recognised as the same molecule).

---

## Dependencies

| Package | Role | Required |
|---|---|---|
| RDKit | Sanitization, canonicalization, InChIKey | yes |

```bash
conda install -c conda-forge rdkit
```

---

## Input format

Whitespace- or comma-delimited SMILES files, one molecule per line. Column order is auto-detected:

```
SMILES [name]
SMILES,name
name,SMILES
```

Lines starting with `#` are treated as comments and skipped. If no identifier is given, molecules are named `mol_<line_index>`. Lines that fail RDKit sanitization are skipped with a warning printed to stderr — they do not count toward either file's totals.

---

## Usage

```
smiles_overlap.py [-h] [-o OUTPUT] [--no-stereo] [--use-inchikey] file1 file2
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `file1`, `file2` | required | The two SMILES files to compare |
| `-o, --output FILE` | none | Write shared structures + identifiers from both files to a TSV |
| `--no-stereo` | off | Ignore stereochemistry (enantiomers count as the same structure) |
| `--use-inchikey` | off | Key on InChIKey instead of canonical SMILES — more tolerant of equivalent representations (e.g. tautomer/protonation differences), at the cost of being a hashed rather than direct match |

By default, comparison is stereo-aware canonical SMILES: molecules must have the same connectivity *and* stereochemistry to count as overlapping.

---

## Output

Console summary:

```
Results (unique structures):
  Unique in file 1      : 3
  Unique in file 2      : 3
  Shared (overlap)      : 2
  Union                 : 4
  Overlap / file 1      : 66.67%
  Overlap / file 2      : 66.67%
  Jaccard index         : 0.5000

Overlapping compounds:
  CC(=O)Oc1ccccc1C(=O)O
    file1: aspirin
    file2: aspirin_dup
  CCO
    file1: ethanol
    file2: ethanol_reordered
```

`Overlap / file 1` and `Overlap / file 2` are the shared count divided by that file's unique-structure count. Jaccard index is shared / union. If a structure appears under multiple identifiers within the same file, all of them are listed comma-separated.

With `-o shared.tsv`, the shared structures are written as:

```
smiles	file1_ids	file2_ids
CC(=O)Oc1ccccc1C(=O)O	aspirin	aspirin_dup
CCO	ethanol	ethanol_reordered
```

---

## Examples

```bash
# Basic comparison
python3 smiles_overlap.py library_a.smi library_b.smi

# Treat enantiomers as identical
python3 smiles_overlap.py library_a.smi library_b.smi --no-stereo

# Tolerate tautomer/protonation differences via InChIKey
python3 smiles_overlap.py library_a.smi library_b.smi --use-inchikey

# Save the overlapping compounds with identifiers from both files
python3 smiles_overlap.py library_a.smi library_b.smi -o shared.tsv
```

---

## Notes

- Duplicate structures within a single file are collapsed before comparison — overlap is computed on unique structures, not raw row counts.
- `--use-inchikey` is a hashed comparison: two different structures could theoretically collide on the same InChIKey (extremely rare in practice), and it does not let you recover the exact canonical SMILES if the original wasn't kept — the identifiers column still reports whatever names were in the input files.
