# fetch_smiles.py — Fetch SMILES from PubChem

Retrieves canonical SMILES and related properties from PubChem for a list of compound names in a CSV or TSV file. Auto-detects the delimiter and name column, falls back through multiple name-cleaning strategies, and optionally validates the retrieved SMILES with RDKit.

---

## Dependencies

| Package | Role | Required |
|---|---|---|
| requests | PubChem PUG REST API calls | yes |
| RDKit | SMILES validity check in `--check` mode | no (falls back to PubChem API) |

```bash
conda install -c conda-forge rdkit
pip install requests
```

---

## Lookup strategy

For each compound name, the script tries three strategies in order until one succeeds:

1. **Exact name** — queries PubChem directly by the name as given
2. **Cleaned name** — strips common salt/form suffixes (` hydrochloride`, ` sodium`, ` sulfate`, ` recombinant`, etc.) and retries
3. **Synonym/word search** — uses PubChem's `name_type=word` endpoint, which matches the query against all registered synonyms

Compounds that remain unfound after all three strategies (biologics, oligonucleotides, polymers, undefined mixtures) are marked `NOT_FOUND`.

---

## Usage

```
fetch_smiles.py [-h] [--name-col COL] [--output FILE] [--delay FLOAT]
                [--check] [--check-only] [--refetch-smiles]
                input
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `input` | required | Input CSV or TSV file |
| `--name-col COL` | auto-detected | Column containing compound names |
| `--output FILE` | `<input>_smiles.csv` | Output file |
| `--delay FLOAT` | `0.2` | Seconds between PubChem requests (≤5 req/s limit) |
| `--check` | off | Validate SMILES and check MW consistency after fetching |
| `--check-only` | off | Skip fetching; run validation on existing output file |
| `--refetch-smiles` | off | Re-fetch SMILES by CID for OK rows with empty SMILES (see note below) |

### Auto-detection

The delimiter is sniffed from the first 4 KB of the file (supports `,`, `\t`, `|`, `;`). The name column is found by scanning headers for keywords: `compound`, `name`, `drug`, `molecule`, `chemical`, `substance`, `ligand` — first match wins. Use `--name-col` to override.

---

## Output columns

The output file contains all original columns plus:

| Column | Description |
|---|---|
| `CID` | PubChem Compound ID |
| `CanonicalSMILES` | Connectivity SMILES (no stereochemistry) |
| `IsomericSMILES` | Isomeric SMILES (with stereo) |
| `IUPACName` | IUPAC name from PubChem |
| `MolecularFormula` | Molecular formula |
| `MolecularWeight` | Molecular weight reported by PubChem |
| `PubChemStatus` | `OK`, `NOT_FOUND`, or `ERROR` |

With `--check`, three additional columns are written:

| Column | Description |
|---|---|
| `SmilesValid` | `True`/`False` — whether RDKit (or PubChem) can parse the SMILES |
| `MWCheck` | `OK`, `MISMATCH`, or `UNKNOWN` |
| `MWDelta_pct` | % difference between MW computed from molecular formula and PubChem-reported MW |
| `CheckNotes` | `OK`, `invalid_smiles`, or `MW: computed=X reported=Y` for mismatches |

MW is flagged if the discrepancy exceeds 1%. Known false positives: radiopharmaceuticals and isotopically-labelled drugs (e.g. ¹⁸F, ²²³Ra, ²H) where PubChem reports isotopic mass but the molecular formula uses natural atomic weights.

---

## Examples

```bash
# Basic fetch — auto-detect everything
python3 fetch_smiles.py compounds.csv

# Specify name column explicitly
python3 fetch_smiles.py compounds.csv --name-col "Drug Name"

# Fetch + validate in one step (requires RDKit)
conda run -n sbdd python3 fetch_smiles.py compounds.csv --check

# Validate an already-fetched output file
conda run -n sbdd python3 fetch_smiles.py compounds.csv --check-only

# Custom output path and slower request rate
python3 fetch_smiles.py compounds.csv --output results.csv --delay 0.5
```

### Resume support

If the output file already exists, the script skips rows already present and appends new results. Safe to interrupt and re-run on large files.

---

## Notes

- **PubChem API property names**: as of 2025, PubChem returns `ConnectivitySMILES` and `SMILES` rather than the older `CanonicalSMILES` / `IsomericSMILES` property names. If you have an existing output file with empty SMILES columns (fetched with an older version), run `--refetch-smiles` to backfill by CID without re-querying names.

- **Biologics and macromolecules** (monoclonal antibodies, recombinant proteins, oligonucleotides, polysaccharides) will remain `NOT_FOUND` — PubChem does not store discrete small-molecule structures for these.

- **Rate limiting**: the default 0.2 s delay keeps requests at ~5/s, within PubChem's recommended limit. Increase `--delay` if you see `429` errors.
