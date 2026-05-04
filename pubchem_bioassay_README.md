# pubchem_bioassay.py — Compound Promiscuity via PubChem Bioassay Data

Annotates a compound table with PubChem bioassay statistics to assess promiscuity.
For each compound (identified by InChIKey) the script queries the PubChem REST API
for the assay summary, counts active vs. tested assays and unique biological targets,
and computes a **Target Promiscuity Index (TPI)**.

---

## Dependencies

| Package | Role | Required |
|---|---|---|
| pandas | table I/O and manipulation | yes |
| requests | PubChem REST API calls | yes |
| numpy | NaN handling | yes |
| tqdm | progress bar | optional |

```bash
pip install pandas requests numpy tqdm
```

---

## Pipeline

```
Input CSV (with InChIKey column)
  → connectivity-layer lookup → PubChem CID
  → assay summary CSV (PubChem REST)
  → count total assays, active assays
  → collect unique target IDs (GeneID / accession / TargetID)
  → Target Promiscuity Index = active_targets / tested_targets × 100
  → output CSV (all original columns + 9 new PubChem columns)
```

---

## Usage

```
pubchem_bioassay.py [-h] -o FILE [--inchikey-col COL | --smiles-col COL] [--delay FLOAT] [--sep CHAR] [--batch-size N] [--failed-output FILE] input
```

`--inchikey-col` and `--smiles-col` are mutually exclusive. If neither is given the script
auto-detects: InChIKey column (name contains "inchi") takes priority, then SMILES (name contains "smiles").

### Options

| Flag | Default | Description |
|---|---|---|
| `input` | required | Input CSV file |
| `-o / --output FILE` | required | Output CSV file |
| `--inchikey-col COL` | auto-detect | InChIKey column name; uses connectivity layer for PubChem lookup |
| `--smiles-col COL` | auto-detect | SMILES column name; SMILES are URL-encoded for PubChem lookup |
| `--delay FLOAT` | `0.35` | Seconds between PubChem requests (NCBI rate limit: ~3 req/s) |
| `--sep CHAR` | `,` | Input CSV delimiter |
| `--batch-size N` | `1000` | Rows per batch; output is written after each batch completes |
| `--prior-n N` | `100` | Virtual inactive targets added to TPI denominator (see Notes) |
| `--failed-output FILE` | auto-derived | CSV for rows that errored (network failures, timeouts, aborted connections); only written if errors occur. Default: `<output>_failed.csv` |

---

## Output columns

Nine columns are appended to the right of all original input columns:

| Column | Description |
|---|---|
| `PubChem_CID` | PubChem compound ID (connectivity-layer match) |
| `Total_Assays` | Total number of bioassays on record |
| `Active_Assays` | Number of assays with outcome = Active |
| `Unique_Targets_Tested` | Number of distinct biological targets tested |
| `Unique_Targets_Active` | Number of distinct biological targets with ≥1 active result |
| `Target_Promiscuity_Index` | `active / (tested + prior_n) × 100` (%). Penalises compounds with few targets screened — see Notes. |
| `List_Targets_Tested` | Comma-separated list of all target IDs tested |
| `List_Targets_Active` | Comma-separated list of target IDs with active result |
| `PubChem_Status` | `Success`, `Skeleton Not Found`, `No bioassay data`, `Invalid InChIKey`, or error message |

---

## Examples

### Basic (auto-detect identifier column)
```bash
python pubchem_bioassay.py compounds.csv -o compounds_bioassay.csv
```

### Explicit InChIKey column
```bash
python pubchem_bioassay.py compounds.csv -o out.csv --inchikey-col std_inchikey
```

### Query by SMILES
```bash
python pubchem_bioassay.py compounds.csv -o out.csv --smiles-col SMILES
```

### Tab-separated input, slower request rate
```bash
python pubchem_bioassay.py compounds.tsv -o out.csv --sep $'\t' --delay 0.5
```

### Large library with smaller batches
```bash
python pubchem_bioassay.py library_10k.csv -o out.csv --batch-size 500
```

### Retry failed rows after a network-interrupted run
```bash
# First run — network failures land in out_failed.csv automatically
python pubchem_bioassay.py library.csv -o out.csv

# Re-run only the failed rows
python pubchem_bioassay.py out_failed.csv -o out_retry.csv
```

---

## Notes

- **InChIKey lookup uses the connectivity layer only** (first segment, before the first `-`).
  This means stereoisomers and salts of the same scaffold map to the same CID, which is appropriate
  for bioassay data aggregation.

- **SMILES lookup** passes the full SMILES string to PubChem (URL-encoded). PubChem performs a
  structure search and returns the best-matching CID. Stereochemistry and salt forms in the SMILES
  are respected, so results may differ from InChIKey-based lookup for the same compound.

- **Target identifier column** is auto-selected from the assay summary: `geneid`, `accession`, or
  `targetid` (whichever is present). If none is found, target counts are reported as 0 and TPI as 0.

- **Rate limiting**: PubChem's public API allows ~3 requests/second. The default `--delay 0.35` keeps
  well within this. For large batches consider using the PubChem Power User Gateway (PUG) async API.

- **Target Promiscuity Index** is a regularised hit rate: `active / (tested + prior_n) × 100`,
  where `prior_n` (default 100) acts as virtual inactive targets appended to every compound's record.
  This shrinks sparse observations toward zero — a compound tested against 2 targets cannot outscore
  one tested against 500 at the same raw hit rate. Equivalent to a Beta-Binomial posterior mean with
  a Beta(0, prior_n) prior. Tune `--prior-n` to reflect how many unscreened targets you consider
  plausibly inactive (e.g. `--prior-n 50` for a smaller assumed universe).

- **Batched output**: Results are written to the output file after each batch of `--batch-size` rows
  (default 1000). If the run is interrupted, completed batches are already on disk; re-run from
  scratch or subset the input to resume from a specific row.

- **Failed-row output**: Rows where the PubChem request raised a network error (timeout, connection
  abort, etc.) are written to a separate CSV containing only the original input columns. By default
  this file is named `<output>_failed.csv` (e.g. `out_failed.csv` when `-o out.csv`). Only rows
  with a transient error status are included; compounds not found in PubChem or with no bioassay
  data are considered valid results and appear only in the main output. The failed file can be passed
  directly as input for a retry run.

- **`PubChem_Status = "Skeleton Not Found"`** means no PubChem compound matches the connectivity
  layer of the InChIKey. This is common for proprietary or very recently synthesised compounds.
