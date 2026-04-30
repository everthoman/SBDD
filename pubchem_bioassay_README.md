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
pubchem_bioassay.py [-h] -o FILE [--inchikey-col COL] [--delay FLOAT] [--sep CHAR] input
```

### Options

| Flag | Default | Description |
|---|---|---|
| `input` | required | Input CSV file containing an InChIKey column |
| `-o / --output FILE` | required | Output CSV file |
| `--inchikey-col COL` | auto-detect | InChIKey column name; auto-detected if any column name contains "inchi" |
| `--delay FLOAT` | `0.35` | Seconds between PubChem requests (NCBI rate limit: ~3 req/s) |
| `--sep CHAR` | `,` | Input CSV delimiter |

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
| `Target_Promiscuity_Index` | `Unique_Targets_Active / Unique_Targets_Tested × 100` (%) |
| `List_Targets_Tested` | Comma-separated list of all target IDs tested |
| `List_Targets_Active` | Comma-separated list of target IDs with active result |
| `PubChem_Status` | `Success`, `Skeleton Not Found`, `No bioassay data`, `Invalid InChIKey`, or error message |

---

## Examples

### Basic
```bash
python pubchem_bioassay.py compounds.csv -o compounds_bioassay.csv
```

### Explicit InChIKey column name
```bash
python pubchem_bioassay.py compounds.csv -o out.csv --inchikey-col std_inchikey
```

### Tab-separated input, slower request rate
```bash
python pubchem_bioassay.py compounds.tsv -o out.csv --sep $'\t' --delay 0.5
```

---

## Notes

- **CID lookup uses the connectivity layer only** (first segment of the InChIKey, before the first `-`).
  This means stereoisomers and salts of the same scaffold map to the same CID, which is appropriate
  for bioassay data aggregation.

- **Target identifier column** is auto-selected from the assay summary: `geneid`, `accession`, or
  `targetid` (whichever is present). If none is found, target counts are reported as 0 and TPI as 0.

- **Rate limiting**: PubChem's public API allows ~3 requests/second. The default `--delay 0.35` keeps
  well within this. For large batches consider using the PubChem Power User Gateway (PUG) async API.

- **`PubChem_Status = "Skeleton Not Found"`** means no PubChem compound matches the connectivity
  layer of the InChIKey. This is common for proprietary or very recently synthesised compounds.
