# merge_pharmit_output.py — Pharmit Hit-List Merger

Merges the per-vendor SDF archives exported by [Pharmit](https://pharmit.csb.pitt.edu/) into one
deduplicated hit list: a sorted SDF (2D layout, vendor IDs and affinity as properties) and a CSV
lookup table (SMILES + affinity + per-vendor ID columns). Deduplication is by neutralized,
desalted InChIKey, so different salt forms or protonation states of the same compound collapse
to a single entry — the original 3D docking pose is left untouched for SDF output.

---

## Dependencies

| Package | Role | Required |
|---|---|---|
| RDKit | SDF I/O, InChI/InChIKey, salt stripping + neutralization, 2D layout | yes |

```bash
conda install -c conda-forge rdkit
```

---

## Pipeline

```
Input SDF archives (one per vendor, .sdf or .sdf.gz)
  → read each record (explicit Hs kept, removeHs=False)
  → standardize: RemoveHs → strip to largest fragment (desalt) → neutralize charges → sanitize
      failures here (and unparsable records) are counted as skipped
  → canonical SMILES + InChI/InChIKey computed from the standardized structure
  → dedup by InChIKey:
      new key            → register (pose, vendor IDs, affinity, SMILES)
      key already seen   → union vendor ID lists; keep the pose with the
                           better (more negative) minimizedAffinity
  → sort all unique compounds by affinity (best first)
  → write CSV  (SMILES from standardized structure, affinity, per-vendor ID columns)
  → write SDF  (original docking-pose geometry; 2D layout applied only now, post-dedup)
```

### Vendor ID classification

Pharmit's `_Name` field is a cross-reference list of a compound's IDs across *every* database
it's known in, not just the one a given input file was searched against — the same ID list shows
up, just reordered, in every file that hits the compound. So IDs are classified per-token by
matching each token's own format against `VENDOR_PATTERNS`, independent of which input file the
token came from:

| Column | Pattern | Example |
|---|---|---|
| `Enamine_ID` | `Z\d{6,}` | `Z996804398` |
| `ZINC_ID` | `ZINC\d+` or bare digits | `ZINC000069514405`, `69514405` → normalized to `ZINC000069514405` |
| `PubChem_ID` | `PubChem-\d+` | `PubChem-53573646` |
| `CHEMBL_ID` | `CHEMBL\d+` | `CHEMBL281593` |
| `MCULE_ID` | `MCULE-\d+` | `MCULE-4125027661` |
| `MolPort_ID` | `(MolPort\|Molport)-\d{3}-\d{3}-\d{3}` | `MolPort-020-137-979` |
| `CSC_ID` | `CSC\d+` | `CSC025928578` |
| `ChemDiv_ID` | `ChemDiv-[A-Z0-9]{4}-\d{4}[A-Z]?` | `ChemDiv-8012-9008`, `ChemDiv-E228-2808`, `ChemDiv-P935-0820F` |
| `ChemSpace_ID` | `CSSS\d+` | `CSSS00058048812` |
| `LabNetwork_ID` | `LN\d+` | `LN00198330` |
| `NSC_ID` | `NSC\d+` | `NSC1901` |
| `MCULE-Ultimate_ID` | `[A-Z]{14}-[A-Z]{10}-[A-Z]` (InChIKey-shaped) | make-on-demand space with no persistent catalog numbers |
| `other_IDs` | anything unmatched | safety net — new/unrecognized vendor formats land here instead of being silently dropped |

`ZINC_ID` folds bare numeric tokens in too: Pharmit often lists a compound's legacy unprefixed
ZINC ID (`69514405`) right next to its prefixed one (`ZINC000069514405`); both — and any 8-digit
legacy-padded form — are normalized to one canonical `ZINC` + 12-digit form so the same accession
never appears twice.

---

## Usage

```
merge_pharmit_output.py FILE1.sdf[.gz] [FILE2.sdf[.gz] ...] [-o NAME] [--no-sdf] [--no-csv]
```

| Flag | Default | Description |
|---|---|---|
| `inputs` | required | One or more input SDF archives (`.sdf` or `.sdf.gz`) |
| `-o / --output NAME` | `pharmit_merged` | Output stem; writes `NAME.sdf` (plain text) and `NAME.csv`. A trailing `.sdf`, `.csv`, or `.gz` is stripped, so `-o merged.sdf` and `-o merged` behave the same |
| `--no-sdf` | off | Skip the merged SDF output |
| `--no-csv` | off | Skip the CSV lookup table |

---

## Output

### CSV columns

`Compound_ID, SMILES, affinity` + one column per vendor in the table above (`other_IDs` last).
Vendor cells hold comma-separated IDs when a compound has more than one for that vendor.

### SDF properties

| Property | Description |
|---|---|
| `_Name` | Best available vendor ID for display (priority: CHEMBL > Enamine > ZINC > PubChem > MCULE > MolPort > CSC > ChemDiv > ChemSpace > LabNetwork > NSC > MCULE-Ultimate > other_IDs > InChIKey) |
| `Compound_ID` / `Structure_ID` | Same sequential value (`ID_000001`, …), guaranteed unique regardless of vendor ID availability; `Structure_ID` matches `gnina.py --id-column Structure_ID` |
| `minimizedAffinity`, `minimizedRMSD`, … | Carried through unmodified from the winning pose's original SDF fields |
| one property per vendor with at least one ID | e.g. `ZINC_ID`, `PubChem_ID`, comma-separated if multiple |

---

## Examples

### Merge every vendor hit list for one target
```bash
merge_pharmit_output.py 5VDH_7C6_*.sdf.gz -o merged     # → merged.sdf + merged.csv
```

### CSV only, no merged SDF
```bash
merge_pharmit_output.py *.sdf.gz -o merged --no-sdf
```

### Merged SDF only, feed straight into gnina.py
```bash
merge_pharmit_output.py *.sdf.gz -o merged --no-csv
gnina.py sp -r receptor.pdb -a ref.sdf -l merged.sdf -o docked --id-column Structure_ID
```

---

## Notes

- **Dedup key comes from the standardized structure, not the raw pose.** Salts are stripped to
  the largest fragment and charges neutralized before computing the InChIKey, so a carboxylic
  acid and its carboxylate salt (for example) collapse to one entry. The surviving pose's original
  3D geometry, `minimizedAffinity`, and `minimizedRMSD` are untouched — only the dedup key and the
  CSV's SMILES are derived from the standardized structure.

- **Ties are winner-take-all, not pooled.** When two records share an InChIKey, the one with the
  more negative `minimizedAffinity` replaces the other entirely (its pose, affinity, and RMSD all
  "win" together); vendor ID lists are the only thing actually unioned across duplicates.

- **2D layout is applied last**, after dedup, so flattening docking-pose Z-coordinates (~477–480 Å
  absolute pocket coordinates, which some viewers like DataWarrior render as invisible) never
  affects which structures get merged.

- **Skipped counter** covers parse/sanitize failures, standardization failures (bad valence,
  unkekulizable), and missing InChI/InChIKey — printed as one total at the end, not broken out
  per cause.
