# hitpicker.py

ML-assisted hit prioritisation for high-throughput screening (HTS) data.

Ranks confirmed hits by likelihood of being true actives (not assay interferents), and surfaces the most likely false negatives from the non-hit pool. Implements the MVS-A method of Boldini et al. 2024 (*ACS Cent. Sci.* 10, 823–832) combined with structural alert flagging (PAINS, REOS).

---

## Dependencies

```
conda activate hitpicker   # or sbdd
# Required: rdkit, lightgbm, scikit-learn, pandas, numpy
# Optional: sqlalchemy (fingerprint cache), matplotlib + reportlab (PDF report)
```

---

## Quick start

```bash
hitpicker.py -i screen.csv --smiles_col smiles --act_col AUC_inhib
```

Writes two files:
- `screen_hitpicker_ranked.csv` — all compounds ranked
- `screen_hitpicker_failed.csv` — any blank / unparseable SMILES (if any)

---

## Workflow

```
CSV input
  │
  ├─ 1. Load & validate columns
  ├─ 2. Compute Morgan fingerprints (ECFP4, r=2, 2048-bit)  [optional SQLite cache]
  ├─ 3. Robust z-score normalise each activity column
  ├─ 4. Label hits per column; combine into composite hit label
  ├─ 5. Flag PAINS / additional RDKit catalogs + REOS rules
  ├─ 6. Train LightGBM (class_weight=balanced, 500 trees)
  │     ├─ Stratified k-fold CV → out-of-fold hit probabilities
  │     └─ Mondrian cross-conformal prediction → p-values, confidence, credibility
  ├─ 7. MVS-A sample influence scores  (Sharchilev 2018 / Boldini 2024)
  ├─ 8. Composite false-positive flags for confirmed hits
  └─ 9. Sort & write ranked CSV
```

---

## Ranking logic

### Confirmed hits (`is_hit == 1`)

Sorted by **false-positive risk tier** then **MVS-A score**:

| Tier | `fp_flags` | Interpretation |
|------|------------|----------------|
| 0 | *(empty)* | No red flags — highest confidence true actives |
| 1 | one of `high_mvsa`, `struct_alert`, `reos` | One warning signal |
| 2+ | multiple flags | Likely assay interferent; deprioritise |

Within each tier, hits are sorted by `mvsa_score` ascending (lower = better).

### Non-hits (`is_hit == 0`)

Sorted by `mvsa_score` **descending** — compounds at the top of this section have the highest sample influence score despite being labelled inactive, indicating the model consistently associated them with the active class. These are the most likely **false negatives** and worth re-examining (Boldini 2024 case study: CHT4 recovered as rank-1 inactive).

---

## MVS-A explained

**Minimal Variance Sampling Analysis** measures how "surprising" each compound is to the trained GBM boundary by accumulating the standardised squared gradient across boosting steps:

$$\text{MVS-A}(i) = \frac{1}{T} \sum_{t} n_{\text{block}} \cdot \frac{(p_t(i) - y_i)^2}{p_t(i)(1-p_t(i))}$$

- **True positives** become confident early — gradient decays to zero → **low MVS-A**
- **False positives** (assay artefacts) remain uncertain throughout → **high MVS-A**
- Computed at 20 evenly-spaced checkpoints over the 500-tree ensemble

For confirmed hits, `mvsa_score` > median(hits) → `high_mvsa` flag. Combining MVS-A with structural alerts improved true positive rate from 25 % → 38 % in Boldini's CHT inhibitor case study.

---

## Output columns

| Column | Description |
|--------|-------------|
| `rank` | 1 = best |
| `zscore` | Robust z-score of activity (or `zscore_{col}` per column if multiple) |
| `is_hit` | 1 if compound meets the hit threshold |
| `is_hit_{col}` | Per-column hit flags (multi-column runs only) |
| `cv_hit_prob` | Out-of-fold ML hit probability (CV) |
| `hit_prob` | Final model hit probability (trained on full dataset) |
| `cp_p_hit` | Conformal p-value under hit hypothesis |
| `cp_p_nonhit` | Conformal p-value under non-hit hypothesis |
| `cp_confidence` | 1 − min(p_hit, p_nonhit); decisiveness of the prediction |
| `cp_credibility` | max(p_hit, p_nonhit); plausibility of the assigned label |
| `cp_prediction` | `hit` / `nonhit` / `uncertain` / `outlier` at significance α |
| `mvsa_score` | MVS-A sample influence (low = likely TP for hits; high = likely FN for non-hits) |
| `fp_flags` | Semicolon-joined false-positive flags for confirmed hits (see below) |
| `struct_alerts` | Matching PAINS / catalog pattern names (empty if clean) |
| `reos_violations` | Violated REOS rule descriptions (empty if clean) |

### `fp_flags` values

| Flag | Trigger |
|------|---------|
| `high_mvsa` | `mvsa_score` > median of confirmed hits |
| `struct_alert` | Any match in the loaded RDKit catalog(s) |
| `reos` | Any REOS rule violated |

---

## CLI reference

### Required

| Argument | Description |
|----------|-------------|
| `-i / --input CSV` | Input CSV file |
| `--smiles_col` | Column containing SMILES (default: `smiles`) |
| `--act_col COL [COL …]` | Activity column(s) to use for hit labelling |

### Activity options

| Argument | Default | Description |
|----------|---------|-------------|
| `--act_method` | `robust_zscore` | `robust_zscore` or `threshold` |
| `--act_threshold N [N …]` | 3.0 / 30.0 | Hit threshold per column (broadcast if single value given) |
| `--act_dir {high,low} […]` | `high` | `high`: hits have high values; `low`: hits have low values |
| `--min_active N` | all columns | Composite hit requires ≥ N columns to be active |

### Model options

| Argument | Default | Description |
|----------|---------|-------------|
| `--no_cv` | off | Skip CV and conformal prediction |
| `--no_mvsa` | off | Skip MVS-A scoring (faster) |
| `--n_folds` | 5 | Number of CV folds |
| `--radius` | 2 | Morgan FP radius |
| `--nbits` | 2048 | Morgan FP bit count |
| `--seed` | 42 | Random seed |
| `--cp_alpha` | 0.1 | Conformal significance level α (90 % coverage guarantee) |
| `--cache DB` | off | SQLite file for fingerprint caching (useful when re-running with different thresholds) |

### Filter options

| Argument | Default | Description |
|----------|---------|-------------|
| `--catalogs NAME [NAME …]` | `PAINS` | RDKit FilterCatalog(s) to load. Pass `none` to disable. Valid: `PAINS`, `PAINS_A/B/C`, `BRENK`, `NIH`, `ZINC`, `CHEMBL*`, `ALL` |
| `--no_reos` | off | Skip REOS rule flagging |
| `--reos_file TXT` | `REOS.txt` | Path to REOS rules file |
| `--exclude_pains_from_training` | off | Flip PAINS-flagged hits to non-hits before model training so the model learns SAR from structurally clean hits only |

### Output options

| Argument | Default | Description |
|----------|---------|-------------|
| `-o / --output PREFIX` | input stem | Output file prefix |
| `--id_col COL` | none | Identifier column (used in log messages) |
| `--report` | off | Write a PDF summary report (requires matplotlib + reportlab) |

---

## Usage examples

### Single activity column

```bash
# Robust z-score, hits = high values, default threshold z ≥ 3
hitpicker.py -i screen.csv --smiles_col smiles --act_col AUC_inhib

# Stricter threshold
hitpicker.py -i screen.csv --smiles_col smiles --act_col AUC_inhib \
    --act_threshold 4.0

# Hard % threshold
hitpicker.py -i screen.csv --smiles_col smiles --act_col pct_inhib \
    --act_method threshold --act_threshold 50

# Counter-screen: hits have LOW viability
hitpicker.py -i screen.csv --smiles_col smiles --act_col viability \
    --act_method threshold --act_threshold 70 --act_dir low
```

### Multiple activity columns

```bash
# Both replicates must be hits
hitpicker.py -i screen.csv --smiles_col smiles \
    --act_col rep1 rep2 --act_threshold 3.0 3.0

# Majority vote: ≥ 2 of 3 replicates
hitpicker.py -i screen.csv --smiles_col smiles \
    --act_col rep1 rep2 rep3 --min_active 2

# Primary screen + counter-screen (different directions)
hitpicker.py -i screen.csv --smiles_col smiles \
    --act_col AUC_inhib cytotox \
    --act_dir high low --act_threshold 3.0 2.0
```

### Structural filters

```bash
# Add BRENK (medicinal chemistry alerts) alongside PAINS
hitpicker.py -i screen.csv --smiles_col smiles --act_col inhib \
    --catalogs PAINS BRENK

# Exclude PAINS-flagged hits from model training
hitpicker.py -i screen.csv --smiles_col smiles --act_col inhib \
    --exclude_pains_from_training

# No structural filters
hitpicker.py -i screen.csv --smiles_col smiles --act_col inhib \
    --catalogs none --no_reos
```

### Speed and caching

```bash
# Cache fingerprints for re-runs (useful when tuning thresholds on a large library)
hitpicker.py -i screen.csv --smiles_col smiles --act_col inhib \
    --cache screen_fps.db

# Fast mode: skip CV, conformal prediction, and MVS-A
hitpicker.py -i screen.csv --smiles_col smiles --act_col inhib \
    --no_cv --no_mvsa

# Full run with PDF report
hitpicker.py -i screen.csv --smiles_col smiles --act_col inhib \
    --id_col compound_id -o results/run1 --report
```

---

## Conformal prediction

At significance level α (default 0.1), each compound receives a prediction set label:

| `cp_prediction` | Meaning |
|----------------|---------|
| `hit` | Only hit hypothesis supported (p_hit > α, p_nonhit ≤ α) |
| `nonhit` | Only non-hit hypothesis supported |
| `uncertain` | Both hypotheses plausible — borderline compound |
| `outlier` | Neither hypothesis supported — structurally novel or noisy measurement |

The coverage guarantee: the true label is in the prediction set for ≥ 1−α of compounds. `cp_confidence` (1 − min p-value) measures how decisive the prediction is; `cp_credibility` (max p-value) measures how plausible the assigned label is.

---

## References

- Boldini D, Friedrich L, Kuhn D, Sieber SA. *Machine-Learning-Assisted Hit Prioritization for High-Throughput Screening in Drug Discovery.* ACS Cent. Sci. 2024, 10, 823–832. https://doi.org/10.1021/acscentsci.3c01517
- Sharchilev B, Ustinovsky Y, Serdyukov P, de Rijke M. *Finding Influential Training Samples for Gradient Boosted Decision Trees.* arXiv 2018. http://arxiv.org/abs/1802.06640
- Baell JB, Holloway GA. *New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS).* J. Med. Chem. 2010, 53, 2719–2740.
