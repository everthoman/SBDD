#!/usr/bin/env python3
"""hitpicker.py — ML-assisted hit picker for HTS screening data.

Workflow:
  1. Read CSV with SMILES + one or more activity columns
  2. Normalize each activity column (robust z-score or absolute threshold)
  3. Label hits per column; combine into composite hit label
  4. Compute Morgan (ECFP4) fingerprints with optional SQLite cache
  5. Flag PAINS / REOS; exclude PAINS-flagged hits from training labels
     so the model learns real SAR, not assay-interference scaffolds
  6. Train LightGBM with optional stratified cross-validation +
     mondrian cross-conformal prediction (p-values, confidence, credibility)
  7. Compute MVS-A (Minimal Variance Sampling Analysis) sample influence
     scores (Sharchilev 2018 / Boldini 2024): low score = true positive,
     high score = likely assay interferent / false positive
  8. Sort output: confirmed hits by MVS-A score (asc), non-hits by CV prob
  9. Write ranked CSV; optionally write a PDF report

Usage examples:
  # Single column — robust z-score, hits have high values (z ≥ 3)
  hitpicker.py -i screen.csv --smiles_col smiles --act_col inhib

  # Multiple columns — compound must be a hit in both
  hitpicker.py -i screen.csv --smiles_col smiles \\
      --act_col inhib_primary inhib_counter \\
      --act_dir high high --act_threshold 3.0 3.0

  # Multiple columns — hit in at least 2 of 3
  hitpicker.py -i screen.csv --smiles_col smiles \\
      --act_col rep1 rep2 rep3 --act_dir high --min_active 2

  # Hard threshold, skip CV
  hitpicker.py -i screen.csv --smiles_col smiles --act_col inhib \\
      --act_method threshold --act_threshold 30 -o screen_at30 --no_cv

  # Low-value hits (e.g. cytotox counter-screen)
  hitpicker.py -i screen.csv --smiles_col smiles --act_col cytotox \\
      --act_method threshold --act_threshold 20 --act_dir low
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import sys
import warnings
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_REOS = _SCRIPT_DIR / "REOS.txt"

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── fingerprints ──────────────────────────────────────────────────────────────

def _mol_from_smiles(smi: str):
    from rdkit import Chem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Chem.MolFromSmiles(str(smi))


def _morgan_fp(mol, radius: int, nbits: int) -> np.ndarray:
    from rdkit.Chem import rdFingerprintGenerator, DataStructs
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    bv = gen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _cache_key(smiles_list, radius, nbits) -> str:
    h = hashlib.md5("|".join(smiles_list).encode() + f"{radius}{nbits}".encode())
    return h.hexdigest()


def compute_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    nbits: int = 2048,
    cache_db: str | None = None,
) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Return (fp_matrix, valid_idx, blank_idx, invalid_idx).

    blank_idx:   rows where SMILES field was empty / NaN
    invalid_idx: rows where SMILES was present but RDKit could not parse it
    """
    if cache_db:
        key = _cache_key(smiles_list, radius, nbits)
        cached = _cache_load(cache_db, key)
        if cached is not None:
            log.info("  Loaded fingerprints from cache.")
            if len(cached) == 2:
                return cached[0], cached[1], [], []
            if len(cached) == 3:
                return cached[0], cached[1], [], cached[2]
            return cached

    fps, valid_idx, blank_idx, invalid_idx = [], [], [], []
    for i, smi in enumerate(smiles_list):
        s = str(smi).strip()
        if not s or s.lower() in ("nan", "none", ""):
            blank_idx.append(i)
            continue
        mol = _mol_from_smiles(s)
        if mol is None:
            invalid_idx.append(i)
        else:
            fps.append(_morgan_fp(mol, radius, nbits))
            valid_idx.append(i)

    result = (np.array(fps, dtype=np.uint8), valid_idx, blank_idx, invalid_idx)

    if cache_db:
        _cache_save(cache_db, key, result)

    return result


def _cache_load(db_path: str, key: str):
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS fp_cache (key TEXT PRIMARY KEY, data BLOB)"
            ))
            row = conn.execute(
                text("SELECT data FROM fp_cache WHERE key=:k"), {"k": key}
            ).fetchone()
        if row:
            return pickle.loads(row[0])
    except Exception:
        pass
    return None


def _cache_save(db_path: str, key: str, data) -> None:
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(f"sqlite:///{db_path}")
        blob = pickle.dumps(data)
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS fp_cache (key TEXT PRIMARY KEY, data BLOB)"
            ))
            conn.execute(
                text("INSERT OR REPLACE INTO fp_cache VALUES (:k, :d)"),
                {"k": key, "d": blob},
            )
    except Exception as e:
        log.debug(f"Cache write failed: {e}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _broadcast(val, n: int, flag: str) -> list:
    if val is None:
        return [None] * n
    lst = val if isinstance(val, list) else [val]
    if len(lst) == 1:
        return lst * n
    if len(lst) != n:
        sys.exit(f"{flag}: expected 1 or {n} values, got {len(lst)}")
    return lst


# ── activity normalization ────────────────────────────────────────────────────

def robust_zscore(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        mad = np.std(values) or 1.0
    return (values - median) / (1.4826 * mad)


def label_hits(
    activity: np.ndarray,
    method: str,
    threshold: float,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary_labels, normalized_scores).

    direction='high': hits have high values (z ≥ threshold, or raw ≥ threshold)
    direction='low':  hits have low values  (z ≤ -threshold, or raw ≤ threshold)
    """
    if method == "robust_zscore":
        scores = robust_zscore(activity)
        labels = (scores >= threshold if direction == "high" else scores <= -threshold)
    elif method == "threshold":
        scores = activity.copy().astype(float)
        labels = (scores >= threshold if direction == "high" else scores <= threshold)
    else:
        sys.exit(f"Unknown --act_method: {method}")
    return labels.astype(int), scores


# ── model ─────────────────────────────────────────────────────────────────────

def _lgbm(seed: int = 42):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        verbose=-1,
    )


def _cp_pvalues(proba_val: np.ndarray, y_val: np.ndarray):
    """Mondrian cross-conformal p-values, LOO within fold.

    Nonconformity for hit hypothesis:    nc = 1 - p(hit)
    Nonconformity for non-hit hypothesis: nc = p(hit)  [= 1 - p(nonhit)]

    Returns (p_hit, p_nonhit) arrays parallel to proba_val.
    """
    n = len(proba_val)
    nc_hit    = 1.0 - proba_val
    nc_nonhit = proba_val

    hit_mask    = y_val == 1
    nonhit_mask = ~hit_mask

    def _count_exceed_chunked(nc_cal, nc_test, chunk=1000):
        counts = np.zeros(len(nc_test), dtype=np.int64)
        for s in range(0, len(nc_cal), chunk):
            counts += (nc_cal[s:s + chunk, None] >= nc_test[None, :]).sum(axis=0)
        return counts

    # p-value under hit hypothesis
    n_hit_cal = hit_mask.sum()
    if n_hit_cal > 0:
        count = _count_exceed_chunked(nc_hit[hit_mask], nc_hit)
        is_hit = hit_mask.astype(np.int64)
        p_hit = np.clip((count - is_hit + 1) / (n_hit_cal - is_hit + 1), 0.0, 1.0)
    else:
        p_hit = np.ones(n)

    # p-value under non-hit hypothesis
    n_nonhit_cal = nonhit_mask.sum()
    if n_nonhit_cal > 0:
        count = _count_exceed_chunked(nc_nonhit[nonhit_mask], nc_nonhit)
        is_nonhit = nonhit_mask.astype(np.int64)
        p_nonhit = np.clip(
            (count - is_nonhit + 1) / (n_nonhit_cal - is_nonhit + 1), 0.0, 1.0
        )
    else:
        p_nonhit = np.ones(n)

    return p_hit, p_nonhit


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run stratified CV with conformal p-values.

    Returns (cv_proba, cp_p_hit, cp_p_nonhit).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_proba  = np.zeros(len(y))
    cp_p_hit  = np.zeros(len(y))
    cp_p_nonhit = np.zeros(len(y))
    aucs, aps = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        m = _lgbm(seed + fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]

        cv_proba[va] = p

        ph, pnh = _cp_pvalues(p, y[va])
        cp_p_hit[va]    = ph
        cp_p_nonhit[va] = pnh

        if y[va].sum() > 0:
            aucs.append(roc_auc_score(y[va], p))
            aps.append(average_precision_score(y[va], p))

    if aucs:
        log.info(
            f"  CV ROC-AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
            f"AP {np.mean(aps):.3f} ± {np.std(aps):.3f}"
        )
    return cv_proba, cp_p_hit, cp_p_nonhit


# ── MVS-A sample influence ───────────────────────────────────────────────────

def compute_mvsa(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_checkpoints: int = 20,
) -> np.ndarray:
    """MVS-A (Minimal Variance Sampling Analysis) sample influence scores.

    Accumulates standardised squared gradient ∑ n_block*(g²/h) across
    n_checkpoints evenly-spaced boosting steps of the final GBM model.
    This approximates the object-importance formula of Sharchilev et al. 2018
    as adapted by Boldini et al. 2024 for HTS false-positive detection.

    Interpretation
    --------------
    Labeled hits:     low score  → fits learned SAR → likely true positive
                      high score → deviates from boundary → likely false positive
    Labeled non-hits: high score → model never learned this region → possible
                                   false negative worth re-examining
    """
    booster = model.booster_
    n_trees = booster.num_trees()
    y_f = y.astype(np.float64)

    n_ck = min(n_checkpoints, n_trees)
    pts = np.unique(np.round(np.linspace(1, n_trees, n_ck)).astype(int))

    oi = np.zeros(len(X), dtype=np.float64)
    prev_t = 0

    for t in pts:
        n_block = int(t) - prev_t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = booster.predict(X, num_iteration=int(t), raw_score=True)
        p = 1.0 / (1.0 + np.exp(-raw))       # sigmoid → probability
        g = p - y_f                            # gradient: p - y
        h = np.maximum(p * (1.0 - p), 1e-8)  # hessian (Bernoulli)
        oi += n_block * (g * g / h)           # weighted sum of standardised grad²
        prev_t = int(t)

    return oi / n_trees   # normalise to be independent of tree count


# ── PAINS / REOS flagging ────────────────────────────────────────────────────

_VALID_CATALOGS = {
    "PAINS", "PAINS_A", "PAINS_B", "PAINS_C",
    "BRENK", "NIH", "ZINC",
    "CHEMBL", "CHEMBL_BMS", "CHEMBL_Dundee", "CHEMBL_Glaxo",
    "CHEMBL_Inpharmatica", "CHEMBL_LINT", "CHEMBL_MLSMR", "CHEMBL_SureChEMBL",
    "ALL",
}
_PAINS_NAMES = {"PAINS", "PAINS_A", "PAINS_B", "PAINS_C", "ALL"}


def _make_filter_catalog(names: list[str]):
    """Build an RDKit FilterCatalog from a list of catalog names."""
    names = [n for n in names if n.lower() != "none"]
    if not names:
        return None
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    for name in names:
        if name not in _VALID_CATALOGS:
            sys.exit(f"Unknown catalog '{name}'. Valid options: {sorted(_VALID_CATALOGS)}")
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog(params)


def _load_reos(path: Path):
    from rdkit import Chem
    rules = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            smarts = parts[0].strip()
            desc = parts[2].strip().strip('"')
            try:
                max_count = int(parts[1].strip())
            except ValueError:
                continue
            q = Chem.MolFromSmarts(smarts)
            if q is not None:
                rules.append((max_count, desc, q))
    return rules


def annotate_filters(
    smiles_list: list[str],
    pains_catalog,
    reos_rules,
) -> tuple[list[str], list[str]]:
    """Return (pains_flags, reos_flags) parallel to smiles_list.

    Each entry is a semicolon-separated string of matching names/descriptions,
    or "" if clean.
    """
    from rdkit import Chem
    pains_out, reos_out = [], []
    for smi in smiles_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            pains_out.append("")
            reos_out.append("")
            continue

        if pains_catalog is not None:
            matches = pains_catalog.GetMatches(mol)
            pains_out.append(";".join(m.GetDescription() for m in matches))
        else:
            pains_out.append("")

        hits_r = [
            desc for max_count, desc, q in reos_rules
            if len(mol.GetSubstructMatches(q)) > max_count
        ]
        reos_out.append(";".join(hits_r))

    return pains_out, reos_out


# ── report ────────────────────────────────────────────────────────────────────

def write_report(
    df_out: pd.DataFrame,
    activity_cols: list[str],
    score_cols: list[str],
    out_prefix: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, Table
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        import tempfile, os

        tmp = tempfile.mkdtemp()
        figs = []

        # Activity + score distributions
        n = len(activity_cols)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n), squeeze=False)
        for i, (ac, sc) in enumerate(zip(activity_cols, score_cols)):
            axes[i, 0].hist(df_out[ac].dropna(), bins=50, color="steelblue", edgecolor="white")
            axes[i, 0].set_xlabel(ac); axes[i, 0].set_ylabel("Count")
            axes[i, 0].set_title(f"Activity: {ac}")
            axes[i, 1].hist(df_out[sc].dropna(), bins=50, color="darkorange", edgecolor="white")
            axes[i, 1].set_xlabel(sc); axes[i, 1].set_ylabel("Count")
            axes[i, 1].set_title(f"Normalized: {sc}")
        fig.tight_layout()
        p = os.path.join(tmp, "hist_activity.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figs.append(p)

        # Score vs rank
        fig, ax = plt.subplots(figsize=(6, 3))
        rank_col = "cv_hit_prob" if "cv_hit_prob" in df_out.columns else "hit_prob"
        ax.plot(df_out["rank"].values, df_out[rank_col].values, lw=0.8, color="firebrick")
        ax.set_xlabel("Rank"); ax.set_ylabel("Predicted hit probability")
        ax.set_title("Score vs rank")
        p = os.path.join(tmp, "score_vs_rank.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figs.append(p)

        # Conformal confidence histogram (if available)
        if "cp_confidence" in df_out.columns:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(df_out["cp_confidence"].dropna(), bins=50, color="seagreen", edgecolor="white")
            ax.set_xlabel("Conformal confidence"); ax.set_ylabel("Count")
            ax.set_title("Prediction confidence distribution")
            p = os.path.join(tmp, "cp_confidence.png")
            fig.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
            figs.append(p)

        pdf_path = out_prefix + "_hitpicker_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("hitpicker.py — Hit Picking Report", styles["Title"]),
                 Spacer(1, 0.4 * cm)]

        n_hits = int(df_out["is_hit"].sum())
        hit_rate = n_hits / len(df_out) * 100
        top100_rate = df_out.head(100)["is_hit"].mean() * 100
        summary = [
            ["Total compounds", str(len(df_out))],
            ["Confirmed hits", f"{n_hits} ({hit_rate:.1f}%)"],
            ["Top-100 hit rate", f"{top100_rate:.1f}%"],
        ]
        story.append(Table(summary))
        story.append(Spacer(1, 0.4 * cm))

        for fig_path in figs:
            story.append(Image(fig_path, width=14 * cm, height=7 * cm))
            story.append(Spacer(1, 0.3 * cm))

        doc.build(story)
        log.info(f"Wrote report → {pdf_path}")

    except Exception as e:
        log.warning(f"Report generation failed ({e}); skipping.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ML-assisted hit picker for HTS screening data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, metavar="CSV")
    p.add_argument("--smiles_col", default="smiles")
    p.add_argument("--act_col", required=True, nargs="+",
                   help="Activity column(s)")
    p.add_argument("--id_col", default=None)
    p.add_argument("-o", "--output", default=None, metavar="PREFIX",
                   help="Output prefix (default: input stem)")

    act = p.add_argument_group("activity")
    act.add_argument("--act_method", choices=["robust_zscore", "threshold"],
                     default="robust_zscore")
    act.add_argument("--act_threshold", type=float, nargs="+", default=None,
                     help="Threshold per column or single broadcast value "
                          "(default: 3.0 for robust_zscore, 30.0 for threshold)")
    act.add_argument("--act_dir", nargs="+", choices=["high", "low"], default=["high"],
                     help="'high': hits have high values; 'low': hits have low values")
    act.add_argument("--min_active", type=int, default=None, metavar="N",
                     help="Composite hit requires at least N columns active (default: all)")

    ml = p.add_argument_group("model")
    ml.add_argument("--no_cv", action="store_true", help="Skip CV and conformal prediction")
    ml.add_argument("--no_mvsa", action="store_true",
                    help="Skip MVS-A sample influence scoring (faster for quick runs)")
    ml.add_argument("--n_folds", type=int, default=5)
    ml.add_argument("--radius", type=int, default=2, help="Morgan FP radius")
    ml.add_argument("--nbits", type=int, default=2048)
    ml.add_argument("--seed", type=int, default=42)
    ml.add_argument("--cache", default=None, metavar="DB",
                    help="SQLite fingerprint cache for repeat runs")
    ml.add_argument("--cp_alpha", type=float, default=0.1, metavar="ALPHA",
                    help="Conformal significance level α (default 0.1 = 90%% confidence). "
                         "Label y is included in the prediction set when p_y > α. "
                         "Coverage guarantee: true label is in the set for ≥ 1-α of compounds.")

    filt = p.add_argument_group("filters")
    filt.add_argument("--catalogs", nargs="+", default=["PAINS"], metavar="NAME",
                      help="RDKit FilterCatalog(s) to flag. Pass 'none' to skip. "
                           f"Valid: {', '.join(sorted(_VALID_CATALOGS))}")
    filt.add_argument("--no_reos", action="store_true")
    filt.add_argument("--reos_file", default=str(_DEFAULT_REOS), metavar="TXT")
    filt.add_argument("--exclude_pains_from_training", action="store_true",
                      help="Exclude PAINS-flagged hits from training labels so the model "
                           "learns SAR from structurally clean hits only")

    p.add_argument("--report", action="store_true", help="Write PDF summary report")
    return p.parse_args()


def main():
    args = parse_args()

    n_cols = len(args.act_col)
    default_thr = 3.0 if args.act_method == "robust_zscore" else 30.0
    thresholds = _broadcast(args.act_threshold or [default_thr], n_cols, "--act_threshold")
    directions = _broadcast(args.act_dir, n_cols, "--act_dir")
    min_active = args.min_active if args.min_active is not None else n_cols
    if not (1 <= min_active <= n_cols):
        sys.exit(f"--min_active must be between 1 and {n_cols}")

    _raw = args.output or Path(args.input).stem
    out_prefix = str(Path(_raw).with_suffix("")) if Path(_raw).suffix in {
        ".csv", ".tsv", ".txt", ".sdf", ".smi"} else _raw

    # ── load ──────────────────────────────────────────────────────────────────
    log.info(f"Loading {args.input}")
    df = pd.read_csv(args.input)
    log.info(f"  {len(df):,} rows")

    for col in [args.smiles_col] + args.act_col:
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found. Available: {list(df.columns)}")
    if args.id_col and args.id_col not in df.columns:
        log.warning(f"ID column '{args.id_col}' not found — using row index.")
        args.id_col = None

    # ── fingerprints ──────────────────────────────────────────────────────────
    log.info("Computing Morgan fingerprints…")
    fps, valid_idx, blank_idx, invalid_idx = compute_fingerprints(
        df[args.smiles_col].astype(str).tolist(),
        radius=args.radius,
        nbits=args.nbits,
        cache_db=args.cache,
    )
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    log.info(f"  {len(fps):,} valid structures")

    failed_rows = []
    for i in blank_idx:
        log.warning(f"  row {i} [{df[args.id_col].iloc[i] if args.id_col else i}]: blank SMILES")
        failed_rows.append({**df.iloc[i].to_dict(), "failure_reason": "blank SMILES"})
    for i in invalid_idx:
        smi = df[args.smiles_col].iloc[i]
        log.warning(f"  row {i} [{df[args.id_col].iloc[i] if args.id_col else i}]: "
                    f"invalid SMILES — {smi}")
        failed_rows.append({**df.iloc[i].to_dict(), "failure_reason": "invalid SMILES"})

    if failed_rows:
        failed_path = out_prefix + "_hitpicker_failed.csv"
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
        log.warning(f"  {len(failed_rows)} failed → {failed_path}")

    if len(fps) == 0:
        sys.exit("No valid SMILES found.")

    # ── activity → labels ─────────────────────────────────────────────────────
    score_prefix = "zscore" if args.act_method == "robust_zscore" else "norm"
    per_col_hits, per_col_scores = [], []
    log.info(f"Hit labelling ({args.act_method}, min_active={min_active}/{n_cols}):")
    for col, thr, dirn in zip(args.act_col, thresholds, directions):
        activity = df_valid[col].astype(float).values
        col_hits, col_scores = label_hits(activity, args.act_method, thr, dirn)
        per_col_hits.append(col_hits)
        per_col_scores.append(col_scores)
        log.info(f"  {col}: {int(col_hits.sum())}/{len(col_hits)} hits "
                 f"(threshold={thr}, direction={dirn})")

    composite_y = (np.stack(per_col_hits, axis=1).sum(axis=1) >= min_active).astype(int)
    n_hits = int(composite_y.sum())
    hit_rate = n_hits / len(composite_y) * 100
    log.info(f"  Composite: {n_hits}/{len(composite_y)} hits ({hit_rate:.1f}%)")
    if n_hits < 5:
        log.warning("Fewer than 5 composite hits — consider adjusting thresholds or --min_active.")

    # ── catalog / REOS flagging (before training) ─────────────────────────────
    catalog_names = [n for n in args.catalogs if n.lower() != "none"]
    pains_catalog, reos_rules = None, []
    if catalog_names:
        pains_catalog = _make_filter_catalog(catalog_names)
        log.info(f"Loaded RDKit catalog(s): {', '.join(catalog_names)}")
    if not args.no_reos:
        rf = Path(args.reos_file)
        if rf.exists():
            reos_rules = _load_reos(rf)
            log.info(f"Loaded {len(reos_rules)} REOS rules.")
        else:
            log.warning(f"REOS file not found: {rf}")

    pains_flags = reos_flags = None
    if pains_catalog is not None or reos_rules:
        log.info("Flagging PAINS / REOS…")
        pains_flags, reos_flags = annotate_filters(
            df_valid[args.smiles_col].astype(str).tolist(), pains_catalog, reos_rules
        )
        if pains_catalog is not None:
            n_pains = sum(1 for f in pains_flags if f)
            log.info(f"  Catalog flags: {n_pains}/{len(pains_flags)} flagged")
        if reos_rules:
            n_reos = sum(1 for f in reos_flags if f)
            log.info(f"  REOS:  {n_reos}/{len(reos_flags)} flagged")

    # ── mask catalog-flagged hits from training labels ────────────────────────
    # Only exclude when PAINS is among the loaded catalogs — PAINS flags are
    # assay-interference artifacts; other catalogs (BRENK etc.) flag
    # undesirable but potentially real actives, so we leave those in.
    y_train = composite_y.copy()
    has_pains = bool(pains_flags) and any(n in _PAINS_NAMES for n in catalog_names)
    if has_pains and args.exclude_pains_from_training:
        pains_mask = np.array([bool(f) for f in pains_flags])
        n_pains_hits = int((composite_y[pains_mask] == 1).sum())
        if n_pains_hits > 0:
            y_train[pains_mask] = 0
            log.info(f"  Excluded {n_pains_hits} PAINS-flagged hits from training labels.")

    n_train_hits = int(y_train.sum())
    if n_train_hits < 5:
        log.warning(f"Only {n_train_hits} clean training hits — model may be weak.")

    X = fps

    # ── cross-validation + conformal prediction ───────────────────────────────
    cv_proba = cp_p_hit = cp_p_nonhit = None
    if not args.no_cv and n_train_hits >= 2 and (len(y_train) - n_train_hits) >= 2:
        log.info(f"Running {args.n_folds}-fold stratified CV with conformal prediction…")
        cv_proba, cp_p_hit, cp_p_nonhit = run_cv(
            X, y_train, n_splits=args.n_folds, seed=args.seed
        )
    elif args.no_cv:
        log.info("CV skipped (--no_cv).")
    else:
        log.warning("Too few training hits or non-hits for CV; skipping.")

    # ── final model ───────────────────────────────────────────────────────────
    log.info("Training final model on full dataset…")
    final_model = _lgbm(args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X, y_train)
        final_proba = final_model.predict_proba(X)[:, 1]

    # ── MVS-A sample influence ────────────────────────────────────────────────
    mvsa_scores = None
    if not args.no_mvsa:
        log.info("Computing MVS-A sample influence scores…")
        mvsa_scores = compute_mvsa(final_model, X, y_train)
        n_hits_check = int(composite_y.sum())
        if n_hits_check > 0:
            hit_mvsa = mvsa_scores[composite_y == 1]
            log.info(
                f"  MVS-A (hits): median={np.median(hit_mvsa):.3f}, "
                f"min={hit_mvsa.min():.3f}, max={hit_mvsa.max():.3f}  "
                f"(low = likely true positive)"
            )

    # ── assemble output ───────────────────────────────────────────────────────
    df_out = df_valid.copy()
    score_cols = []

    if n_cols == 1:
        sc = "zscore" if score_prefix == "zscore" else "norm_activity"
        df_out[sc] = np.round(per_col_scores[0], 4)
        df_out["is_hit"] = per_col_hits[0]
        score_cols.append(sc)
    else:
        for col, scores_i, hits_i in zip(args.act_col, per_col_scores, per_col_hits):
            sc = f"{score_prefix}_{col}"
            df_out[sc] = np.round(scores_i, 4)
            df_out[f"is_hit_{col}"] = hits_i
            score_cols.append(sc)
        df_out["is_hit"] = composite_y

    if cv_proba is not None:
        df_out["cv_hit_prob"] = np.round(cv_proba, 4)
    df_out["hit_prob"] = np.round(final_proba, 4)

    if cp_p_hit is not None:
        alpha = args.cp_alpha
        df_out["cp_p_hit"]       = np.round(cp_p_hit, 4)
        df_out["cp_p_nonhit"]    = np.round(cp_p_nonhit, 4)
        df_out["cp_confidence"]  = np.round(1.0 - np.minimum(cp_p_hit, cp_p_nonhit), 4)
        df_out["cp_credibility"] = np.round(np.maximum(cp_p_hit, cp_p_nonhit), 4)

        # Prediction set at significance level α
        in_hit    = cp_p_hit    > alpha
        in_nonhit = cp_p_nonhit > alpha
        pred = np.where(
            in_hit & ~in_nonhit,  "hit",
            np.where(~in_hit & in_nonhit, "nonhit",
            np.where(in_hit & in_nonhit,  "uncertain", "outlier"))
        )
        df_out["cp_prediction"] = pred

        # Validity and efficiency on the CV data (uses y_train labels)
        y_arr = y_train
        valid = np.mean(
            ((y_arr == 1) & in_hit) | ((y_arr == 0) & in_nonhit)
        )
        singletons = np.mean(in_hit.astype(int) + in_nonhit.astype(int) == 1)
        log.info(
            f"Conformal prediction (α={alpha}, {int((1-alpha)*100)}% confidence): "
            f"validity={valid:.3f} (target≥{1-alpha:.2f}), "
            f"efficiency={singletons:.3f} (fraction singleton sets)"
        )

    if pains_flags is not None:
        df_out["struct_alerts"] = pains_flags
    if reos_flags is not None:
        df_out["reos_violations"] = reos_flags
    if mvsa_scores is not None:
        df_out["mvsa_score"] = np.round(mvsa_scores, 4)

    # ── fp_flags: composite false-positive risk for confirmed hits ─────────────
    # Follows Boldini 2024: rank hits by MVS-A, then use structural alerts as
    # an additional post-filter. Hits are flagged by any of:
    #   high_mvsa    — mvsa_score above median of all confirmed hits
    #                  (model was never confident this is a real active)
    #   struct_alert — matched a PAINS / catalog pattern
    #   reos         — violated a REOS rule
    # Sort: 0-flag hits first (safest), then 1-flag, then 2+; MVS-A as tiebreak.
    # Non-hits get no fp_flags — high MVS-A there means likely false negative.
    if mvsa_scores is not None:
        hit_rows = df_out["is_hit"] == 1
        hit_mvsa_median = df_out.loc[hit_rows, "mvsa_score"].median()

        fp_flag_lists: list[str] = []
        for _, row in df_out.iterrows():
            f: list[str] = []
            if row["is_hit"] == 1:
                if row.get("mvsa_score", np.nan) > hit_mvsa_median:
                    f.append("high_mvsa")
                if row.get("struct_alerts", ""):
                    f.append("struct_alert")
                if row.get("reos_violations", ""):
                    f.append("reos")
            fp_flag_lists.append(";".join(f))

        df_out["fp_flags"] = fp_flag_lists
        df_out["_n_fp"]    = df_out["fp_flags"].apply(
            lambda s: len(s.split(";")) if s else 0
        )

        n_high_mvsa = (df_out.loc[hit_rows, "fp_flags"].str.contains("high_mvsa")).sum()
        n_struct     = (df_out.loc[hit_rows, "fp_flags"].str.contains("struct_alert")).sum()
        n_reos_fp    = (df_out.loc[hit_rows, "fp_flags"].str.contains("reos")).sum()
        n_multi      = (df_out.loc[hit_rows, "_n_fp"] >= 2).sum()
        log.info(
            f"False-positive flags (confirmed hits): "
            f"high_mvsa={n_high_mvsa}, struct_alert={n_struct}, "
            f"reos={n_reos_fp}, 2+_flags={n_multi}"
        )

    # ── sort ──────────────────────────────────────────────────────────────────
    # Confirmed hits: fewest FP flags first, then MVS-A ascending within each tier
    # Non-hits:       MVS-A descending (high = likely false negative; Boldini 2024)
    # Fallback to z-score / cv_hit_prob when MVS-A was skipped (--no_mvsa).
    ml_rank_col = "cv_hit_prob" if cv_proba is not None else "hit_prob"
    primary_score = score_cols[0]

    if mvsa_scores is not None:
        df_hits    = (df_out[df_out["is_hit"] == 1]
                      .sort_values(["_n_fp", "mvsa_score"], ascending=[True, True]))
        df_nonhits = df_out[df_out["is_hit"] == 0].sort_values("mvsa_score", ascending=False)
        df_out = pd.concat([df_hits, df_nonhits]).reset_index(drop=True)
        df_out = df_out.drop(columns=["_n_fp"])
    else:
        df_hits    = df_out[df_out["is_hit"] == 1].sort_values(primary_score, ascending=False)
        df_nonhits = df_out[df_out["is_hit"] == 0].sort_values(ml_rank_col,   ascending=False)
        df_out = pd.concat([df_hits, df_nonhits]).reset_index(drop=True)
    df_out.insert(0, "rank", df_out.index + 1)

    out_path = out_prefix + "_hitpicker_ranked.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"Wrote {len(df_out):,} compounds → {out_path}")

    top_n = min(100, n_hits) if n_hits > 0 else 100
    top_hit_rate = df_out.head(top_n)["is_hit"].mean() * 100
    log.info(
        f"Top-{top_n} hit rate: {top_hit_rate:.1f}%  "
        f"(overall: {hit_rate:.1f}%,  clean training hits: {n_train_hits})"
    )

    if args.report:
        write_report(df_out, args.act_col, score_cols, out_prefix)


if __name__ == "__main__":
    main()
