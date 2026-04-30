#!/usr/bin/env python3
"""
pubchem_bioassay.py — assess compound promiscuity via PubChem bioassay data

Usage:
    python pubchem_bioassay.py input.csv -o output.csv
    python pubchem_bioassay.py input.csv -o output.csv --inchikey-col InChIKey --delay 0.5
"""

import argparse
import sys
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def query_compound(key: str, delay: float) -> dict:
    result = {
        "PubChem_CID": "None",
        "Total_Assays": np.nan,
        "Active_Assays": np.nan,
        "Unique_Targets_Tested": np.nan,
        "Unique_Targets_Active": np.nan,
        "Target_Promiscuity_Index": np.nan,
        "List_Targets_Tested": "",
        "List_Targets_Active": "",
        "PubChem_Status": "",
    }

    if pd.isna(key) or len(str(key)) < 14:
        result["PubChem_Status"] = "Invalid InChIKey"
        return result

    clean_key = str(key).strip()
    connectivity_layer = clean_key.split("-")[0]

    try:
        cid_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey"
            f"/{connectivity_layer}/cids/JSON"
        )
        cid_resp = requests.get(cid_url, timeout=15)

        if cid_resp.status_code != 200:
            result["PubChem_Status"] = "Skeleton Not Found"
            return result

        cid = cid_resp.json()["IdentifierList"]["CID"][0]
        result["PubChem_CID"] = str(cid)

        activity_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
            f"/{cid}/assaysummary/CSV"
        )
        activity_resp = requests.get(activity_url, timeout=25)

        if activity_resp.status_code != 200 or len(activity_resp.text.strip()) <= 30:
            result["PubChem_Status"] = "No bioassay data"
            return result

        df_assay = pd.read_csv(StringIO(activity_resp.text), engine="python")
        df_assay.columns = [
            c.strip().lower().replace(" ", "_") for c in df_assay.columns
        ]

        outcome_col = next(
            (c for c in df_assay.columns if "outcome" in c), None
        )
        target_col = next(
            (
                c
                for c in df_assay.columns
                if "geneid" in c or "accession" in c or "targetid" in c
            ),
            None,
        )

        if not outcome_col:
            result["PubChem_Status"] = "Outcome col missing"
            return result

        active_df = df_assay[
            df_assay[outcome_col].astype(str).str.lower() == "active"
        ]
        total_count = len(df_assay)

        if target_col:
            all_targets = df_assay[target_col].dropna().unique()
            active_targets = active_df[target_col].dropna().unique()

            def fmt(x):
                return str(int(x)) if isinstance(x, (int, float)) else str(x)

            t_tested_str = ", ".join(sorted(fmt(t) for t in all_targets))
            t_active_str = ", ".join(sorted(fmt(t) for t in active_targets))
            c_tested, c_active = len(all_targets), len(active_targets)
        else:
            t_tested_str = t_active_str = ""
            c_tested = c_active = 0

        tpi = (c_active / c_tested * 100) if c_tested > 0 else 0.0

        result.update(
            {
                "Total_Assays": float(total_count),
                "Active_Assays": float(len(active_df)),
                "Unique_Targets_Tested": float(c_tested),
                "Unique_Targets_Active": float(c_active),
                "Target_Promiscuity_Index": round(tpi, 2),
                "List_Targets_Tested": t_tested_str,
                "List_Targets_Active": t_active_str,
                "PubChem_Status": "Success",
            }
        )

    except Exception as e:
        result["PubChem_CID"] = "Error"
        result["PubChem_Status"] = f"Error: {str(e)[:60]}"

    finally:
        time.sleep(delay)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Assess compound promiscuity using PubChem bioassay data."
    )
    parser.add_argument("input", help="Input CSV file with an InChIKey column")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--inchikey-col",
        default=None,
        help="InChIKey column name (default: auto-detect by 'inchi' in name)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Seconds between PubChem requests (default: 0.35)",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Input CSV delimiter (default: ',')",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=args.sep)

    if args.inchikey_col:
        ikey_col = args.inchikey_col
        if ikey_col not in df.columns:
            sys.exit(f"Error: column '{ikey_col}' not found. Available: {list(df.columns)}")
    else:
        candidates = [c for c in df.columns if "inchi" in c.lower()]
        if not candidates:
            sys.exit(f"Error: no InChIKey column found. Use --inchikey-col. Available: {list(df.columns)}")
        ikey_col = candidates[0]

    print(f"Input:       {args.input}  ({len(df)} rows)")
    print(f"InChIKey col: {ikey_col}")
    print(f"Output:      {args.output}")

    keys = df[ikey_col].tolist()
    rows = []

    iterator = tqdm(keys, unit="cpd") if HAS_TQDM else keys
    if not HAS_TQDM:
        print("(install tqdm for a progress bar)")

    for i, key in enumerate(iterator):
        result = query_compound(key, args.delay)
        rows.append(result)
        if not HAS_TQDM and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(keys)} done")

    results_df = pd.DataFrame(rows)
    out_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    out_df.to_csv(args.output, index=False)

    n_ok = (results_df["PubChem_Status"] == "Success").sum()
    print(f"\nDone: {n_ok}/{len(df)} succeeded  →  {args.output}")


if __name__ == "__main__":
    main()
