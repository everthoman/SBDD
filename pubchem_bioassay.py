#!/usr/bin/env python3
"""
pubchem_bioassay.py — assess compound promiscuity via PubChem bioassay data

Usage:
    python pubchem_bioassay.py input.csv -o output.csv
    python pubchem_bioassay.py input.csv -o output.csv --inchikey-col InChIKey
    python pubchem_bioassay.py input.csv -o output.csv --smiles-col SMILES
"""

import argparse
import sys
import time
from io import StringIO
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _cid_url_inchikey(key: str) -> tuple[str, str]:
    """Return (cid_url, invalid_msg) for an InChIKey identifier."""
    if pd.isna(key) or len(str(key)) < 14:
        return "", "Invalid InChIKey"
    connectivity_layer = str(key).strip().split("-")[0]
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey"
        f"/{connectivity_layer}/cids/JSON"
    )
    return url, ""


def _cid_url_smiles(smiles: str) -> tuple[str, str]:
    """Return (cid_url, invalid_msg) for a SMILES identifier."""
    if pd.isna(smiles) or not str(smiles).strip():
        return "", "Invalid SMILES"
    encoded = quote(str(smiles).strip(), safe="")
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles"
        f"/{encoded}/cids/JSON"
    )
    return url, ""


def query_compound(identifier: str, id_type: str, delay: float) -> dict:
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

    if id_type == "inchikey":
        cid_url, invalid_msg = _cid_url_inchikey(identifier)
    else:
        cid_url, invalid_msg = _cid_url_smiles(identifier)

    if not cid_url:
        result["PubChem_Status"] = invalid_msg
        return result

    try:
        cid_resp = requests.get(cid_url, timeout=15)

        if cid_resp.status_code != 200:
            result["PubChem_Status"] = "Not Found in PubChem"
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
                "Total_Assays": int(total_count),
                "Active_Assays": int(len(active_df)),
                "Unique_Targets_Tested": int(c_tested),
                "Unique_Targets_Active": int(c_active),
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
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")

    id_group = parser.add_mutually_exclusive_group()
    id_group.add_argument(
        "--inchikey-col",
        default=None,
        help="InChIKey column name (auto-detected if any column contains 'inchi')",
    )
    id_group.add_argument(
        "--smiles-col",
        default=None,
        help="SMILES column name (auto-detected if any column contains 'smiles')",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Seconds between PubChem requests (default: 0.35)",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Input file delimiter (default: auto-detect)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=args.sep, engine="python")

    # Resolve identifier column and type
    if args.smiles_col:
        id_col = args.smiles_col
        id_type = "smiles"
        if id_col not in df.columns:
            sys.exit(f"Error: column '{id_col}' not found. Available: {list(df.columns)}")
    elif args.inchikey_col:
        id_col = args.inchikey_col
        id_type = "inchikey"
        if id_col not in df.columns:
            sys.exit(f"Error: column '{id_col}' not found. Available: {list(df.columns)}")
    else:
        # Auto-detect: prefer InChIKey, fall back to SMILES
        inchi_candidates = [c for c in df.columns if "inchi" in c.lower()]
        smiles_candidates = [c for c in df.columns if "smiles" in c.lower()]
        if inchi_candidates:
            id_col, id_type = inchi_candidates[0], "inchikey"
        elif smiles_candidates:
            id_col, id_type = smiles_candidates[0], "smiles"
        else:
            sys.exit(
                "Error: no InChIKey or SMILES column found. "
                "Use --inchikey-col or --smiles-col. "
                f"Available: {list(df.columns)}"
            )

    print(f"Input:          {args.input}  ({len(df)} rows)")
    print(f"Identifier col: {id_col}  ({id_type})")
    print(f"Output:         {args.output}")

    identifiers = df[id_col].tolist()
    rows = []

    iterator = tqdm(identifiers, unit="cpd") if HAS_TQDM else identifiers
    if not HAS_TQDM:
        print("(install tqdm for a progress bar)")

    for i, identifier in enumerate(iterator):
        result = query_compound(identifier, id_type, args.delay)
        rows.append(result)
        if not HAS_TQDM and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(identifiers)} done")

    results_df = pd.DataFrame(rows)
    int_cols = ["Total_Assays", "Active_Assays", "Unique_Targets_Tested", "Unique_Targets_Active"]
    results_df[int_cols] = results_df[int_cols].astype("Int64")
    out_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    out_df.to_csv(args.output, index=False)

    n_ok = (results_df["PubChem_Status"] == "Success").sum()
    print(f"\nDone: {n_ok}/{len(df)} succeeded  →  {args.output}")


if __name__ == "__main__":
    main()
