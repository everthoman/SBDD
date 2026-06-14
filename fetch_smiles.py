"""
Fetch canonical SMILES from PubChem for compounds in a CSV/TSV file.

Usage:
    python3 fetch_smiles.py input.csv
    python3 fetch_smiles.py input.csv --name-col "Drug Name"
    python3 fetch_smiles.py input.csv --output results.csv --delay 0.3
    python3 fetch_smiles.py input.csv --check          # validate after fetching
    python3 fetch_smiles.py input.csv --check-only     # validate existing output
"""

import argparse
import csv
import re
import sys
import time
import requests
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT = True
except ImportError:
    RDKIT = False

# PubChem PUG REST endpoints
# Note: PubChem renamed properties — SMILES=isomeric, ConnectivitySMILES=canonical
_PROPS = "SMILES,ConnectivitySMILES,IUPACName,MolecularFormula,MolecularWeight"
NAME_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    f"/property/{_PROPS}/JSON"
)
CID_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/cids/JSON?name_type=word"
)
PROP_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
    f"/property/{_PROPS}/JSON"
)
SMILES_VALIDATE_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
)

# Column name keywords used for auto-detection (checked in order)
NAME_KEYWORDS = ["compound", "name", "drug", "molecule", "chemical", "substance", "ligand"]

# Salt/form suffixes stripped during name cleaning, tried left-to-right
STRIP_PATTERNS = [
    r"\s+hydrochloride$", r"\s+dihydrochloride$", r"\s+sodium$", r"\s+sulfate$",
    r"\s+bisulfate$", r"\s+calcium$", r"\s+potassium$", r"\s+acetate$",
    r"\s+phosphate$", r"\s+maleate$", r"\s+mesylate$", r"\s+pegol$",
    r"\s+polysulfate\s+sodium$", r"\s+polysulfate$", r"\s+sorbitex\s+calcium$",
    r"\s+recombinant\s+human$", r"\s+recombinant$", r"\s+synthetic\s+human$",
    r",\s*recombinant$", r"\s+complex$", r"\s+kit$",
    r"\s+alfa[-/][^\s]*$", r"\s+alfa$", r"\s+beta[-/][^\s]*$", r"\s+beta$",
    r"\s+gamma[-\w]*$", r"\s+type\s+[ab]$",
]

RETRY_DELAY = 5
MW_TOLERANCE_PCT = 1.0  # flag if computed MW differs by more than this %

# Atomic weights (IUPAC 2021 standard)
ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.96, "Tc": 98.0, "Ru": 101.07, "Rh": 102.906,
    "Pd": 106.42, "Ag": 107.868, "Cd": 112.411, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905,
    "Ba": 137.327, "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925,
    "Dy": 162.500, "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.054,
    "Lu": 174.967, "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.592,
    "Tl": 204.383, "Pb": 207.2, "Bi": 208.980, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.038, "Pa": 231.036,
    "U": 238.029, "Np": 237.0, "Pu": 244.0,
}


def mw_from_formula(formula: str) -> float | None:
    """Compute MW from a molecular formula string like C14H13N3O4S2."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    mw = 0.0
    for element, count in tokens:
        if not element:
            continue
        w = ATOMIC_WEIGHTS.get(element)
        if w is None:
            return None
        mw += w * (int(count) if count else 1)
    return round(mw, 3) if mw > 0 else None


def validate_smiles_pubchem(smiles: str, delay: float) -> bool:
    """Ask PubChem to parse the SMILES; returns True if it gets a CID back."""
    resp = http_get(SMILES_VALIDATE_URL.format(smiles=requests.utils.quote(smiles)), delay)
    time.sleep(delay)
    if resp is None or resp.status_code != 200:
        return False
    cids = resp.json().get("IdentifierList", {}).get("CID", [])
    return len(cids) > 0


def validate_smiles_rdkit(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def check_row(row: dict, delay: float) -> dict:
    """
    Validate SMILES and check MW consistency.
    Returns dict with SmilesValid, MWCheck, MWDelta_pct, CheckNotes.
    """
    smiles = (row.get("CanonicalSMILES") or row.get("IsomericSMILES") or "").strip()
    formula = row.get("MolecularFormula", "").strip()
    reported_mw_str = row.get("MolecularWeight", "").strip()

    if not smiles:
        return {"SmilesValid": "", "MWCheck": "", "MWDelta_pct": "", "CheckNotes": "no_smiles"}

    # SMILES validity
    if RDKIT:
        valid = validate_smiles_rdkit(smiles)
    else:
        valid = validate_smiles_pubchem(smiles, delay)

    # MW consistency via molecular formula
    mw_check = ""
    mw_delta = ""
    notes = []

    if formula and reported_mw_str:
        try:
            reported_mw = float(reported_mw_str)
            computed_mw = mw_from_formula(formula)
            if computed_mw is not None and reported_mw > 0:
                delta_pct = abs(computed_mw - reported_mw) / reported_mw * 100
                mw_delta = f"{delta_pct:.2f}"
                mw_check = "OK" if delta_pct <= MW_TOLERANCE_PCT else "MISMATCH"
                if mw_check == "MISMATCH":
                    notes.append(f"MW: computed={computed_mw:.2f} reported={reported_mw:.2f}")
            else:
                mw_check = "UNKNOWN"
        except ValueError:
            mw_check = "UNKNOWN"

    if not valid:
        notes.append("invalid_smiles")

    return {
        "SmilesValid": str(valid),
        "MWCheck": mw_check,
        "MWDelta_pct": mw_delta,
        "CheckNotes": "; ".join(notes) if notes else "OK",
    }


def refetch_smiles(output_path: Path, name_col: str, delay: float):
    """Re-fetch SMILES by CID for OK rows that have an empty CanonicalSMILES."""
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    targets = [r for r in rows if r.get("PubChemStatus") == "OK"
               and not r.get("CanonicalSMILES", "").strip()
               and r.get("CID", "").strip()]
    print(f"Re-fetching SMILES for {len(targets)} OK rows with empty SMILES...")

    updated = 0
    for i, row in enumerate(targets):
        cid = row["CID"].strip()
        resp = http_get(PROP_URL.format(cid=cid), delay)
        time.sleep(delay)
        if resp and resp.status_code == 200:
            props = _extract_props(resp.json())
            row.update(props)
            updated += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(targets)}...")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {updated}/{len(targets)} rows → {output_path}")


def run_checks(output_path: Path, name_col: str, delay: float):
    print(f"\n{'='*60}")
    backend = "RDKit" if RDKIT else "PubChem API"
    print(f"Running validation checks (SMILES backend: {backend})...")

    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    check_fields = ["SmilesValid", "MWCheck", "MWDelta_pct", "CheckNotes"]
    out_fieldnames = [f for f in fieldnames if f not in check_fields] + check_fields

    total = sum(1 for r in rows if r.get("PubChemStatus") == "OK")
    done = 0
    flagged = 0

    for row in rows:
        if row.get("PubChemStatus") != "OK":
            for f in check_fields:
                row[f] = ""
            continue
        result = check_row(row, delay)
        row.update(result)
        done += 1
        if result["CheckNotes"] not in ("OK", ""):
            flagged += 1
            print(f"  FLAG  {row[name_col]}: {result['CheckNotes']}")
        if done % 100 == 0:
            print(f"  checked {done}/{total}...")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nChecked {done} compounds — {flagged} flagged. Results written to {output_path}")


def detect_encoding(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                f.read(4096)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def sniff_delimiter(path: Path, encoding: str = "utf-8") -> str:
    with open(path, newline="", encoding=encoding) as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        return dialect.delimiter
    except csv.Error:
        return ","


def detect_name_column(fieldnames: list[str]) -> str | None:
    lower = [f.lower() for f in fieldnames]
    for kw in NAME_KEYWORDS:
        matches = [fieldnames[i] for i, l in enumerate(lower) if kw in l]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches[0]  # take the first/best match
    return None


def name_variants(name: str) -> list[str]:
    name = name.strip()
    variants = [name]
    for pattern in STRIP_PATTERNS:
        cleaned = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
        if cleaned and cleaned != name and cleaned not in variants:
            variants.append(cleaned)
    return variants


def http_get(url: str, delay: float) -> requests.Response | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code in (429, 503):
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return resp
        except requests.RequestException:
            time.sleep(RETRY_DELAY)
    return None


def fetch_by_name(name: str, delay: float) -> dict | None:
    """Exact name lookup. Returns props dict, {} if not found, None on error."""
    resp = http_get(NAME_URL.format(name=requests.utils.quote(name)), delay)
    time.sleep(delay)
    if resp is None:
        return None
    if resp.status_code == 404:
        return {}
    if resp.status_code == 200:
        return _extract_props(resp.json())
    return None


def fetch_by_synonym(name: str, delay: float) -> dict | None:
    """Word/synonym search fallback. Returns props dict, {} if not found, None on error."""
    resp = http_get(CID_URL.format(name=requests.utils.quote(name)), delay)
    time.sleep(delay)
    if resp is None:
        return None
    if resp.status_code == 404:
        return {}
    if resp.status_code != 200:
        return None
    cids = resp.json().get("IdentifierList", {}).get("CID", [])
    if not cids:
        return {}
    resp2 = http_get(PROP_URL.format(cid=cids[0]), delay)
    time.sleep(delay)
    if resp2 is None or resp2.status_code != 200:
        return None
    return _extract_props(resp2.json())


def _extract_props(data: dict) -> dict:
    p = data["PropertyTable"]["Properties"][0]
    return {
        "CID": p.get("CID", ""),
        "CanonicalSMILES": p.get("ConnectivitySMILES", ""),
        "IsomericSMILES": p.get("SMILES", ""),
        "IUPACName": p.get("IUPACName", ""),
        "MolecularFormula": p.get("MolecularFormula", ""),
        "MolecularWeight": p.get("MolecularWeight", ""),
    }


def lookup(name: str, delay: float) -> tuple[dict | None, str]:
    """
    Try exact name, then cleaned name variants, then synonym search.
    Returns (result, method_note).
    """
    # 1. Exact name
    result = fetch_by_name(name.strip(), delay)
    if result:
        return result, "exact"
    if result is None:
        return None, "error"

    # 2. Cleaned name variants (exact search)
    for variant in name_variants(name)[1:]:  # skip [0] = original
        result = fetch_by_name(variant, delay)
        if result:
            return result, f"cleaned:'{variant}'"
        if result is None:
            return None, "error"

    # 3. Synonym / word search on original and variants
    for variant in name_variants(name):
        result = fetch_by_synonym(variant, delay)
        if result:
            return result, f"synonym:'{variant}'"
        if result is None:
            return None, "error"

    return {}, "not_found"


def parse_args():
    p = argparse.ArgumentParser(description="Fetch SMILES from PubChem by compound name.")
    p.add_argument("input", help="Input CSV/TSV file")
    p.add_argument("--name-col", help="Column containing compound names (auto-detected if omitted)")
    p.add_argument("--output", help="Output file (default: <input>_smiles.csv)")
    p.add_argument("--delay", type=float, default=0.2, help="Seconds between requests (default: 0.2)")
    p.add_argument("--check", action="store_true",
                   help="Validate SMILES and check MW after fetching")
    p.add_argument("--check-only", action="store_true",
                   help="Skip fetching; validate existing output file only")
    p.add_argument("--refetch-smiles", action="store_true",
                   help="Re-fetch SMILES by CID for OK rows with empty SMILES (fixes API rename issue)")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: file not found: {args.input}")

    encoding = detect_encoding(input_path)
    delimiter = sniff_delimiter(input_path, encoding)
    with open(input_path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        sys.exit("Error: input file is empty.")

    # Resolve name column
    name_col = args.name_col
    if name_col and name_col not in fieldnames:
        sys.exit(f"Error: column '{name_col}' not found. Available: {', '.join(fieldnames)}")
    if not name_col:
        name_col = detect_name_column(fieldnames)
        if not name_col:
            print(f"Could not auto-detect name column. Available columns: {', '.join(fieldnames)}")
            sys.exit("Use --name-col to specify it.")
        print(f"Auto-detected name column: '{name_col}'")

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_smiles.csv")

    if args.check_only:
        if not output_path.exists():
            sys.exit(f"Error: output file not found for --check-only: {output_path}")
        run_checks(output_path, name_col, args.delay)
        return

    if args.refetch_smiles:
        if not output_path.exists():
            sys.exit(f"Error: output file not found for --refetch-smiles: {output_path}")
        refetch_smiles(output_path, name_col, args.delay)
        if args.check:
            run_checks(output_path, name_col, args.delay)
        return

    extra_fields = ["CID", "CanonicalSMILES", "IsomericSMILES",
                    "IUPACName", "MolecularFormula", "MolecularWeight", "PubChemStatus"]
    out_fieldnames = [f for f in fieldnames if f not in extra_fields] + extra_fields

    # Resume support: load already-processed names
    done: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done[row[name_col]] = row
        print(f"Resuming: {len(done)}/{len(rows)} already done")
        out_file = open(output_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_file, fieldnames=out_fieldnames, extrasaction="ignore")
    else:
        out_file = open(output_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_file, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()

    counts = {"OK": 0, "NOT_FOUND": 0, "ERROR": 0}
    for existing in done.values():
        counts[existing.get("PubChemStatus", "ERROR")] += 1

    try:
        for i, row in enumerate(rows):
            name = row[name_col]
            if name in done:
                continue

            result, method = lookup(name, args.delay)

            empty = {k: "" for k in ["CID", "CanonicalSMILES", "IsomericSMILES",
                                      "IUPACName", "MolecularFormula", "MolecularWeight"]}
            if result is None:
                status = "ERROR"
                row = {**row, **empty, "PubChemStatus": status}
            elif result == {}:
                status = "NOT_FOUND"
                row = {**row, **empty, "PubChemStatus": status}
            else:
                status = "OK"
                row = {**row, **result, "PubChemStatus": status}

            counts[status] += 1
            writer.writerow(row)
            out_file.flush()
            done[name] = row

            total_done = len(done)
            if total_done % 50 == 0 or total_done == len(rows):
                pct = 100 * total_done / len(rows)
                print(f"  {total_done}/{len(rows)} ({pct:.0f}%)  "
                      f"found={counts['OK']}  not_found={counts['NOT_FOUND']}  errors={counts['ERROR']}")
    finally:
        out_file.close()

    print(f"\nDone. {counts['OK']} found, {counts['NOT_FOUND']} not found, "
          f"{counts['ERROR']} errors → {output_path}")

    if args.check or args.check_only:
        run_checks(output_path, name_col, args.delay)


if __name__ == "__main__":
    main()
