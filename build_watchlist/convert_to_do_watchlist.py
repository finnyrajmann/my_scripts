#!/usr/bin/env python3
"""
Convert to DO Watchlist Format
===============================
Converts .NS format watchlist CSVs into DigitalOcean system compatible
watchlist CSVs with Symbol, Industry, IsBanking columns.

Usage:
    python3 convert_to_do_watchlist.py                    # converts all three
    python3 convert_to_do_watchlist.py watchlist_fno.csv  # specific file

Input:  watchlist_nifty500.csv  (Symbol column with .NS suffix)
Output: do_watchlist_nifty500.csv (Symbol, Industry, IsBanking — no .NS)

Industry data is fetched from Yahoo Finance and cached in
industry_cache.json so subsequent runs are instant.

Add to .gitignore:
    do_watchlist_*.csv
    industry_cache.json
"""

import pandas as pd
import yfinance as yf
import json
import os
import sys
import time
import argparse
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE   = os.path.join(SCRIPT_DIR, "industry_cache.json")

BATCH_SIZE   = 10   # smaller batches for info fetching — more reliable
BATCH_DELAY  = 2    # seconds between batches

# Keywords that identify banking/financial stocks for IsBanking flag
BANKING_KEYWORDS = [
    "bank", "financial services", "finance", "nbfc", "insurance",
    "microfinance", "housing finance", "asset management",
    "brokerage", "lending", "credit", "fintech",
]

# Healthcare/pharma sectors excluded from all watchlists (ethical exclusion)
HEALTHCARE_EXCLUDE_KEYWORDS = [
    "healthcare",
    "pharmaceutical",
    "pharma",
    "biotechnology",
    "drug",
    "hospital",
    "diagnostic",
    "medical",
]

# Fallback industry if yfinance returns nothing
FALLBACK_INDUSTRY = "Unknown"

# Which input files to process by default
DEFAULT_FILES = [
    "watchlist_nifty500.csv",
    "watchlist_nifty250.csv",
    "watchlist_fno.csv",
]

# ── Cache ─────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        print(f"Cache loaded: {len(data)} symbols already known")
        return data
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ── Industry Fetch ────────────────────────────────────────────────────────────

def fetch_industry_batch(symbols_ns: list[str], cache: dict) -> dict:
    """
    Fetch industry for symbols not already in cache.
    symbols_ns: list of Yahoo Finance tickers e.g. ['RELIANCE.NS', ...]
    Updates cache in place, returns updated cache.
    """
    to_fetch = [s for s in symbols_ns if s not in cache]

    if not to_fetch:
        print("All symbols already in cache — skipping fetch.")
        return cache

    print(f"\nFetching industry data for {len(to_fetch)} symbols "
          f"({len(symbols_ns) - len(to_fetch)} already cached)...")
    print("This is a one-time operation. Grab a coffee ☕\n")

    fetched = 0
    failed  = 0

    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i:i + BATCH_SIZE]

        for sym in batch:
            try:
                info = yf.Ticker(sym).info
                industry = (
                    info.get("industry")
                    or info.get("sector")
                    or FALLBACK_INDUSTRY
                )
                cache[sym] = industry
                fetched += 1
            except Exception:
                cache[sym] = FALLBACK_INDUSTRY
                failed += 1

        done = min(i + BATCH_SIZE, len(to_fetch))
        print(f"  Progress: {done}/{len(to_fetch)}  "
              f"(fetched: {fetched}, failed/unknown: {failed})", end="\r")

        # Save cache periodically so progress isn't lost on interruption
        if (i // BATCH_SIZE) % 5 == 0:
            save_cache(cache)

        if i + BATCH_SIZE < len(to_fetch):
            time.sleep(BATCH_DELAY)

    print()
    save_cache(cache)
    print(f"Cache saved: {len(cache)} total symbols")
    return cache


# ── IsBanking ─────────────────────────────────────────────────────────────────

def is_banking(industry: str) -> bool:
    """True if industry string matches BFSI keywords."""
    if not industry or industry == FALLBACK_INDUSTRY:
        return False
    industry_lower = industry.lower()
    return any(kw in industry_lower for kw in BANKING_KEYWORDS)


# ── IsHealthcare ──────────────────────────────────────────────────────────────

def is_healthcare(industry: str) -> bool:
    """True if industry string matches healthcare/pharma keywords."""
    if not industry or industry == FALLBACK_INDUSTRY:
        return False
    industry_lower = industry.lower()
    return any(kw in industry_lower for kw in HEALTHCARE_EXCLUDE_KEYWORDS)


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_watchlist(input_file: str, cache: dict) -> str:
    """
    Convert a .NS format watchlist CSV to DO format.
    Returns path of output file.
    """
    input_path = input_file if os.path.isabs(input_file) \
                 else os.path.join(SCRIPT_DIR, input_file)

    if not os.path.exists(input_path):
        print(f"⚠  File not found: {input_path}")
        return None

    df = pd.read_csv(input_path)
    symbols_ns = df["Symbol"].tolist()   # e.g. RELIANCE.NS

    # Fetch industry for all symbols (uses cache)
    cache = fetch_industry_batch(symbols_ns, cache)

    # Build output rows
    rows = []
    for sym_ns in symbols_ns:
        sym_clean = sym_ns.replace(".NS", "")
        industry  = cache.get(sym_ns, FALLBACK_INDUSTRY)
        banking   = is_banking(industry)
        rows.append({
            "Symbol":    sym_clean,
            "Industry":  industry,
            "IsBanking": banking,
        })

    df_out = pd.DataFrame(rows)

    # Exclude healthcare/pharma symbols
    healthcare_mask = df_out["Industry"].apply(is_healthcare)
    excluded_hc = df_out[healthcare_mask]["Symbol"].tolist()
    if excluded_hc:
        print(f"   Excluded (healthcare/pharma): {len(excluded_hc)} symbols")
        print(f"   → {', '.join(excluded_hc)}")
    df_out = df_out[~healthcare_mask].reset_index(drop=True)

    # Output filename: do_watchlist_nifty500.csv etc.
    base     = os.path.splitext(os.path.basename(input_file))[0]
    out_name = f"do_{base}.csv"
    out_path = os.path.join(SCRIPT_DIR, out_name)
    df_out.to_csv(out_path, index=False)

    banking_count = df_out["IsBanking"].sum()
    unknown_count = (df_out["Industry"] == FALLBACK_INDUSTRY).sum()

    print(f"\n✅ {out_name}")
    print(f"   Total symbols : {len(df_out)}")
    print(f"   IsBanking=True: {banking_count}")
    print(f"   Unknown industry: {unknown_count}")
    print(f"   Healthcare excluded: {len(excluded_hc)}")

    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert .NS watchlists to DO format with Industry and IsBanking"
    )
    parser.add_argument(
        "files", nargs="*",
        help="Watchlist CSV files to convert (default: all three)"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-fetch all industry data (ignore cache)"
    )
    args = parser.parse_args()

    files = args.files if args.files else DEFAULT_FILES

    print("=" * 60)
    print("DO Watchlist Converter")
    print(f"Date  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Files : {files}")
    print("=" * 60)

    # Load or reset cache
    cache = {} if args.refresh else load_cache()

    outputs = []
    for f in files:
        print(f"\n{'─'*60}")
        print(f"Processing: {f}")
        print(f"{'─'*60}")
        out = convert_watchlist(f, cache)
        if out:
            outputs.append(out)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for out in outputs:
        df_check = pd.read_csv(out)
        banking = df_check["IsBanking"].sum()
        hc_excluded = sum(1 for s in pd.read_csv(out.replace("do_", ""))["Symbol"]
                         if False)  # excluded already, just note count
        print(f"  {os.path.basename(out):<35} "
              f"{len(df_check)} symbols  |  {banking} banking")

    print("\nDone. Copy do_watchlist_*.csv to your DO system data directories.")
    print("Add to .gitignore: do_watchlist_*.csv  industry_cache.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
