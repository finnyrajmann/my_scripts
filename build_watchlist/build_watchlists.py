#!/usr/bin/env python3
"""
NSE Watchlist Builder
=====================
Converts NSE Market Watch CSV downloads into Yahoo Finance compatible
watchlist CSVs for use in DO trading systems.

Usage:
    python3 build_watchlists.py

Input files (place in same directory or set paths below):
    - MW-NIFTY-500-<date>.csv
    - MW-NIFTY-LARGEMID250-<date>.csv
    - MW-SECURITIES-IN-F_O-<date>.csv

Output files (written to same directory):
    - watchlist_nifty500.csv
    - watchlist_nifty250.csv
    - watchlist_fno.csv

Run this quarterly or bi-annually to refresh your universe.
"""

import pandas as pd
import os
import sys
import glob
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────

# Directory where your NSE CSV downloads live
# Change this to the folder where you save the NSE files
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory (same as input by default)
OUTPUT_DIR = INPUT_DIR

# ── Known Symbol Corrections ─────────────────────────────────────────────────
# NSE symbol → correct Yahoo Finance ticker (without .NS)
# Add any new corrections here as you discover them

SYMBOL_CORRECTIONS = {
    "ZOMATO":    "ETERNAL",   # Renamed to Eternal Ltd
    "LTIM":      "LTIM",      # Usually fine, included for reference
}

# Symbols to exclude entirely (demerged, illiquid, or problematic on YF)
SYMBOLS_TO_EXCLUDE = {
    "TATAMTRDVR",   # DVR share - Yahoo Finance unreliable
}

# ── Helper Functions ──────────────────────────────────────────────────────────

def find_latest_file(pattern: str) -> str | None:
    """Find the most recent file matching a glob pattern."""
    files = glob.glob(os.path.join(INPUT_DIR, pattern))
    if not files:
        return None
    # Sort by modification time, return latest
    return sorted(files, key=os.path.getmtime)[-1]


def parse_nse_csv(filepath: str, label: str) -> pd.DataFrame:
    """
    Parse an NSE Market Watch CSV.
    - Strips whitespace/newlines from column names
    - Drops the index row (first row where SYMBOL = index name)
    - Returns a clean DataFrame with a 'SYMBOL' column
    """
    print(f"\n[{label}] Reading: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    # Clean column names (they have \n and trailing spaces)
    df.columns = [col.strip().replace("\n", "").replace(" ", "_") for col in df.columns]

    # The symbol column after cleaning
    sym_col = "SYMBOL"
    if sym_col not in df.columns:
        # Try to find it
        sym_col = [c for c in df.columns if "SYMBOL" in c.upper()][0]

    df = df.rename(columns={sym_col: "SYMBOL"})
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

    # Drop the index summary row (first row — it's the index itself e.g. "NIFTY 500")
    df = df[~df["SYMBOL"].str.contains(" ", na=False)].copy()

    print(f"[{label}] Raw symbols loaded: {len(df)}")
    return df[["SYMBOL"]].reset_index(drop=True)


def apply_corrections(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Apply symbol corrections and exclusions."""
    original_count = len(df)

    # Apply renames
    df["SYMBOL"] = df["SYMBOL"].replace(SYMBOL_CORRECTIONS)

    # Remove excluded symbols
    excluded = df[df["SYMBOL"].isin(SYMBOLS_TO_EXCLUDE)]["SYMBOL"].tolist()
    if excluded:
        print(f"[{label}] Excluding symbols: {excluded}")
    df = df[~df["SYMBOL"].isin(SYMBOLS_TO_EXCLUDE)].copy()

    # Log renames
    for old, new in SYMBOL_CORRECTIONS.items():
        if new in df["SYMBOL"].values:
            print(f"[{label}] Renamed: {old} → {new}")

    print(f"[{label}] After corrections: {len(df)} symbols (was {original_count})")
    return df


def to_yahoo_format(df: pd.DataFrame) -> pd.DataFrame:
    """Append .NS suffix to convert NSE symbols to Yahoo Finance format."""
    df = df.copy()
    df["Symbol"] = df["SYMBOL"] + ".NS"
    return df[["Symbol"]]


def save_watchlist(df: pd.DataFrame, filename: str, label: str):
    """Save the final watchlist CSV."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"[{label}] ✅ Saved: {filename} ({len(df)} symbols)")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NSE Watchlist Builder")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Input dir: {INPUT_DIR}")
    print("=" * 60)

    saved_files = []

    # ── 1. Nifty 500 ──────────────────────────────────────────────────────────
    f500 = find_latest_file("MW-NIFTY-500-*.csv")
    if f500:
        df500 = parse_nse_csv(f500, "NIFTY500")
        df500 = apply_corrections(df500, "NIFTY500")
        df500_out = to_yahoo_format(df500)
        saved_files.append(save_watchlist(df500_out, "watchlist_nifty500.csv", "NIFTY500"))
    else:
        print("\n[NIFTY500] ⚠️  File not found. Expected: MW-NIFTY-500-<date>.csv")

    # ── 2. Nifty LargeMid 250 ─────────────────────────────────────────────────
    f250 = find_latest_file("MW-NIFTY-LARGEMID250-*.csv")
    if f250:
        df250 = parse_nse_csv(f250, "NIFTY250")
        df250 = apply_corrections(df250, "NIFTY250")
        df250_out = to_yahoo_format(df250)
        saved_files.append(save_watchlist(df250_out, "watchlist_nifty250.csv", "NIFTY250"))
    else:
        print("\n[NIFTY250] ⚠️  File not found. Expected: MW-NIFTY-LARGEMID250-<date>.csv")

    # ── 3. F&O Universe ───────────────────────────────────────────────────────
    ffno = find_latest_file("MW-SECURITIES-IN-F_O-*.csv")
    if ffno:
        dffno = parse_nse_csv(ffno, "FNO")
        dffno = apply_corrections(dffno, "FNO")
        dffno_out = to_yahoo_format(dffno)
        saved_files.append(save_watchlist(dffno_out, "watchlist_fno.csv", "FNO"))
    else:
        print("\n[FNO] ⚠️  File not found. Expected: MW-SECURITIES-IN-F_O-<date>.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for f in saved_files:
        df_check = pd.read_csv(f)
        print(f"  {os.path.basename(f):<30} {len(df_check)} symbols")

    print("\nDone. Copy these CSVs to your GitHub watchlist repo to update DO systems.")
    print("=" * 60)


if __name__ == "__main__":
    main()
