#!/usr/bin/env python3
"""
Watchlist Verifier
==================
Spot-checks a random sample of symbols from each watchlist CSV
by fetching the latest closing price from Yahoo Finance.

Usage:
    python3 verify_watchlists.py                  # default 20 samples per list
    python3 verify_watchlists.py --sample 50      # custom sample size
    python3 verify_watchlists.py --all            # check ALL symbols (slow ~10 min)

Place this script in the same folder as your watchlist CSVs.
"""

import pandas as pd
import yfinance as yf
import os
import sys
import time
import random
import argparse
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))

WATCHLISTS = {
    "NIFTY500":  "watchlist_nifty500.csv",
    "NIFTY250":  "watchlist_nifty250.csv",
    "FNO":       "watchlist_fno.csv",
}

# Delay between batches to avoid YF rate limiting
BATCH_SIZE   = 20
BATCH_DELAY  = 2   # seconds

# ── Core ─────────────────────────────────────────────────────────────────────

def check_symbols(symbols: list[str], label: str) -> dict:
    """
    Fetch last close price for a list of symbols in batches.
    Returns dict with 'ok' and 'failed' lists.
    """
    ok, failed = [], []
    total = len(symbols)
    print(f"\n[{label}] Checking {total} symbols in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_str = " ".join(batch)

        try:
            data = yf.download(
                batch_str,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=True,
            )

            if data.empty:
                failed.extend(batch)
                continue

            # Get close prices
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data[["Close"]] if "Close" in data.columns else data

            for sym in batch:
                if sym in close.columns:
                    last = close[sym].dropna()
                    if not last.empty:
                        ok.append((sym, round(last.iloc[-1], 2)))
                    else:
                        failed.append(sym)
                else:
                    failed.append(sym)

        except Exception as e:
            print(f"  Batch error: {e}")
            failed.extend(batch)

        # Progress
        done = min(i + BATCH_SIZE, total)
        print(f"  Progress: {done}/{total}", end="\r")

        if i + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)

    print()  # newline after \r
    return {"ok": ok, "failed": failed}


def verify_watchlist(name: str, filepath: str, sample_n: int | None):
    """Load watchlist, optionally sample, then verify."""
    if not os.path.exists(filepath):
        print(f"\n[{name}] ⚠️  File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    all_symbols = df["Symbol"].tolist()
    total_in_file = len(all_symbols)

    if sample_n and sample_n < total_in_file:
        symbols = random.sample(all_symbols, sample_n)
        print(f"\n[{name}] {total_in_file} symbols in file → sampling {sample_n}")
    else:
        symbols = all_symbols
        print(f"\n[{name}] Checking ALL {total_in_file} symbols")

    result = check_symbols(symbols, name)

    ok_count     = len(result["ok"])
    failed_count = len(result["failed"])
    checked      = ok_count + failed_count

    print(f"\n[{name}] Results: {ok_count}/{checked} OK  |  {failed_count} failed")

    if result["failed"]:
        print(f"[{name}] ❌ Failed symbols:")
        for sym in result["failed"]:
            print(f"       {sym}")
    else:
        print(f"[{name}] ✅ All sampled symbols responded fine")

    # Show a few successful ones as sanity check
    if result["ok"]:
        sample_ok = result["ok"][:5]
        print(f"[{name}] Sample prices: " +
              ", ".join([f"{s}=₹{p}" for s, p in sample_ok]))

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify watchlist symbols on Yahoo Finance")
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of symbols to spot-check per list (default: 20)")
    parser.add_argument("--all", action="store_true",
                        help="Check ALL symbols (slow, ~10 min for 500 stocks)")
    parser.add_argument("--list", choices=["NIFTY500", "NIFTY250", "FNO"],
                        help="Check only one specific watchlist")
    args = parser.parse_args()

    sample_n = None if args.all else args.sample

    print("=" * 60)
    print("Watchlist Verifier")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode : {'ALL symbols' if args.all else f'Random sample of {sample_n}'}")
    print("=" * 60)

    lists_to_check = (
        {args.list: WATCHLISTS[args.list]} if args.list else WATCHLISTS
    )

    all_failed = {}
    for name, filename in lists_to_check.items():
        filepath = os.path.join(INPUT_DIR, filename)
        result = verify_watchlist(name, filepath, sample_n)
        if result and result["failed"]:
            all_failed[name] = result["failed"]

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if all_failed:
        print("Symbols that need attention (add to SYMBOL_CORRECTIONS or EXCLUDE list):")
        for name, syms in all_failed.items():
            print(f"  [{name}]: {', '.join(syms)}")
        print("\nRe-run build_watchlists.py after adding corrections.")
    else:
        print("✅ All checked symbols are working fine on Yahoo Finance.")
    print("=" * 60)


if __name__ == "__main__":
    main()
