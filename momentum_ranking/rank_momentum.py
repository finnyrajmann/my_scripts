#!/usr/bin/env python3
"""
Momentum Ranking Script
=======================
Ranks stocks in a watchlist CSV by momentum score.

Formula:
    Score = (1M return + 3M return + 6M return) / 1M volatility

    - 1M  = ~21 trading days
    - 3M  = ~63 trading days
    - 6M  = ~126 trading days
    - 1M volatility = std dev of daily returns over last 21 trading days

Higher score = stronger, smoother recent momentum.

Usage:
    python3 rank_momentum.py                              # uses watchlist_nifty500.csv
    python3 rank_momentum.py watchlist_fno.csv            # any other watchlist
    python3 rank_momentum.py watchlist_nifty250.csv --top 30

Output:
    <input_name>_ranked.csv  — full ranked list
    Prints top N to terminal (default 20)
"""

import pandas as pd
import yfinance as yf
import numpy as np
import os
import sys
import time
import argparse
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

# Trading days approximations
DAYS_1M  = 21
DAYS_3M  = 63
DAYS_6M  = 126

# Need extra buffer for rolling calculations
FETCH_PERIOD = "9mo"

BATCH_SIZE  = 20
BATCH_DELAY = 2   # seconds between batches

# ── Data Fetch ────────────────────────────────────────────────────────────────

def fetch_close_prices(symbols: list[str]) -> pd.DataFrame:
    """Fetch daily close prices for all symbols in batches."""
    all_data = []
    total = len(symbols)
    print(f"Fetching price data for {total} symbols...")

    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_str = " ".join(batch)

        try:
            raw = yf.download(
                batch_str,
                period=FETCH_PERIOD,
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=True,
            )

            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})

            all_data.append(close)

        except Exception as e:
            print(f"  Batch {i//BATCH_SIZE + 1} error: {e}")

        done = min(i + BATCH_SIZE, total)
        print(f"  Progress: {done}/{total}", end="\r")

        if i + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)

    print()  # newline

    if not all_data:
        print("ERROR: No data fetched.")
        sys.exit(1)

    df = pd.concat(all_data, axis=1)
    df = df.sort_index()
    return df


# ── Momentum Calculations ─────────────────────────────────────────────────────

def calc_return(close: pd.Series, days: int) -> float | None:
    """Percentage return over last N trading days."""
    data = close.dropna()
    if len(data) < days + 1:
        return None
    start = data.iloc[-(days + 1)]
    end   = data.iloc[-1]
    if start <= 0:
        return None
    return (end - start) / start * 100


def calc_volatility(close: pd.Series, days: int) -> float | None:
    """Annualised volatility of daily returns over last N trading days."""
    data = close.dropna()
    if len(data) < days + 1:
        return None
    recent = data.iloc[-days:]
    daily_returns = recent.pct_change().dropna()
    if len(daily_returns) < 5:
        return None
    vol = daily_returns.std() * np.sqrt(252)  # annualised
    return vol if vol > 0 else None


def score_symbol(close: pd.Series, symbol: str) -> dict | None:
    """Calculate momentum score for a single symbol."""
    r1m = calc_return(close, DAYS_1M)
    r3m = calc_return(close, DAYS_3M)
    r6m = calc_return(close, DAYS_6M)
    vol = calc_volatility(close, DAYS_1M)

    # Need all components
    if any(v is None for v in [r1m, r3m, r6m, vol]):
        return None

    score = (r1m + r3m + r6m) / vol
    last_price = close.dropna().iloc[-1]

    return {
        "Symbol":   symbol.replace(".NS", ""),
        "Price":    round(last_price, 2),
        "1M%":      round(r1m, 2),
        "3M%":      round(r3m, 2),
        "6M%":      round(r6m, 2),
        "1M_Vol":   round(vol * 100, 2),   # as percentage
        "Score":    round(score, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank watchlist by momentum score")
    parser.add_argument(
        "watchlist",
        nargs="?",
        default="watchlist_nifty500.csv",
        help="Watchlist CSV file (default: watchlist_nifty500.csv)"
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top stocks to print (default: 20)"
    )
    parser.add_argument(
        "--bottom", type=int, default=0,
        help="Also print N bottom ranked stocks"
    )
    args = parser.parse_args()

    # Resolve file path
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    watchlist_path = args.watchlist if os.path.isabs(args.watchlist) \
                     else os.path.join(script_dir, args.watchlist)

    if not os.path.exists(watchlist_path):
        print(f"ERROR: File not found: {watchlist_path}")
        sys.exit(1)

    # Load watchlist
    df_wl = pd.read_csv(watchlist_path)
    symbols = df_wl["Symbol"].tolist()
    print("=" * 60)
    print("Momentum Ranking")
    print(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Watchlist : {os.path.basename(watchlist_path)} ({len(symbols)} symbols)")
    print(f"Formula   : (1M% + 3M% + 6M%) / 1M_Volatility")
    print("=" * 60)

    # Fetch prices
    close_df = fetch_close_prices(symbols)

    # Score each symbol
    print("Calculating momentum scores...")
    results = []
    skipped = 0

    for sym in symbols:
        if sym in close_df.columns:
            row = score_symbol(close_df[sym], sym)
            if row:
                results.append(row)
            else:
                skipped += 1
        else:
            skipped += 1

    if not results:
        print("ERROR: No scores calculated.")
        sys.exit(1)

    # Build ranked DataFrame
    df_ranked = pd.DataFrame(results)
    df_ranked = df_ranked.sort_values("Score", ascending=False).reset_index(drop=True)
    df_ranked.insert(0, "Rank", df_ranked.index + 1)

    print(f"Scored: {len(df_ranked)} symbols  |  Skipped (insufficient data): {skipped}")

    # ── Save output ───────────────────────────────────────────────────────────
    base     = os.path.splitext(os.path.basename(watchlist_path))[0]
    out_name = f"{base}_ranked.csv"
    out_path = os.path.join(script_dir, out_name)
    df_ranked.to_csv(out_path, index=False)
    print(f"Saved  : {out_name}")

    # ── Print top N ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"TOP {args.top} by Momentum Score")
    print(f"{'─'*60}")
    top_df = df_ranked.head(args.top)
    print(top_df.to_string(index=False))

    # ── Print bottom N if requested ───────────────────────────────────────────
    if args.bottom > 0:
        print(f"\n{'─'*60}")
        print(f"BOTTOM {args.bottom} (weakest momentum)")
        print(f"{'─'*60}")
        bottom_df = df_ranked.tail(args.bottom)
        print(bottom_df.to_string(index=False))

    # ── Quick stats ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("UNIVERSE STATS")
    print(f"{'─'*60}")
    print(f"  Avg Score  : {df_ranked['Score'].mean():.4f}")
    print(f"  Avg 1M%    : {df_ranked['1M%'].mean():.2f}%")
    print(f"  Avg 3M%    : {df_ranked['3M%'].mean():.2f}%")
    print(f"  Avg 6M%    : {df_ranked['6M%'].mean():.2f}%")
    print(f"  +ve 1M%    : {(df_ranked['1M%'] > 0).sum()} / {len(df_ranked)} stocks")
    print(f"  +ve 6M%    : {(df_ranked['6M%'] > 0).sum()} / {len(df_ranked)} stocks")
    print("=" * 60)


if __name__ == "__main__":
    main()
