"""
NSE Entry Scanner — Stage 2
============================
Reads watchlist.csv (output of technical_watchlist.py) and checks
each stock for a good entry zone using daily chart indicators.

STAGE 2 — Entry Zone Confirmation:
    - RSI between 40-55  : cooled off, not overbought, not in freefall
    - Price within 3% of 20 EMA : pulled back to support
    - Price at or below BB middle line : room to run upward
    - RVOL > 1.0 : volume confirming the move (6 month average)

Run every morning before market opens (9:00-9:15 AM IST).

Output:
    buy_today.csv  — full signal data
    buy_today.txt  — clean symbol list
"""

import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_FILE    = "watchlist.csv"
OUTPUT_CSV    = "buy_today.csv"
OUTPUT_TXT    = "buy_today.txt"
SLEEP_SECONDS = 0.3

# RSI
RSI_MIN       = 40
RSI_MAX       = 55
RSI_PERIOD    = 14

# EMA
EMA_PERIOD    = 20
EMA_DIST_MAX  = 3.0      # max % price can be away from 20 EMA

# Bollinger Bands
BB_PERIOD     = 20
BB_STD        = 2

# Volume
VOLUME_PERIOD = 125      # 6 month average
RVOL_MIN      = 0.75      # minimum relative volume for entry confirmation


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def nse(symbol):
    return f"{symbol.upper().strip()}.NS"


def load_watchlist(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found.\n"
            f"Run technical_watchlist.py first to generate it."
        )
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} stocks from {filepath}")
    return df


# ─────────────────────────────────────────────
# CALCULATE INDICATORS
# ─────────────────────────────────────────────
def get_indicators(symbol):
    try:
        ticker = yf.Ticker(nse(symbol))
        df     = ticker.history(period='1y', interval='1d')

        if df.empty or len(df) < BB_PERIOD + RSI_PERIOD + 5:
            return None

        close  = df['Close']
        volume = df['Volume']

        # RSI
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).ewm(span=RSI_PERIOD, adjust=False).mean()
        loss  = (-delta.where(delta < 0, 0)).ewm(span=RSI_PERIOD, adjust=False).mean()
        rs    = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 20 EMA
        df['EMA20'] = close.ewm(span=EMA_PERIOD, adjust=False).mean()

        # Bollinger Bands
        df['BB_Mid']   = close.rolling(BB_PERIOD).mean()
        df['BB_Std']   = close.rolling(BB_PERIOD).std()
        df['BB_Upper'] = df['BB_Mid'] + BB_STD * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - BB_STD * df['BB_Std']

        # RVOL — 6 month average (125 trading days)
        df['AvgVol'] = volume.rolling(VOLUME_PERIOD).mean()
        df['RVOL']   = volume / df['AvgVol']

        latest   = df.iloc[-1]
        prev     = df.iloc[-2]

        price    = round(float(latest['Close']), 2)
        ema20    = round(float(latest['EMA20']), 2)
        rsi      = round(float(latest['RSI']), 2)
        bb_mid   = round(float(latest['BB_Mid']), 2)
        bb_up    = round(float(latest['BB_Upper']), 2)
        bb_lo    = round(float(latest['BB_Lower']), 2)
        ema_dist = round(abs(price - ema20) / ema20 * 100, 2)
        change   = round((price - float(prev['Close'])) / float(prev['Close']) * 100, 2)
        rvol     = round(float(latest['RVOL']), 2)

        return {
            'price':    price,
            'change':   change,
            'rsi':      rsi,
            'ema20':    ema20,
            'ema_dist': ema_dist,
            'bb_upper': bb_up,
            'bb_mid':   bb_mid,
            'bb_lower': bb_lo,
            'rvol':     rvol,
        }

    except Exception:
        return None


# ─────────────────────────────────────────────
# APPLY ENTRY ZONE FILTERS
# ─────────────────────────────────────────────
def check_entry_zone(ind):
    rsi_pass  = RSI_MIN <= ind['rsi'] <= RSI_MAX
    ema_pass  = ind['ema_dist'] <= EMA_DIST_MAX
    bb_pass   = ind['price'] <= ind['bb_mid']
    rvol_pass = ind['rvol'] >= RVOL_MIN

    return (rsi_pass and ema_pass and bb_pass and rvol_pass), {
        'RSI Pass':  rsi_pass,
        'EMA Pass':  ema_pass,
        'BB Pass':   bb_pass,
        'RVOL Pass': rvol_pass,
    }


# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────
def print_summary(df):
    print(f"\n{'═'*85}")
    print(f"  BUY ZONE STOCKS — {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    print(f"{'═'*85}")

    if df.empty:
        print("  No stocks in entry zone today.")
        print("  Watchlist is healthy — waiting for the right moment.")
        print(f"{'═'*85}\n")
        return

    # Separate banking and non-banking
    banking     = df[df['IsBanking'] == True]
    non_banking = df[df['IsBanking'] == False]

    for label, subset in [('NON-FINANCIAL', non_banking), ('FINANCIAL/BANKING', banking)]:
        if subset.empty:
            continue
        print(f"\n  ── {label} ──")
        print(f"  {'SYMBOL':<12} {'PRICE':>8} {'CHG%':>7} {'RSI':>6} "
              f"{'EMA20':>8} {'DIST%':>7} {'BB MID':>8} {'BB LOW':>8} "
              f"{'RVOL':>6} {'INDUSTRY'}")
        print(f"  {'-'*90}")
        for _, row in subset.iterrows():
            print(
                f"  {row['Symbol']:<12} "
                f"{row['Price']:>8.2f} "
                f"{row['Change %']:>7.2f} "
                f"{row['RSI']:>6.1f} "
                f"{row['EMA20']:>8.2f} "
                f"{row['EMA Dist %']:>7.2f} "
                f"{row['BB Middle']:>8.2f} "
                f"{row['BB Lower']:>8.2f} "
                f"{row['RVOL']:>6.2f} "
                f"  {row['Industry']}"
            )

    print(f"\n{'═'*85}")
    print(f"  Total in buy zone : {len(df)} "
          f"({len(non_banking)} non-financial, {len(banking)} financial)")
    print(f"{'═'*85}\n")


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(df):
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} stocks to {OUTPUT_CSV}")

    with open(OUTPUT_TXT, 'w') as f:
        for sym in df['Symbol']:
            f.write(sym + '\n')
    print(f"Saved symbol list to {OUTPUT_TXT}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    watchlist  = load_watchlist(INPUT_FILE)
    total      = len(watchlist)

    print(f"\nEntry zone filters (daily chart):")
    print(f"  RSI        : {RSI_MIN} – {RSI_MAX}")
    print(f"  EMA Dist   : price within {EMA_DIST_MAX}% of 20 EMA")
    print(f"  BB Position: price at or below middle Bollinger Band")
    print(f"  RVOL       : >= {RVOL_MIN} (6 month average)")
    print(f"\nScanning {total} stocks...\n")

    results    = []
    start_time = time.time()

    for i, row in watchlist.iterrows():
        symbol     = row['Symbol']
        industry   = row['Industry']
        is_banking = bool(row['IsBanking'])

        ind = get_indicators(symbol)

        elapsed = time.time() - start_time
        rate    = (i + 1) / elapsed if elapsed > 0 else 1
        eta     = int((total - i - 1) / rate)

        print(
            f"[{i+1:>4}/{total}] "
            f"In zone: {len(results)} | "
            f"ETA: {eta}s | {symbol:<15}",
            end='\r'
        )

        if ind is None:
            time.sleep(SLEEP_SECONDS)
            continue

        passed, details = check_entry_zone(ind)

        if passed:
            results.append({
                'Symbol':      symbol,
                'Industry':    industry,
                'IsBanking':   is_banking,
                'Price':       ind['price'],
                'Change %':    ind['change'],
                'RSI':         ind['rsi'],
                'EMA20':       ind['ema20'],
                'EMA Dist %':  ind['ema_dist'],
                'BB Upper':    ind['bb_upper'],
                'BB Middle':   ind['bb_mid'],
                'BB Lower':    ind['bb_lower'],
                'RVOL':        ind['rvol'],
                'RSI Pass':    details['RSI Pass'],
                'EMA Pass':    details['EMA Pass'],
                'BB Pass':     details['BB Pass'],
                'RVOL Pass':   details['RVOL Pass'],
            })

        time.sleep(SLEEP_SECONDS)

    print('\n')

    # Sort by RSI closest to 50
    final_df = pd.DataFrame(results) if results else pd.DataFrame()
    if not final_df.empty:
        final_df['RSI Dist'] = abs(final_df['RSI'] - 50)
        final_df = final_df.sort_values(
            ['IsBanking', 'RSI Dist']
        ).drop(columns=['RSI Dist'])

    print_summary(final_df)

    if not final_df.empty:
        save_results(final_df)
    else:
        open(OUTPUT_TXT, 'w').close()
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False)
        print(f"No stocks in entry zone today.")
        print(f"If consistently empty, consider relaxing:")
        print(f"  RSI_MAX      → 60  (currently {RSI_MAX})")
        print(f"  EMA_DIST_MAX → 5.0 (currently {EMA_DIST_MAX})")
        print(f"  RVOL_MIN     → 0.8 (currently {RVOL_MIN})")

    elapsed_total = int(time.time() - start_time)
    print(f"Completed in {elapsed_total} seconds")
