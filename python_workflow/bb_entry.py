"""
BB Entry Scanner
=================
Entry signal: Price touches or crosses below lower Bollinger Band
Additional filter: Price above EMA200 (avoid downtrending stocks)

Run every morning before market opens (9:00-9:15 AM IST)

Output:
    bb_buy_today.csv
    bb_buy_today.txt
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
OUTPUT_CSV    = "bb_buy_today.csv"
OUTPUT_TXT    = "bb_buy_today.txt"
SLEEP_SECONDS = 0.3

BB_PERIOD     = 20
BB_STD        = 2
EMA_SLOW      = 200
STOP_LOSS_PCT = 10.0


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def nse(symbol):
    return f"{symbol.upper().strip()}.NS"


def load_watchlist(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"'{filepath}' not found.")
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

        if df.empty or len(df) < EMA_SLOW + 5:
            return None

        close = df['Close']

        # Bollinger Bands
        df['BB_Mid']   = close.rolling(BB_PERIOD).mean()
        df['BB_Std']   = close.rolling(BB_PERIOD).std()
        df['BB_Upper'] = df['BB_Mid'] + BB_STD * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - BB_STD * df['BB_Std']

        # EMA200
        df['EMA200'] = close.ewm(span=EMA_SLOW, adjust=False).mean()

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        price    = round(float(latest['Close']), 2)
        bb_upper = round(float(latest['BB_Upper']), 2)
        bb_mid   = round(float(latest['BB_Mid']), 2)
        bb_lower = round(float(latest['BB_Lower']), 2)
        ema200   = round(float(latest['EMA200']), 2)
        change   = round((price - float(prev['Close'])) / float(prev['Close']) * 100, 2)
        stop     = round(price * (1 - STOP_LOSS_PCT / 100), 2)
        above200 = price > ema200

        return {
            'price':    price,
            'change':   change,
            'bb_upper': bb_upper,
            'bb_mid':   bb_mid,
            'bb_lower': bb_lower,
            'ema200':   ema200,
            'above200': above200,
            'stop':     stop,
        }

    except Exception:
        return None


# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────
def print_summary(df):
    print(f"\n{'═'*90}")
    print(f"  BB BUY ZONE — {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    print(f"{'═'*90}")

    if df.empty:
        print("  No stocks at lower BB today.")
        print("  Market may be elevated — check again tomorrow.")
        print(f"{'═'*90}\n")
        return

    banking     = df[df['IsBanking'] == True]
    non_banking = df[df['IsBanking'] == False]

    for label, subset in [('NON-FINANCIAL', non_banking), ('FINANCIAL/BANKING', banking)]:
        if subset.empty:
            continue
        print(f"\n  ── {label} ──")
        print(f"  {'SYMBOL':<12} {'PRICE':>8} {'CHG%':>7} {'BB LOW':>8} "
              f"{'BB MID':>8} {'BB UP':>8} {'EMA200':>8} "
              f"{'STOP':>8} {'>EMA200':>8} {'INDUSTRY'}")
        print(f"  {'-'*95}")
        for _, row in subset.iterrows():
            trend = '✅' if row['>EMA200'] else '❌'
            print(
                f"  {row['Symbol']:<12} "
                f"{row['Price']:>8.2f} "
                f"{row['Change%']:>7.2f} "
                f"{row['BB Lower']:>8.2f} "
                f"{row['BB Mid']:>8.2f} "
                f"{row['BB Upper']:>8.2f} "
                f"{row['EMA200']:>8.2f} "
                f"{row['Stop']:>8.2f} "
                f"{trend:>8} "
                f"  {row['Industry']}"
            )

    print(f"\n{'═'*90}")
    print(f"  Total : {len(df)} stocks at lower BB")
    print(f"  Above EMA200 (uptrend) : {df['>EMA200'].sum()}")
    print(f"  Below EMA200 (caution) : {(~df['>EMA200']).sum()}")
    print(f"{'═'*90}\n")


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

    print(f"\nBB Entry Scanner")
    print(f"  Entry    : Price at or below lower BB")
    print(f"  Filter   : Price above EMA200 flagged as uptrend")
    print(f"  Stop loss: {STOP_LOSS_PCT}% below entry")
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

        # Entry condition — price at or below lower BB
        if ind['price'] <= ind['bb_lower']:
            results.append({
                'Symbol':    symbol,
                'Industry':  industry,
                'IsBanking': is_banking,
                'Price':     ind['price'],
                'Change%':   ind['change'],
                'BB Lower':  ind['bb_lower'],
                'BB Mid':    ind['bb_mid'],
                'BB Upper':  ind['bb_upper'],
                'EMA200':    ind['ema200'],
                '>EMA200':   ind['above200'],
                'Stop':      ind['stop'],
            })

        time.sleep(SLEEP_SECONDS)

    print('\n')

    final_df = pd.DataFrame(results) if results else pd.DataFrame()

    print_summary(final_df)

    if not final_df.empty:
        save_results(final_df)
    else:
        # FIX: Do not write an empty CSV — archive_and_update.py will skip
        # a missing file cleanly. Writing 0-byte CSV causes EmptyDataError.
        open(OUTPUT_TXT, 'w').close()
        print("No stocks at lower BB today. bb_buy_today.csv not written.")

    elapsed_total = int(time.time() - start_time)
    print(f"Completed in {elapsed_total} seconds")

# At the end of bb_entry.py, when no signals found:
open(OUTPUT_CSV, 'w').close()  # truncate to empty
