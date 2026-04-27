"""
NSE Technical Watchlist Builder — Stage 1
==========================================
Reads fundamental_results.csv (passed stocks only) and applies
two technical stages:

STAGE 1A — Multi-Timeframe 20 EMA Slope:
    20 EMA sloping up on at least 4 of 5 timeframes:
    1hr, 3hr, 1day, 1week, 1month

STAGE 1B — 20/200 EMA Crossover (Daily):
    20 EMA crossed above 200 EMA within the last 30 days

Output:
    watchlist.csv  — stocks passing both stages with full data
    watchlist.txt  — clean symbol list (fed into entry_scanner.py)

Run once a week to refresh your watchlist.
"""

import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_FILE      = "fundamental_results.csv"
OUTPUT_CSV      = "watchlist.csv"
OUTPUT_TXT      = "watchlist.txt"
CHECKPOINT_FILE = "technical_checkpoint.csv"
BATCH_SIZE      = 20
SLEEP_SECONDS   = 0.5

# Stage 1A
EMA_PERIOD      = 20
MIN_TF_PASS     = 4      # minimum timeframes that must show upward slope

# Stage 1B
CROSSOVER_DAYS  = 30     # how recent the 20/200 crossover must be

# Financial sector keywords for IsBanking flag
BANKING_SECTORS = {'Financial Services'}


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def nse(symbol):
    return f"{symbol.upper().strip()}.NS"


def load_passed_stocks(filepath):
    df = pd.read_csv(filepath)
    passed = df[df['PassFilter'] == True][['Symbol', 'Industry']].copy()
    passed['IsBanking'] = passed['Industry'].isin(BANKING_SECTORS)
    print(f"Loaded {len(passed)} passed stocks from {filepath}")
    print(f"  Banking/Financial : {passed['IsBanking'].sum()}")
    print(f"  Non-financial     : {(~passed['IsBanking']).sum()}")
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
def load_checkpoint(all_symbols):
    if not os.path.exists(CHECKPOINT_FILE):
        return [], list(all_symbols)

    df = pd.read_csv(CHECKPOINT_FILE)
    if 'Processed' not in df.columns:
        os.remove(CHECKPOINT_FILE)
        return [], list(all_symbols)

    processed = set(df['Processed'].dropna().tolist())
    passed_df = df[df['Symbol'].notna() & (df['Symbol'] != '')].copy()
    passed_df = passed_df.drop(columns=['Processed'], errors='ignore')
    results   = passed_df.to_dict('records')
    remaining = [s for s in all_symbols if s not in processed]

    print(f"\nResuming from checkpoint:")
    print(f"  Already processed : {len(processed)}")
    print(f"  Passed so far     : {len(results)}")
    print(f"  Remaining         : {len(remaining)}")

    return results, remaining


def save_checkpoint(results, processed):
    passed_symbols = {r['Symbol'] for r in results}
    failed         = [s for s in processed if s not in passed_symbols]
    passed_df      = pd.DataFrame(results) if results else pd.DataFrame()
    if not passed_df.empty:
        passed_df['Processed'] = passed_df['Symbol']
    failed_df = pd.DataFrame({'Processed': failed})
    combined  = pd.concat([passed_df, failed_df], ignore_index=True)
    combined.to_csv(CHECKPOINT_FILE, index=False)


# ─────────────────────────────────────────────
# STAGE 1A — MULTI-TIMEFRAME 20 EMA SLOPE
# ─────────────────────────────────────────────
def check_ema_slope(ticker):
    timeframes = {
        '1hr':  ('1h',  '60d'),
        '3hr':  ('1h',  '60d'),
        '1day': ('1d',  '6mo'),
        '1wk':  ('1wk', '2y'),
        '1mo':  ('1mo', '5y'),
    }
    results = {}

    for label, (interval, period) in timeframes.items():
        try:
            df = ticker.history(period=period, interval=interval)
            if df.empty or len(df) < EMA_PERIOD + 2:
                results[label] = False
                continue

            if label == '3hr':
                df = df.resample('3h').agg({
                    'Open': 'first', 'High': 'max',
                    'Low': 'min', 'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                if len(df) < EMA_PERIOD + 2:
                    results[label] = False
                    continue

            df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
            results[label] = bool(df['EMA'].iloc[-1] > df['EMA'].iloc[-2])

        except Exception:
            results[label] = False

    score = sum(results.values())
    return score, results


# ─────────────────────────────────────────────
# STAGE 1B — 20/200 EMA CROSSOVER (DAILY)
# ─────────────────────────────────────────────
def check_ema_crossover(ticker):
    try:
        df = ticker.history(period='1y', interval='1d')
        if df.empty or len(df) < 205:
            return False, None

        df['EMA20']  = df['Close'].ewm(span=20,  adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['above']  = df['EMA20'] > df['EMA200']
        df['cross']  = df['above'] & ~df['above'].shift(1).fillna(False)

        cutoff = pd.Timestamp(datetime.now() - timedelta(days=CROSSOVER_DAYS))
        if df.index.tz:
            cutoff = cutoff.tz_localize(df.index.tz)

        recent = df[(df.index >= cutoff) & (df['cross'])]
        if not recent.empty:
            return True, str(recent.index[-1].date())

        return False, None

    except Exception:
        return False, None


# ─────────────────────────────────────────────
# SCAN ONE STOCK
# ─────────────────────────────────────────────
def scan_stock(symbol, industry, is_banking):
    try:
        ticker = yf.Ticker(nse(symbol))

        # Stage 1A
        tf_score, tf_details = check_ema_slope(ticker)
        if tf_score < MIN_TF_PASS:
            return None

        # Stage 1B
        crossover, cross_date = check_ema_crossover(ticker)
        if not crossover:
            return None

        # Current price
        hist  = ticker.history(period='5d', interval='1d')
        price = round(float(hist['Close'].iloc[-1]), 2) if not hist.empty else None

        return {
            'Symbol':     symbol,
            'Industry':   industry,
            'IsBanking':  is_banking,
            'Price':      price,
            'TF Score':   f"{tf_score}/5",
            'Cross Date': cross_date,
            '1hr':        '✅' if tf_details.get('1hr')  else '❌',
            '3hr':        '✅' if tf_details.get('3hr')  else '❌',
            '1day':       '✅' if tf_details.get('1day') else '❌',
            '1wk':        '✅' if tf_details.get('1wk')  else '❌',
            '1mo':        '✅' if tf_details.get('1mo')  else '❌',
        }

    except Exception:
        return None


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(results):
    df = pd.DataFrame(results)

    # Sort: TF score descending, then cross date most recent first
    df['TF Int'] = df['TF Score'].str.split('/').str[0].astype(int)
    df = df.sort_values(
        ['TF Int', 'Cross Date'],
        ascending=[False, False]
    ).drop(columns=['TF Int'])

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} stocks to {OUTPUT_CSV}")

    # Save TXT
    with open(OUTPUT_TXT, 'w') as f:
        for sym in df['Symbol']:
            f.write(sym + '\n')
    print(f"Saved symbol list to {OUTPUT_TXT}")

    # Preview
    print(f"\nTop watchlist candidates:")
    print('-' * 75)
    preview_cols = ['Symbol', 'Industry', 'IsBanking', 'Price',
                    'TF Score', 'Cross Date', '1hr', '3hr', '1day', '1wk', '1mo']
    print(df[preview_cols].head(20).to_string(index=False))

    # Sector breakdown
    print(f"\nSector breakdown of watchlist:")
    print(df.groupby('Industry').size().sort_values(ascending=False).to_string())


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    stocks  = load_passed_stocks(INPUT_FILE)
    symbols = stocks['Symbol'].tolist()
    total   = len(symbols)

    print(f"\nFilters:")
    print(f"  Stage 1A : 20 EMA sloping up on >= {MIN_TF_PASS}/5 timeframes")
    print(f"  Stage 1B : 20/200 EMA crossover within last {CROSSOVER_DAYS} days")
    print(f"\nEstimated time: 15-25 minutes for {total} stocks")

    results, remaining = load_checkpoint(symbols)
    processed = [s for s in symbols if s not in set(remaining)]

    print(f"\nScanning {len(remaining)} stocks...\n")

    start_time = time.time()

    for i, symbol in enumerate(remaining, 1):
        row        = stocks[stocks['Symbol'] == symbol].iloc[0]
        industry   = row['Industry']
        is_banking = bool(row['IsBanking'])

        result = scan_stock(symbol, industry, is_banking)

        if result:
            results.append(result)

        processed.append(symbol)

        if i % BATCH_SIZE == 0:
            save_checkpoint(results, processed)

        elapsed = time.time() - start_time
        rate    = i / elapsed if elapsed > 0 else 1
        eta     = int((len(remaining) - i) / rate)
        print(
            f"[{i:>4}/{len(remaining)}] "
            f"Passed: {len(results)} | "
            f"ETA: {eta}s | {symbol:<15}",
            end='\r'
        )

        time.sleep(SLEEP_SECONDS)

    print('\n')

    if results:
        save_results(results)
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    else:
        print("No stocks passed technical filters today.")
        print("Market may be in a broad downtrend or consolidation phase.")
        print(f"Consider increasing CROSSOVER_DAYS beyond {CROSSOVER_DAYS}")

    elapsed_total = int(time.time() - start_time)
    print(f"\nCompleted in {elapsed_total} seconds")
