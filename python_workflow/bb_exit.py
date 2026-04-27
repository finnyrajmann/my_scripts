"""
BB Exit Monitor
================
Reads positions_bb.csv and checks each open position daily.

Exit signal : Price touches or crosses above upper BB
Stop loss   : Price falls 10% below entry price

positions_bb.csv format:
    Symbol,EntryDate,EntryPrice,Quantity,TrackType
    NMDC,2026-03-23,75.07,133,Paper
    POWERGRID,2026-03-23,299.40,26,Real
"""

import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
POSITIONS_FILE = "positions_bb.csv"
SLEEP_SECONDS  = 0.3

BB_PERIOD      = 20
BB_STD         = 2
STOP_LOSS_PCT  = 10.0


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def nse(symbol):
    return f"{symbol.upper().strip()}.NS"


def load_positions(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found.\n"
            f"Create it with columns: Symbol,EntryDate,EntryPrice,Quantity,TrackType"
        )
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Symbol']     = df['Symbol'].str.strip()
    df['EntryPrice'] = df['EntryPrice'].astype(float)
    df['Quantity']   = df['Quantity'].astype(int)
    if 'TrackType' not in df.columns:
        df['TrackType'] = 'Real'
    df['TrackType'] = df['TrackType'].str.strip()
    print(f"Loaded {len(df)} open positions from {filepath}")
    real  = (df['TrackType'] == 'Real').sum()
    paper = (df['TrackType'] == 'Paper').sum()
    print(f"  Real  : {real}")
    print(f"  Paper : {paper}")
    return df


# ─────────────────────────────────────────────
# CALCULATE INDICATORS
# ─────────────────────────────────────────────
def get_indicators(symbol):
    try:
        ticker = yf.Ticker(nse(symbol))
        df     = ticker.history(period='3mo', interval='1d')

        if df.empty or len(df) < BB_PERIOD + 2:
            return None

        close = df['Close']

        df['BB_Mid']   = close.rolling(BB_PERIOD).mean()
        df['BB_Std']   = close.rolling(BB_PERIOD).std()
        df['BB_Upper'] = df['BB_Mid'] + BB_STD * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - BB_STD * df['BB_Std']

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        price    = round(float(latest['Close']), 2)
        bb_upper = round(float(latest['BB_Upper']), 2)
        bb_mid   = round(float(latest['BB_Mid']), 2)
        bb_lower = round(float(latest['BB_Lower']), 2)
        change   = round((price - float(prev['Close'])) / float(prev['Close']) * 100, 2)

        return {
            'price':    price,
            'change':   change,
            'bb_upper': bb_upper,
            'bb_mid':   bb_mid,
            'bb_lower': bb_lower,
        }

    except Exception:
        return None


# ─────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────
def print_report(results):
    print(f"\n{'═'*90}")
    print(f"  BB EXIT MONITOR — {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    print(f"{'═'*90}")

    exits = [r for r in results if r['ExitType'] is not None]
    holds = [r for r in results if r['ExitType'] is None]

    # ── EXIT signals ──
    if exits:
        print(f"\n  ⚠  EXIT SIGNALS ({len(exits)} positions)")
        print(f"  {'-'*85}")
        for r in exits:
            pnl_flag  = '🟢' if r['PnL'] >= 0 else '🔴'
            track_tag = '📋 PAPER' if r['TrackType'] == 'Paper' else '💰 REAL'
            print(f"\n  {pnl_flag} {track_tag} | {r['Symbol']:<12} "
                  f"Entry: ₹{r['EntryPrice']:<10} "
                  f"Current: ₹{r['Price']:<10} "
                  f"P&L: {r['PnL%']:+.2f}%  (₹{r['PnL']:+.0f})")
            print(f"     Exit Type : {r['ExitType']}")
            print(f"     Reason    : {r['ExitReason']}")
            print(f"     BB Lower: ₹{r['BBLower']:.2f}  "
                  f"BB Mid: ₹{r['BBMid']:.2f}  "
                  f"BB Upper: ₹{r['BBUpper']:.2f}  "
                  f"Stop: ₹{r['Stop']:.2f}  "
                  f"Days Held: {r['DaysHeld']}")

    # ── HOLD positions ──
    if holds:
        print(f"\n  ✅ HOLD ({len(holds)} positions)")
        print(f"  {'TYPE':<8} {'SYMBOL':<12} {'ENTRY':>8} {'PRICE':>8} "
              f"{'CHG%':>7} {'P&L%':>7} {'P&L₹':>9} "
              f"{'BB LOW':>8} {'BB MID':>8} {'BB UP':>8} "
              f"{'STOP':>8} {'DAYS':>5}")
        print(f"  {'-'*105}")
        for r in holds:
            pnl_flag  = '🟢' if r['PnL'] >= 0 else '🔴'
            track_tag = '📋' if r['TrackType'] == 'Paper' else '💰'
            print(
                f"  {pnl_flag} {track_tag} {r['TrackType']:<6} "
                f"{r['Symbol']:<12} "
                f"₹{r['EntryPrice']:>8.2f} "
                f"₹{r['Price']:>8.2f} "
                f"{r['Change%']:>7.2f} "
                f"{r['PnL%']:>+7.2f}% "
                f"₹{r['PnL']:>+9.0f} "
                f"₹{r['BBLower']:>8.2f} "
                f"₹{r['BBMid']:>8.2f} "
                f"₹{r['BBUpper']:>8.2f} "
                f"₹{r['Stop']:>8.2f} "
                f"{r['DaysHeld']:>5}"
            )

    # ── Portfolio summary ──
    real_pnl  = sum(r['PnL'] for r in results if r['TrackType'] == 'Real')
    paper_pnl = sum(r['PnL'] for r in results if r['TrackType'] == 'Paper')
    total_pnl = sum(r['PnL'] for r in results)

    print(f"\n{'═'*90}")
    flag = '🟢' if total_pnl >= 0 else '🔴'
    print(f"  💰 Real P&L          : ₹{real_pnl:+.0f}")
    print(f"  📋 Paper P&L         : ₹{paper_pnl:+.0f}")
    print(f"  {flag} Total Portfolio P&L : ₹{total_pnl:+.0f}")
    print(f"  Positions monitored  : {len(results)}")
    print(f"  Exit signals         : {len(exits)}")
    print(f"  Holding              : {len(holds)}")
    print(f"{'═'*90}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    positions  = load_positions(POSITIONS_FILE)
    total      = len(positions)
    results    = []

    print(f"\nBB Exit Monitor")
    print(f"  Exit (profit) : Price at or above BB Upper")
    print(f"  Exit (stop)   : Price {STOP_LOSS_PCT}% below entry")
    print(f"\nChecking {total} positions...\n")

    for _, pos in positions.iterrows():
        symbol      = pos['Symbol']
        entry_price = pos['EntryPrice']
        quantity    = pos['Quantity']
        entry_date  = pd.to_datetime(pos['EntryDate'])
        days_held   = (datetime.now() - entry_date).days
        track_type  = pos['TrackType']
        stop_price  = round(entry_price * (1 - STOP_LOSS_PCT / 100), 2)

        print(f"  Checking {symbol}...", end='\r')

        ind = get_indicators(symbol)

        if ind is None:
            print(f"  {symbol}: Could not fetch data — skipping")
            continue

        pnl_per_share = ind['price'] - entry_price
        pnl_total     = round(pnl_per_share * quantity, 2)
        pnl_pct       = round((pnl_per_share / entry_price) * 100, 2)

        exit_type   = None
        exit_reason = None

        if ind['price'] >= ind['bb_upper']:
            exit_type   = 'PROFIT'
            exit_reason = f"Price at BB Upper (₹{ind['bb_upper']:.2f})"
        elif ind['price'] <= stop_price:
            exit_type   = 'STOP'
            exit_reason = f"Stop loss hit (₹{stop_price:.2f})"

        results.append({
            'Symbol':     symbol,
            'TrackType':  track_type,
            'EntryPrice': entry_price,
            'Quantity':   quantity,
            'Price':      ind['price'],
            'Change%':    ind['change'],
            'PnL':        pnl_total,
            'PnL%':       pnl_pct,
            'BBLower':    ind['bb_lower'],
            'BBMid':      ind['bb_mid'],
            'BBUpper':    ind['bb_upper'],
            'Stop':       stop_price,
            'DaysHeld':   days_held,
            'ExitType':   exit_type,
            'ExitReason': exit_reason,
        })

        time.sleep(SLEEP_SECONDS)

    print_report(results)
