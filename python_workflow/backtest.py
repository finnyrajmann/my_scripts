"""
NSE Backtest — Bollinger Band Mean Reversion v5
=================================================
Simple BB mean reversion strategy:
- Entry  : Price touches or crosses below lower BB
- Exit   : Price touches or crosses above upper BB
- Stop   : Price falls 5% below entry price
- Force close at end of period

Capital : ₹50,000 total, ₹10,000 per trade, max 5 positions

Input  : watchlist.csv
Output : backtest_trades.csv
         backtest_summary.csv
         capital_curve.csv
"""

import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from math import floor


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_FILE        = "watchlist.csv"
TRADES_FILE       = "backtest_trades.csv"
SUMMARY_FILE      = "backtest_summary.csv"
CACHE_DIR         = "data"

START_DATE        = "2025-01-01"
END_DATE          = "2025-12-31"

# Capital management
STARTING_CAPITAL  = 100000
CAPITAL_PER_TRADE = 10000
MAX_POSITIONS     = 10

# Bollinger Band settings
BB_PERIOD         = 20
BB_STD            = 2

# Stop loss
STOP_LOSS_PCT     = 5.0   # 5% below entry price

# Minimum hold before exit triggers
MIN_HOLD_DAYS     = 1


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def nse(symbol):
    return f"{symbol.upper().strip()}.NS"


def load_watchlist(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} stocks from {filepath}")
    return df['Symbol'].tolist()


# ─────────────────────────────────────────────
# DATA CACHE
# ─────────────────────────────────────────────
def fetch_all_data(symbols):
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_data = {}

    print(f"\nFetching data for {len(symbols)} stocks...")

    fetch_start = str(
        (datetime.strptime(START_DATE, "%Y-%m-%d")
         - timedelta(days=365)).date()
    )

    for symbol in symbols:
        cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")
        use_cache  = False

        if os.path.exists(cache_file):
            age = datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(cache_file)
            )
            if age.total_seconds() < 86400:
                use_cache = True

        if use_cache:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"  {symbol}: loaded from cache ({len(df)} days)")
        else:
            try:
                ticker = yf.Ticker(nse(symbol))
                df     = ticker.history(start=fetch_start, end=END_DATE)
                if df.empty or len(df) < BB_PERIOD + 5:
                    print(f"  {symbol}: insufficient data — skipping")
                    continue
                df.index = df.index.tz_localize(None)
                df.to_csv(cache_file)
                print(f"  {symbol}: fetched {len(df)} days")
                time.sleep(0.3)
            except Exception as e:
                print(f"  {symbol}: error — {e}")
                continue

        all_data[symbol] = df

    print(f"Data ready for {len(all_data)} stocks")
    return all_data


# ─────────────────────────────────────────────
# CALCULATE BOLLINGER BANDS
# ─────────────────────────────────────────────
def calc_bb(df, date):
    try:
        sub = df[df.index <= date].copy()
        if len(sub) < 200 + 2:
            return None

        close = sub['Close']
        sub['BB_Mid']   = close.rolling(BB_PERIOD).mean()
        sub['BB_Std']   = close.rolling(BB_PERIOD).std()
        sub['BB_Upper'] = sub['BB_Mid'] + BB_STD * sub['BB_Std']
        sub['BB_Lower'] = sub['BB_Mid'] - BB_STD * sub['BB_Std']
        sub['EMA200']    = close.ewm(span=200, adjust=False).mean()

        latest = sub.iloc[-1]

        return {
            'price':             float(latest['Close']),
            'bb_upper':          float(latest['BB_Upper']),
            'bb_mid':            float(latest['BB_Mid']),
            'bb_lower':          float(latest['BB_Lower']),
            'price_above_ema200': float(latest['Close']) > float(latest['EMA200']),
        }

    except Exception:
        return None


# ─────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────
def run_backtest(symbols, all_data):
    start  = datetime.strptime(START_DATE, "%Y-%m-%d")
    end    = datetime.strptime(END_DATE,   "%Y-%m-%d")

    sample     = list(all_data.values())[0]
    trade_days = sample[
        (sample.index >= start) &
        (sample.index <= end)
    ].index.tolist()

    print(f"\nSimulating {len(trade_days)} trading days "
          f"({START_DATE} to {END_DATE})...")
    print(f"Starting capital : ₹{STARTING_CAPITAL:,}")
    print(f"Capital/trade    : ₹{CAPITAL_PER_TRADE:,}")
    print(f"Max positions    : {MAX_POSITIONS}")
    print(f"Entry            : Price touches lower BB")
    print(f"Exit             : Price touches upper BB")
    print(f"Stop loss        : {STOP_LOSS_PCT}% below entry")

    capital        = STARTING_CAPITAL
    open_positions = {}
    closed_trades  = []
    capital_curve  = []
    skipped_trades = 0

    for day in trade_days:
        day_str = day.strftime('%Y-%m-%d')

        # ── Check exits first ──
        for symbol in list(open_positions.keys()):
            if symbol not in all_data:
                continue

            bb = calc_bb(all_data[symbol], day)
            if bb is None:
                continue

            pos        = open_positions[symbol]
            days_held  = (day - pos['buy_date']).days
            stop_price = round(pos['buy_price'] * (1 - STOP_LOSS_PCT / 100), 2)

            exit_type   = None
            exit_reason = None

            # Profit exit — price touches upper BB
            if bb['price'] >= bb['bb_upper']:
                exit_type   = 'PROFIT'
                exit_reason = f"Price at BB Upper ({bb['bb_upper']:.2f})"

            # Stop loss exit
            elif bb['price'] <= stop_price:
                exit_type   = 'STOP'
                exit_reason = f"Stop loss hit ({stop_price:.2f})"

            if exit_type:
                sell_price = bb['price']
                qty        = pos['quantity']
                pnl        = round((sell_price - pos['buy_price']) * qty, 2)
                pnl_pct    = round(
                    (sell_price - pos['buy_price'])
                    / pos['buy_price'] * 100, 2
                )
                capital += pos['capital'] + pnl

                closed_trades.append({
                    'Symbol':      symbol,
                    'Buy Date':    pos['buy_date'].strftime('%Y-%m-%d'),
                    'Buy Price':   round(pos['buy_price'], 2),
                    'Stop Price':  stop_price,
                    'Sell Date':   day_str,
                    'Sell Price':  round(sell_price, 2),
                    'Quantity':    qty,
                    'Capital':     pos['capital'],
                    'PnL':         pnl,
                    'PnL%':        pnl_pct,
                    'Exit Type':   exit_type,
                    'Exit Reason': exit_reason,
                    'Days Held':   days_held,
                })

                del open_positions[symbol]

        # ── Check entries ──
        for symbol in symbols:
            if symbol not in all_data:
                continue
            if symbol in open_positions:
                continue
            if len(open_positions) >= MAX_POSITIONS:
                break
            if capital < CAPITAL_PER_TRADE:
                skipped_trades += 1
                continue

            bb = calc_bb(all_data[symbol], day)
            if bb is None:
                continue

            # Entry — price touches lower BB AND above EMA200 (uptrend only)
            if bb['price'] <= bb['bb_lower'] and bb['price_above_ema200']:
                qty = floor(CAPITAL_PER_TRADE / bb['price'])
                if qty > 0:
                    allocated = round(qty * bb['price'], 2)
                    capital  -= allocated

                    open_positions[symbol] = {
                        'buy_date':  day,
                        'buy_price': bb['price'],
                        'quantity':  qty,
                        'capital':   allocated,
                    }

        # ── Capital curve ──
        open_value = 0
        for symbol, pos in open_positions.items():
            bb = calc_bb(all_data[symbol], day)
            if bb:
                open_value += bb['price'] * pos['quantity']

        capital_curve.append({
            'Date':           day_str,
            'Cash':           round(capital, 2),
            'Open Positions': len(open_positions),
            'Open Value':     round(open_value, 2),
            'Total Value':    round(capital + open_value, 2),
        })

    # ── Force close remaining positions ──
    last_day = trade_days[-1]
    for symbol, pos in list(open_positions.items()):
        bb = calc_bb(all_data[symbol], last_day)
        if bb:
            sell_price = bb['price']
            qty        = pos['quantity']
            pnl        = round((sell_price - pos['buy_price']) * qty, 2)
            pnl_pct    = round(
                (sell_price - pos['buy_price'])
                / pos['buy_price'] * 100, 2
            )
            capital   += pos['capital'] + pnl
            days_held  = (last_day - pos['buy_date']).days

            closed_trades.append({
                'Symbol':      symbol,
                'Buy Date':    pos['buy_date'].strftime('%Y-%m-%d'),
                'Buy Price':   round(pos['buy_price'], 2),
                'Stop Price':  round(pos['buy_price'] * (1 - STOP_LOSS_PCT / 100), 2),
                'Sell Date':   last_day.strftime('%Y-%m-%d'),
                'Sell Price':  round(sell_price, 2),
                'Quantity':    qty,
                'Capital':     pos['capital'],
                'PnL':         pnl,
                'PnL%':        pnl_pct,
                'Exit Type':   'FORCE CLOSE',
                'Exit Reason': 'End of backtest period',
                'Days Held':   days_held,
            })

    return closed_trades, capital_curve, skipped_trades, capital


# ─────────────────────────────────────────────
# PRINT AND SAVE RESULTS
# ─────────────────────────────────────────────
def print_results(trades, capital_curve, skipped, final_capital):
    if not trades:
        print("\nNo trades were generated in this period.")
        return

    df = pd.DataFrame(trades)

    # ── Per trade output ──
    print(f"\n{'═'*110}")
    print(f"  BACKTEST RESULTS v5 (BB Mean Reversion) — {START_DATE} to {END_DATE}")
    print(f"{'═'*110}")
    print(f"\n  {'SYMBOL':<12} {'BUY DATE':<12} {'BUY':>8} {'STOP':>8} "
          f"{'SELL DATE':<12} {'SELL':>8} {'QTY':>5} "
          f"{'PnL₹':>8} {'PnL%':>7} {'DAYS':>5} "
          f"{'EXIT TYPE':<12} {'REASON'}")
    print(f"  {'-'*105}")

    for _, row in df.sort_values(['Symbol', 'Buy Date']).iterrows():
        flag = '🟢' if row['PnL'] >= 0 else '🔴'
        print(
            f"  {flag} {row['Symbol']:<12} "
            f"{row['Buy Date']:<12} "
            f"₹{row['Buy Price']:>8.2f} "
            f"₹{row['Stop Price']:>8.2f} "
            f"{row['Sell Date']:<12} "
            f"₹{row['Sell Price']:>8.2f} "
            f"{row['Quantity']:>5} "
            f"₹{row['PnL']:>8.2f} "
            f"{row['PnL%']:>+7.2f}% "
            f"{row['Days Held']:>5} "
            f"{row['Exit Type']:<12} "
            f"{row['Exit Reason']}"
        )

    # ── Per stock summary ──
    print(f"\n{'═'*110}")
    print(f"  PER STOCK SUMMARY")
    print(f"{'═'*110}")
    print(f"  {'SYMBOL':<12} {'TRADES':>7} {'WINS':>6} {'STOPS':>7} "
          f"{'FORCE':>7} {'WIN%':>6} {'TOTAL PnL':>10} "
          f"{'AVG PnL':>9} {'BEST':>9} {'WORST':>9} {'AVG DAYS':>9}")
    print(f"  {'-'*100}")

    stock_summary = []
    for symbol, grp in df.groupby('Symbol'):
        wins      = (grp['Exit Type'] == 'PROFIT').sum()
        stops     = (grp['Exit Type'] == 'STOP').sum()
        force     = (grp['Exit Type'] == 'FORCE CLOSE').sum()
        losses    = stops + force
        win_pct   = round(wins / len(grp) * 100, 1)
        total     = round(grp['PnL'].sum(), 2)
        avg       = round(grp['PnL'].mean(), 2)
        best      = round(grp['PnL'].max(), 2)
        worst     = round(grp['PnL'].min(), 2)
        avg_days  = round(grp['Days Held'].mean(), 1)

        stock_summary.append({
            'Symbol':    symbol,
            'Trades':    len(grp),
            'Wins':      wins,
            'Stops':     stops,
            'Force':     force,
            'Win%':      win_pct,
            'Total PnL': total,
            'Avg PnL':   avg,
            'Best':      best,
            'Worst':     worst,
            'Avg Days':  avg_days,
        })

        flag = '🟢' if total >= 0 else '🔴'
        print(
            f"  {flag} {symbol:<12} "
            f"{len(grp):>7} "
            f"{wins:>6} "
            f"{stops:>7} "
            f"{force:>7} "
            f"{win_pct:>6.1f}% "
            f"₹{total:>10.2f} "
            f"₹{avg:>9.2f} "
            f"₹{best:>9.2f} "
            f"₹{worst:>9.2f} "
            f"{avg_days:>9.1f}"
        )

    # ── Capital curve highlights ──
    cc_df     = pd.DataFrame(capital_curve)
    max_value = cc_df['Total Value'].max()
    min_value = cc_df['Total Value'].min()
    max_date  = cc_df.loc[cc_df['Total Value'].idxmax(), 'Date']
    min_date  = cc_df.loc[cc_df['Total Value'].idxmin(), 'Date']
    max_dd    = round(
        (cc_df['Total Value'].cummax() - cc_df['Total Value']).max(), 2
    )

    # ── Overall summary ──
    total_trades  = len(df)
    total_wins    = (df['Exit Type'] == 'PROFIT').sum()
    total_stops   = (df['Exit Type'] == 'STOP').sum()
    total_force   = (df['Exit Type'] == 'FORCE CLOSE').sum()
    overall_win   = round(total_wins / total_trades * 100, 1)
    overall_pnl   = round(df['PnL'].sum(), 2)
    avg_pnl       = round(df['PnL'].mean(), 2)
    avg_days      = round(df['Days Held'].mean(), 1)
    best_trade    = df.loc[df['PnL'].idxmax()]
    worst_trade   = df.loc[df['PnL'].idxmin()]
    total_return  = round(
        (final_capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2
    )

    flag = '🟢' if final_capital >= STARTING_CAPITAL else '🔴'

    print(f"\n{'═'*110}")
    print(f"  OVERALL SUMMARY")
    print(f"{'═'*110}")
    print(f"  Period             : {START_DATE} to {END_DATE}")
    print(f"  Starting capital   : ₹{STARTING_CAPITAL:,}")
    print(f"  {flag} Ending capital     : ₹{final_capital:,.2f}")
    print(f"  Total return       : {total_return:+.2f}%")
    print(f"  Max portfolio value: ₹{max_value:,.2f} on {max_date}")
    print(f"  Min portfolio value: ₹{min_value:,.2f} on {min_date}")
    print(f"  Max drawdown       : ₹{max_dd:,.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  Total trades       : {total_trades}")
    print(f"  Profit exits       : {total_wins} ({overall_win}%)")
    print(f"  Stop loss exits    : {total_stops}")
    print(f"  Force closes       : {total_force}")
    print(f"  Skipped (no cap)   : {skipped}")
    print(f"  Avg days held      : {avg_days}")
    print(f"  Avg P&L/trade      : ₹{avg_pnl:+.2f}")
    print(f"  Best trade         : {best_trade['Symbol']} "
          f"₹{best_trade['PnL']:+.2f} ({best_trade['PnL%']:+.2f}%)")
    print(f"  Worst trade        : {worst_trade['Symbol']} "
          f"₹{worst_trade['PnL']:+.2f} ({worst_trade['PnL%']:+.2f}%)")
    print(f"{'═'*110}\n")

    # Save files
    df.to_csv(TRADES_FILE, index=False)
    print(f"Trade log saved to {TRADES_FILE}")

    pd.DataFrame(stock_summary).to_csv(SUMMARY_FILE, index=False)
    print(f"Summary saved to {SUMMARY_FILE}")

    cc_df.to_csv('capital_curve.csv', index=False)
    print(f"Capital curve saved to capital_curve.csv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\nNSE Backtest v5 — BB Mean Reversion")
    print(f"Period            : {START_DATE} to {END_DATE}")
    print(f"Starting capital  : ₹{STARTING_CAPITAL:,}")
    print(f"Capital per trade : ₹{CAPITAL_PER_TRADE:,}")
    print(f"Max positions     : {MAX_POSITIONS}")
    print(f"BB Period         : {BB_PERIOD}, Std Dev: {BB_STD}")
    print(f"Stop loss         : {STOP_LOSS_PCT}% below entry")

    symbols  = load_watchlist(INPUT_FILE)
    all_data = fetch_all_data(symbols)

    if not all_data:
        print("No data fetched — exiting.")
        exit(1)

    trades, curve, skipped, final_capital = run_backtest(symbols, all_data)
    print_results(trades, curve, skipped, final_capital)
