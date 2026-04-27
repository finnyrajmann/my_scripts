import csv
import time
import re
import requests
import yfinance as yf
import pandas as pd
from tqdm import tqdm

FINANCIAL_SECTORS = {'Financial Services'}

# Thresholds
MIN_PROMOTER_HOLDING = 25.0
MIN_ROE = 10.0
MAX_DE_NON_FINANCIAL = 1.5
MIN_REVENUE_GROWTH = 0.0

def get_promoter_holding(symbol):
    try:
        url = f'https://www.screener.in/company/{symbol}/'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        match = re.search(r'Promoter Holding: ([\d.]+)%', r.text)
        if match:
            return float(match.group(1))
        return None
    except:
        return None

def get_yfinance_data(symbol):
    try:
        ticker = yf.Ticker(f'{symbol}.NS')
        info = ticker.info
        bs = ticker.balance_sheet

        # ROE - try info first, then calculate from financials
        roe = info.get('returnOnEquity')
        roe = roe * 100 if roe is not None else None

        if roe is None:
            try:
                income = ticker.financials
                net_income = income.loc['Net Income'].iloc[0]
                equity = bs.loc['Stockholders Equity'].iloc[0]
                if equity and equity != 0:
                    roe = round((net_income / equity) * 100, 1)
            except:
                pass

        # Revenue Growth
        rev_growth = info.get('revenueGrowth')

        # D/E from balance sheet
        de_ratio = None
        if bs is not None and not bs.empty:
            try:
                total_debt = bs.loc['Total Debt'].iloc[0]
                equity = bs.loc['Stockholders Equity'].iloc[0]
                if equity and equity != 0:
                    de_ratio = round(total_debt / equity, 2)
            except:
                pass

        return roe, rev_growth, de_ratio
    except:
        return None, None, None

def passes_filter(symbol, industry):
    is_financial = industry in FINANCIAL_SECTORS

    # Get promoter holding
    promoter = get_promoter_holding(symbol)
    time.sleep(1)  # polite delay for screener.in

    # Get yfinance data
    roe, rev_growth, de_ratio = get_yfinance_data(symbol)

    result = {
        'Symbol': symbol,
        'Industry': industry,
        'Promoter%': promoter,
        'ROE%': round(float(roe), 1) if roe is not None else None,
        'RevenueGrowth%': round(rev_growth * 100, 1) if rev_growth is not None else None,
        'DE_Ratio': float(de_ratio) if de_ratio is not None else None,
        'IsFinancial': is_financial,
        'PassFilter': False,
        'FailReason': []
    }

    # Apply filters
    if promoter is not None and promoter < MIN_PROMOTER_HOLDING:
        result['FailReason'].append(f'Promoter {promoter}%')

    if roe is None or float(roe) < MIN_ROE:
        result['FailReason'].append(f'ROE {result["ROE%"]}%')

    if rev_growth is None or rev_growth < MIN_REVENUE_GROWTH:
        result['FailReason'].append(f'RevGrowth {rev_growth}')

    if not is_financial:
        if de_ratio is None or float(de_ratio) > MAX_DE_NON_FINANCIAL:
            result['FailReason'].append(f'DE {de_ratio}')

    if not result['FailReason']:
        result['PassFilter'] = True

    return result

if __name__ == '__main__':
    # Load universe
    universe = []
    with open('universe.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            universe.append((row['Symbol'], row['Industry']))

    print(f"Processing {len(universe)} stocks...")

    results = []
    for symbol, industry in tqdm(universe):
        result = passes_filter(symbol, industry)
        results.append(result)
        time.sleep(0.5)

    # Save results
    passed = [r for r in results if r['PassFilter']]
    failed = [r for r in results if not r['PassFilter']]

    df = pd.DataFrame(results)
    df['FailReason'] = df['FailReason'].apply(lambda x: ', '.join(x) if x else '')
    df.to_csv('fundamental_results.csv', index=False)

    print(f"\nTotal processed  : {len(results)}")
    print(f"Passed filter    : {len(passed)}")
    print(f"Failed filter    : {len(failed)}")

    print(f"\nPassed stocks:")
    for r in passed:
        print(f"  {r['Symbol']:<20} {r['Industry']:<45} Promoter:{r['Promoter%']}% ROE:{r['ROE%']}%")

    print(f"\nResults saved to fundamental_results.csv")
