import csv
import subprocess
from collections import Counter

NSE_FO_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

FO_FILE = "fo_stocks.csv"
NIFTY500_FILE = "nifty500.csv"
UNIVERSE_FILE = "universe.csv"

def download_file(url, filename):
    print(f"Downloading {filename}...")
    result = subprocess.run([
        'curl', '-k',
        '-H', 'User-Agent: Mozilla/5.0',
        '-H', 'Referer: https://www.nseindia.com',
        url, '-o', filename
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Saved to {filename}")
    else:
        print(f"  Failed: {result.stderr}")
        raise Exception(f"Download failed for {url}")

def get_fo_symbols(filepath=FO_FILE):
    symbols = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header_passed = False
        for row in reader:
            if not row:
                continue
            second_col = row[1].strip() if len(row) > 1 else ''
            if second_col == 'Symbol':
                header_passed = True
                continue
            if not header_passed:
                continue
            if second_col and second_col != 'SYMBOL':
                symbols.append(second_col)
    return set(symbols)

def get_nifty500_data(filepath=NIFTY500_FILE):
    symbols = []
    industry_map = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row['Symbol'].strip()
            industry = row['Industry'].strip()
            symbols.append(symbol)
            industry_map[symbol] = industry
    return set(symbols), industry_map

def build_universe():
    # Download fresh data
    download_file(NSE_FO_URL, FO_FILE)
    download_file(NIFTY500_URL, NIFTY500_FILE)

    # Load both lists
    fo_symbols = get_fo_symbols()
    nifty500_symbols, industry_map = get_nifty500_data()

    # Compute intersection
    universe = fo_symbols & nifty500_symbols

    # Save universe with industry tags
    with open(UNIVERSE_FILE, 'w') as f:
        f.write('Symbol,Industry\n')
        for s in sorted(universe):
            industry = industry_map.get(s, 'Unknown')
            f.write(f'{s},{industry}\n')

    # Print summary
    print(f"\nF&O eligible stocks     : {len(fo_symbols)}")
    print(f"Nifty 500 stocks        : {len(nifty500_symbols)}")
    print(f"Intersection (universe) : {len(universe)}")
    print(f"In F&O but not Nifty500 : {len(fo_symbols - nifty500_symbols)}")
    print(f"In Nifty500 but not F&O : {len(nifty500_symbols - fo_symbols)}")
    print(f"\nUniverse saved to {UNIVERSE_FILE}")

    # Sector breakdown
    sectors = [industry_map.get(s, 'Unknown') for s in universe]
    sector_count = Counter(sectors)
    print(f"\nSector breakdown:")
    for sector, count in sector_count.most_common():
        print(f"  {sector:<45} {count}")

if __name__ == '__main__':
    build_universe()
