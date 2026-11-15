"""Project-wide settings: universe, dates, prediction horizon, file locations.

Everything that a reviewer might want to change lives here, so the rest of the
code never hard-codes a ticker, a date or a path.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (data/ is git-ignored: data never goes to GitHub)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # untouched downloads (incrementally updated)
PROCESSED_DIR = DATA_DIR / "processed"  # one combined, feature-rich dataframe
MODEL_DIR = DATA_DIR / "models"       # fitted models + chosen parameters
REPORTS_DIR = PROJECT_DIR / "reports"  # small result tables/plots (committed)

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
HORIZON = 10                      # predict the return over the next 10 trading days (~2 weeks)
TARGET_GROWTH = f"growth_future_{HORIZON}d"
TARGET = f"is_positive_growth_{HORIZON}d_future"   # absolute label (lecture style): stock goes up
TARGET_REL = f"is_outperform_snp500_{HORIZON}d_future"  # relative label: stock beats the S&P 500

START_DATE = "2000-01-01"         # history used for features/training
FETCH_BUFFER_DAYS = 400           # extra history so 365d growth / 252d windows are defined at START_DATE

# temporal split (by calendar date, in that order)
TRAIN_END = "2018-12-31"
VALID_END = "2022-06-30"          # validation: 2019-01 .. 2022-06, test: after that
EMBARGO_DAYS = HORIZON            # drop rows whose target window overlaps the next split

# trading costs used by every simulation (per side, as a fraction of the trade)
FEE_PER_SIDE = 0.001              # 0.1% per side ~ commission + spread for liquid US large caps
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Universe: top US stocks with > $50B market cap (same list as the 2026 course Module 05,
# companiesmarketcap.com), so results can be compared with the lectures.
# ---------------------------------------------------------------------------
US_STOCKS = [
    "NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "AVGO", "META", "TSLA", "LLY", "BRK-B",
    "MU", "WMT", "AMD", "JPM", "V", "XOM", "JNJ", "MA", "ABBV", "INTC",
    "PLTR", "CVX", "COST", "CSCO", "ORCL", "KO", "PG", "UNH", "MRK", "BAC",
    "AMAT", "HD", "PM", "GE", "LRCX", "PANW", "DELL", "CAT", "MS", "TXN",
    "GS", "RTX", "CRWD", "TMO", "NFLX", "WFC", "AMGN", "IBM", "GEV", "ANET",
    "KLAC", "AXP", "ADI", "C", "GILD", "APH", "QCOM", "MRVL", "SNDK", "DIS",
    "BLK", "CRM", "PEP", "ABT", "MCD", "VZ", "COP", "UNP", "DE", "NEE",
    "WELL", "PFE", "DHR", "SCCO", "ISRG", "TMUS", "TJX", "SCHW", "LMT", "WDC",
    "BX", "T", "IBKR", "BA", "VRTX", "UBER", "SPGI", "PGR", "MO", "COF",
    "MPC", "BMY", "PLD", "SBUX", "SYK", "PH", "BKNG", "SNOW", "LOW", "NEM",
    "CVS", "MCK", "NOW", "ADP", "GLW", "NET", "VLO", "SO", "PSX", "CME",
    "HCA", "MAR", "EQIX", "FTNT", "BNY", "GD", "ABNB", "DUK", "CSX", "ADBE",
    "EMR", "PWR", "DDOG", "FCX", "VRT", "USB", "CDNS", "SNPS", "CEG", "MRSH",
    "PNC", "WM", "WMB", "ICE", "ELV", "MMM", "UPS", "MNST", "MRNA", "INTU",
    "KKR", "CTAS", "EOG", "AMT", "APP", "ECL", "EPD", "REGN", "MCO", "LITE",
    "RCL", "HOOD", "HWM", "SPG", "TRV", "MSI", "MDLZ", "ITW", "NOC", "SHW",
    "SLB", "HLT", "CMCSA", "CI", "HPE", "WBD", "DASH", "NSC", "CVNA", "KMI",
    "CL", "URI", "ROST", "FDX", "GM", "RSG", "ET", "CMI", "ORLY", "AEP",
    "HON", "APO", "APD", "DLR", "TGT", "AJG", "TRGP", "LNG", "TDG", "BE",
]

# Market-wide series from yfinance: column prefix -> yfinance symbol
INDEXES = {
    "snp500": "^GSPC",
    "dji": "^DJI",
    "nasdaq": "^IXIC",
    "russell2000": "^RUT",
    "vix": "^VIX",
    "gold": "GC=F",
    "wti_oil": "CL=F",
    "btc_usd": "BTC-USD",
    "usd_index": "DX-Y.NYB",
}
BENCHMARK = "SPY"   # total-return S&P 500 ETF (dividends included) for strategy comparison

# Macro series from FRED: series id -> frequency (used to lag releases correctly)
FRED_SERIES = {
    "GDPPOT": "Q",        # real potential GDP
    "CPILFESL": "M",      # core CPI
    "FEDFUNDS": "M",      # fed funds rate
    "UNRATE": "M",        # unemployment rate
    "DGS3MO": "D",        # 3-month treasury
    "DGS1": "D",
    "DGS5": "D",
    "DGS10": "D",
    "BAA10Y": "D",        # Baa corporate minus 10y treasury yield: credit spread (risk appetite)
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# NEW data source #1 (not in the lectures): CBOE official volatility index history.
# VIX9D / VIX3M give the volatility *term structure*; VVIX = volatility of VIX; SKEW = tail-risk pricing.
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
CBOE_SERIES = ["VIX9D", "VIX3M", "VVIX", "SKEW"]

# NEW data source #2 (web scraping): GICS sector / sub-industry of S&P 500 members from Wikipedia
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-markets-analytics-zoomcamp course project)"}
