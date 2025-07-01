# updater.py

import time
import os
import json
from datetime import datetime
import yfinance as yf
import pandas as pd

# --- Configuration ---
PORTFOLIOS_DIR = "portfolios"
UPDATE_INTERVAL_SECONDS = 60 # Set to update every 60 seconds

# --- Helper Functions (Copied from your dashboard for standalone use) ---

def get_portfolio_path(portfolio_name: str) -> str:
    """Constructs the full path for a given portfolio name."""
    return os.path.join(PORTFOLIOS_DIR, f"{portfolio_name}.json")

def get_available_portfolios() -> list:
    """Scans the portfolios directory and returns portfolio names."""
    if not os.path.exists(PORTFOLIOS_DIR):
        return []
    files = [f for f in os.listdir(PORTFOLIOS_DIR) if f.endswith('.json')]
    return [os.path.splitext(f)[0] for f in files]

def get_live_prices(tickers: list) -> tuple:
    """Fetches the latest market prices from yfinance."""
    if not tickers:
        return (False, "No tickers provided.")

    # Format tickers for yfinance
    yf_tickers = [f"{t}.NS" if not t.startswith('^') else t for t in tickers]

    try:
        data = yf.download(yf_tickers, period="2d", progress=False, group_by='ticker')
        if data.empty:
            return (False, "yfinance returned no data.")

        latest_prices = {}
        for i, ticker in enumerate(tickers):
            yf_ticker_key = yf_tickers[i]
            price_series = None
            if isinstance(data.columns, pd.MultiIndex):
                if yf_ticker_key in data.columns:
                    price_series = data[yf_ticker_key]['Close']
            else:
                price_series = data['Close']

            if price_series is not None and not price_series.dropna().empty:
                latest_prices[ticker] = price_series.dropna().iloc[-1]

        return (True, latest_prices)
    except Exception as e:
        return (False, f"yfinance download failed: {e}")

# --- Core Updater Logic ---

def update_all_portfolios():
    """Finds and updates every portfolio file in the directory."""
    log_prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    portfolio_names = get_available_portfolios()

    if not portfolio_names:
        print(f"{log_prefix} No portfolios found. Sleeping...")
        return

    print(f"{log_prefix} Found {len(portfolio_names)} portfolios. Preparing update...")

    all_tickers = set()
    for name in portfolio_names:
        try:
            path = get_portfolio_path(name)
            with open(path, 'r') as f:
                data = json.load(f)
                all_tickers.update(data.get('positions', {}).keys())
        except Exception as e:
            print(f"{log_prefix} Error reading portfolio {name}: {e}")
            continue

    if not all_tickers:
        print(f"{log_prefix} No positions found across all portfolios. Sleeping...")
        return

    print(f"{log_prefix} Fetching live prices for {len(all_tickers)} unique tickers...")
    success, price_data = get_live_prices(list(all_tickers))
    if not success:
        print(f"{log_prefix} Failed to fetch live prices. Reason: {price_data}. Skipping update cycle.")
        return

    for name in portfolio_names:
        try:
            path = get_portfolio_path(name)
            with open(path, 'r+') as f:
                portfolio = json.load(f)

                total_market_value = 0
                for ticker, pos in portfolio.get('positions', {}).items():
                    if ticker in price_data:
                        pos['last_price'] = price_data[ticker]

                    # Use the last known price if a live price isn't available
                    current_price = pos.get('last_price', pos.get('avg_price', 0))
                    pos['market_value'] = pos.get('quantity', 0) * current_price
                    total_market_value += pos['market_value']

                portfolio['balances']['market_value'] = total_market_value
                portfolio['balances']['total_value'] = portfolio['balances'].get('cash', 0) + total_market_value

                f.seek(0)
                json.dump(portfolio, f, indent=4)
                f.truncate()
        except Exception as e:
            print(f"{log_prefix} Error updating portfolio {name}: {e}")

    print(f"{log_prefix} Successfully updated all portfolios.")

# --- Main Execution Loop ---

if __name__ == "__main__":
    print("Starting the 24/7 Portfolio Updater Service...")
    while True:
        try:
            update_all_portfolios()
        except Exception as e:
            print(f"A critical error occurred in the main loop: {e}")

        print(f"Next update cycle in {UPDATE_INTERVAL_SECONDS} seconds...")
        time.sleep(UPDATE_INTERVAL_SECONDS)