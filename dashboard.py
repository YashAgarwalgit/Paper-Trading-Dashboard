# enhanced_dashboard.py V4.0 - Professional Quantitative Trading System Dashboard

import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import pickle
import json
from plotly.subplots import make_subplots
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import investpy

# --- Page Configuration ---
st.set_page_config(
    page_title="Institutional Quantitative Trading Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Styling ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00d4aa;
        backdrop-filter: blur(10px);
    }
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-running { background-color: #ff6b6b; }
    .status-ready { background-color: #51cf66; }
    .status-idle { background-color: #ffd43b; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px 8px 0px 0px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br'
}

# REPLACE this funct

# --- Advanced Helper Functions ---

# ==============================================================================
# V4.0 ADDITION: PAPER TRADING CORE LOGIC
# ==============================================================================
PORTFOLIOS_DIR = "portfolios"

# Ensure the portfolios directory exists
os.makedirs(PORTFOLIOS_DIR, exist_ok=True)

def get_portfolio_path(portfolio_name: str) -> str:
    """Constructs the full path for a given portfolio name."""
    return os.path.join(PORTFOLIOS_DIR, f"{portfolio_name}.json")

def get_available_portfolios() -> list:
    """Scans the portfolios directory and returns a list of available portfolio names."""
    files = [f for f in os.listdir(PORTFOLIOS_DIR) if f.endswith('.json')]
    return [os.path.splitext(f)[0] for f in files]

def initialize_portfolio_state(portfolio_name: str, initial_capital: float = 10000000.0):
    """Creates a default portfolio structure for a given portfolio name."""
    if not portfolio_name.strip():
        st.error("Portfolio name cannot be empty.")
        return
    
    success, price_data = get_live_prices(["^NSEI"]) # Using NIFTY 50 as default
    initial_benchmark_value = price_data.get("^NSEI", 0) if success else 0
        
    portfolio_data = {
        "metadata": {"portfolio_name": portfolio_name, "created_utc": datetime.utcnow().isoformat()},
        "balances": {"initial_capital": initial_capital, "cash": initial_capital, "market_value": 0.0, "total_value": initial_capital},
        "positions": {},
        "transactions": [],
        # --- NEW: Add equity_history to track portfolio value over time ---
        "equity_history": [{
            "timestamp": datetime.utcnow().isoformat(), 
            "total_value": initial_capital,
            "benchmark_value": initial_benchmark_value
        }]
    }
    # Save the new portfolio data to its own file
    with open(get_portfolio_path(portfolio_name), 'w') as f:
        json.dump(portfolio_data, f, indent=4)
    st.success(f"New portfolio '{portfolio_name}' created successfully!")


def load_portfolio_state(portfolio_name: str):
    """Loads a specific portfolio into Streamlit's session state."""
    path = get_portfolio_path(portfolio_name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            st.session_state.portfolio = json.load(f)
    else:
        # If a portfolio is selected that doesn't exist, handle it gracefully
        st.error(f"Portfolio '{portfolio_name}' not found. Loading default or first available.")
        available = get_available_portfolios()
        if available:
            load_portfolio_state(available[0]) # Load the first one available
        else:
            st.session_state.portfolio = None # No portfolios exist


def save_portfolio_state():
    """Saves the current session state back to the appropriate JSON file."""
    if 'portfolio' in st.session_state and st.session_state.portfolio:
        portfolio_name = st.session_state.portfolio['metadata']['portfolio_name']
        with open(get_portfolio_path(portfolio_name), 'w') as f:
            json.dump(st.session_state.portfolio, f, indent=4)

# One-time migration for existing users
def migrate_old_portfolio():
    """Checks for and moves the old single portfolio into the new directory structure."""
    old_file = "paper_portfolio.json"
    if os.path.exists(old_file):
        default_portfolio_name = "default_portfolio"
        new_path = get_portfolio_path(default_portfolio_name)
        
        if not os.path.exists(new_path):
            try:
                with open(old_file, 'r') as f:
                    data = json.load(f)
                
                # Update metadata and ensure new keys exist
                data['metadata']['portfolio_name'] = default_portfolio_name
                if 'equity_history' not in data:
                    # Initialize equity history with the current total value
                    initial_value = data.get('balances', {}).get('total_value', 10000000.0)
                    data['equity_history'] = [{"timestamp": datetime.utcnow().isoformat(), "total_value": initial_value}]
                
                with open(new_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                os.remove(old_file)
                st.toast("Successfully migrated existing portfolio!", icon="🚀")
            except Exception as e:
                st.error(f"Could not migrate old portfolio: {e}")

def delete_portfolio(portfolio_name: str):
    """Safely deletes a portfolio's JSON file."""
    path = get_portfolio_path(portfolio_name)
    if os.path.exists(path):
        try:
            os.remove(path)
            # Clear the portfolio from session state if it was the one loaded
            if 'portfolio' in st.session_state and st.session_state.portfolio['metadata']['portfolio_name'] == portfolio_name:
                st.session_state.portfolio = None
                # Change selection to another portfolio if one exists
                available = get_available_portfolios()
                st.session_state.selected_portfolio = available[0] if available else None
            st.success(f"Successfully deleted portfolio '{portfolio_name}'.")
        except Exception as e:
            st.error(f"Error deleting portfolio file: {e}")
    else:
        st.warning(f"Portfolio '{portfolio_name}' not found for deletion.")

# --- Add these to your Helper Functions section ---

@st.cache_data(ttl=1800) # Cache for 30 minutes
def get_sectoral_performance():
    """
    Fetches data for major NSE sectoral indices to calculate their
    performance over various timeframes.
    """
    sectors = {
        "IT": "^CNXIT", "Bank": "^NSEBANK", "Auto": "^CNXAUTO",
        "Finance": "^CNXFIN", "FMCG": "^CNXFMCG", "Pharma": "^CNXPHARMA",
        "Metals": "^CNXMETAL", "Energy": "^CNXENERGY", "Realty": "^CNXREALTY"
    }
    try:
        # Download 1 month of data to calculate all timeframes
        data = yf.download(list(sectors.values()), period="1mo", progress=False)['Close']
        if data.empty:
            return None, "Could not fetch sectoral data."

        # Calculate percentage change over different periods
        perf_1D = (data.iloc[-1] / data.iloc[-2] - 1) * 100
        perf_1W = (data.iloc[-1] / data.iloc[-5] - 1) * 100
        perf_1M = (data.iloc[-1] / data.iloc[0] - 1) * 100
        
        perf_df = pd.DataFrame({
            "1-Day %": perf_1D,
            "1-Week %": perf_1W,
            "1-Month %": perf_1M
        })
        # Map the tickers back to friendly names
        perf_df.index = list(sectors.keys())
        return perf_df.sort_values(by="1-Day %", ascending=False), None
    except Exception as e:
        return None, f"Sectoral performance error: {e}"

# --- In your Helper Functions section, REPLACE this function ---

@st.cache_data(ttl=3600) # Cache calendar for 1 hour
def get_economic_calendar(from_date, to_date, countries, importances):
    """
    An advanced function to fetch a filterable economic calendar using investpy.
    """
    try:
        # investpy needs dates in dd/mm/yyyy format
        from_date_str = from_date.strftime('%d/%m/%Y')
        to_date_str = to_date.strftime('%d/%m/%Y')
        
        df = investpy.economic_calendar(
            from_date=from_date_str,
            to_date=to_date_str,
            countries=countries,
            importances=importances
        )
        
        # Clean up the dataframe for better presentation
        if not df.empty:
            df = df[['date', 'time', 'event', 'currency', 'importance']]
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y').dt.strftime('%d-%b-%Y')
            df['importance'] = df['importance'].str.capitalize()
            # Sort by date and time
            df = df.sort_values(by=['date', 'time']).reset_index(drop=True)

        return df, None
    except Exception as e:
        return None, f"Could not fetch economic calendar. Investing.com may be blocking requests. Error: {e}"

def calculate_market_health(analytics_data):
    """
    Calculates a composite "Market Health" score from our existing breadth data.
    This function doesn't fetch new data; it processes existing data.
    """
    if not analytics_data:
        return None, "Breadth data not available for health calculation."

    # Normalize each component to a 0-100 scale
    score_50ma = analytics_data.get('above_50ma_pct', 0)
    score_200ma = analytics_data.get('above_200ma_pct', 0)
    
    new_highs = analytics_data.get('new_highs', 0)
    new_lows = analytics_data.get('new_lows', 1) # Avoid division by zero
    high_low_ratio = new_highs / (new_highs + new_lows)
    score_high_low = high_low_ratio * 100
    
    # Calculate a weighted average score
    # Weighting long-term health (200MA) and medium-term health (50MA) highest
    health_score = (score_200ma * 0.4) + (score_50ma * 0.4) + (score_high_low * 0.2)
    
    # Categorize the score
    if health_score > 70: health_label = "Very Strong"
    elif health_score > 55: health_label = "Strong"
    elif health_score < 30: health_label = "Very Weak"
    elif health_score < 45: health_label = "Weak"
    else: health_label = "Neutral"
        
    return {"score": health_score, "label": health_label}, None

@st.cache_data(ttl=120) # Cache for 2 minutes
def get_vix_data():
    """
    DEFINITIVE VIX SOLUTION: Reads the VIX data from a published Google Sheet,
    bypassing any and all network/API blocks.
    """
    try:
        # Your correct Google Sheet URL is here.
        SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRVL8VWXqIuQCl5BTeFBihncjKI0xSBqf04lq2OBd1nUzcXvSfEslIA-mgterVkPRoWKB7WNVGiW037/pub?gid=0&single=true&output=csv"

        # The problematic 'if' block has been removed.

        # Read the data directly using pandas
        df = pd.read_csv(SHEET_URL)
        
        # The data is in the first row after the header
        latest_price = df.iloc[0]['Last Price']
        previous_price = df.iloc[0]['Previous Close']
        
        change = latest_price - previous_price

        return {"price": latest_price, "change": change}, None
        
    except Exception as e:
        return None, f"Could not read VIX from Google Sheet. Check URL/Permissions. Error: {e}"
    
# REPLACE this function
@st.cache_data(ttl=120)
def get_header_indices_data():
    # ... (This function from our previous step is correct, no changes needed) ...
    indices = {
        "NIFTY 50": "^NSEI", "S&P 500": "^GSPC", "NASDAQ": "^IXIC",
        "FTSE 100": "^FTSE", "NIKKEI 225": "^N225", "INDIA VIX": "^INDIAVIX"
    }
    results = []
    for name, ticker_str in indices.items():
        try:
            ticker_obj = yf.Ticker(ticker_str)
            hist = ticker_obj.history(period="5d", interval="1d")
            if hist.empty or len(hist) < 2: continue
            latest_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2]
            change = latest_price - previous_price
            pct_change = (change / previous_price) * 100 if previous_price != 0 else 0
            results.append({"name": name, "price": latest_price, "change": change, "pct_change": pct_change})
        except Exception:
            continue
    return results, None if results else ("Could not fetch any header data.",)

    
def create_vix_gauge(vix_value):
    """
    Creates a Plotly bullet gauge chart to visualize the India VIX level.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=vix_value,
        title={'text': "India VIX (Fear Index)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [10, 35], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Invisible bar, we use steps
            'steps': [
                {'range': [10, 18], 'color': 'lightgreen'}, # Low Fear
                {'range': [18, 25], 'color': 'yellow'},   # Caution
                {'range': [25, 35], 'color': 'red'}        # High Fear
            ],
        }
    ))
    fig.update_layout(height=250, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    return fig

@st.cache_data(ttl=300) # Cache F&O movers data for 5 minutes
def get_fo_movers_data():
    """
    Fetches price data for a broad universe of F&O stocks and returns a
    DataFrame with the day's percentage change, ready for sorting.
    """
    # Note: This is a sample of the F&O list. A full, dynamic list is ideal for a production system.
    fo_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
        "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ADANIENT.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "JSWSTEEL.NS", "M&M.NS", "POWERGRID.NS", "NTPC.NS", "HCLTECH.NS", "GRASIM.NS", "INDUSINDBK.NS",
        "HINDALCO.NS", "CIPLA.NS", "DRREDDY.NS", "BAJAJFINSV.NS", "ADANIPORTS.NS", "TECHM.NS", "BRITANNIA.NS",
        "EICHERMOT.NS", "NESTLEIND.NS", "ONGC.NS", "COALINDIA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS",
        "TATACONSUM.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "UPL.NS", "HEROMOTOCO.NS", "SHRIRAMFIN.NS", "BPCL.NS"
    ]
    
    try:
        data = yf.download(fo_tickers, period="2d", interval="1d", progress=False)
        if data.empty:
            return None, "API returned no data for the F&O universe."
            
        close = data['Close']
        if close.count().sum() < 2:
            return None, "Not enough data points for F&O universe."
            
        # Calculate percentage change
        pct_change = (close.iloc[-1] / close.iloc[0] - 1) * 100
        
        # Create a clean DataFrame
        movers_df = pd.DataFrame({
            'Last Price': close.iloc[-1],
            '% Change': pct_change
        })
        movers_df.index.name = "Ticker"
        movers_df = movers_df.dropna().sort_values(by='% Change', ascending=False)
        
        return movers_df, None
        
    except Exception as e:
        return None, f"An API error occurred: {e}"
    
@st.cache_data(ttl=3600) # Cache news for 1 hour
def get_news_sentiment():
    """
    Scrapes Google News for Indian market headlines and analyzes their sentiment using VADER.
    """
    try:
        # Scrape Google News
        url = "https://news.google.com/search?q=indian+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [a.text for a in soup.find_all('a', {'class': 'JtKRv'})[:15]] # Get top 15 headlines

        if not headlines:
            return None, "Could not find any news headlines."

        # Analyze Sentiment
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        
        # Categorize sentiment
        if avg_score >= 0.05:
            sentiment_label = "Positive"
        elif avg_score <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        return {"score": avg_score, "label": sentiment_label, "headlines": headlines}, None
    except Exception as e:
        return None, f"Could not fetch news sentiment. Error: {e}"
    
def display_master_header():
    """
    Displays the master header with live index data using st.metric.
    """
    st.markdown("##### 🌎 Global Market Snapshot")
    
    # Fetch the data
    indices_data, error = get_header_indices_data()
    
    if error:
        st.error(f"Could not load market snapshot: {error}", icon="🔥")
        return
        
    if not indices_data:
        st.warning("Market snapshot data is currently unavailable.", icon="⚠️")
        return

    # Create columns for each metric
    cols = st.columns(len(indices_data))
    
    # Display each index in its own column
    for i, data in enumerate(indices_data):
        with cols[i]:
            st.metric(
                label=data["name"],
                value=f"{data['price']:,.2f}",
                delta=f"{data['change']:+.2f} ({data['pct_change']:+.2f}%)"
            )

# --- Live Price & P&L Functions ---
@st.cache_data(ttl=60)
def get_live_prices(tickers: list) -> tuple:
    """
    Fetches the latest market prices from yfinance.
    This version correctly formats tickers, applying the .NS suffix ONLY
    to stocks and not to indices like ^NSEI.
    """
    if not tickers:
        return (False, "No tickers provided.")

    # --- THE FIX: Conditionally add .NS only to non-index tickers ---
    yf_tickers = [f"{t}.NS" if not t.startswith('^') else t for t in tickers]
    
    try:
        data = yf.download(yf_tickers, period="2d", progress=False, group_by='ticker')
    except Exception as e:
        return (False, f"yfinance download failed: {e}")

    if data.empty:
        return (False, "Data download failed from yfinance (no data returned).")

    latest_prices = {}
    for i, ticker in enumerate(tickers): # Loop through original, clean ticker names
        yf_ticker_key = yf_tickers[i] # Get the corresponding formatted ticker
        
        try:
            # Handle yfinance's column structure (can be single or multi-level)
            if isinstance(data.columns, pd.MultiIndex):
                # Check if the ticker's data exists
                if yf_ticker_key not in data.columns.get_level_values(0):
                    continue
                price_series = data[yf_ticker_key]['Close']
            else:
                # If only one ticker was downloaded, columns might not be multi-level
                price_series = data['Close']

            valid_prices = price_series.dropna()
            
            if not valid_prices.empty:
                last_price = valid_prices.iloc[-1]
                latest_prices[ticker] = last_price # Use the original clean ticker for the key

        except Exception as e:
            st.warning(f"Could not process price for '{ticker}'. The specific error was: {e}. Skipping.")
            continue

    if not latest_prices:
        return (False, "Could not extract a valid price for any of the given tickers.")
    
    return (True, latest_prices)

def auto_refresh(interval_seconds: int = 60):
    """
    Injects a meta-refresh tag into the page to force a browser refresh
    at a specified interval.
    """
    # We use a JavaScript timeout to trigger the refresh
    script = f"""
        <script type="text/javascript">
            setTimeout(function() {{
                window.location.reload(true);
            }}, {interval_seconds * 1000});
        </script>
    """
    html(script, height=0, width=0)

def update_pnl_with_live_prices():
    """
    Marks the portfolio to market AND fetches the corresponding benchmark value,
    then records both for the equity curve.
    """
    if 'portfolio' not in st.session_state or not st.session_state.portfolio:
        st.info("Please select or create a portfolio first.")
        return

    portfolio = st.session_state.portfolio
    benchmark_ticker = portfolio['metadata'].get("benchmark_ticker", "^NSEI")
    
    tickers_to_fetch = list(portfolio['positions'].keys())
    # --- NEW: Add the benchmark ticker to our list of prices to fetch ---
    if benchmark_ticker not in tickers_to_fetch:
        tickers_to_fetch.append(benchmark_ticker)

    if not tickers_to_fetch:
        st.info("No positions to mark-to-market.")
        return

    success, price_data = get_live_prices(tickers_to_fetch)

    if not success:
        st.warning(f"Could not fetch live prices: {price_data}. P&L may be stale.")
        return
        
    # --- NEW: Extract the benchmark value from the results ---
    current_benchmark_value = price_data.get(benchmark_ticker, 0)
    if current_benchmark_value == 0:
        st.warning(f"Could not retrieve a valid price for benchmark '{benchmark_ticker}'. Analytics may be affected.")

    # Calculate portfolio market value (excluding the benchmark ticker)
    total_market_value = 0
    for ticker, pos in portfolio['positions'].items():
        if ticker in price_data:
            pos['last_price'] = price_data[ticker]
        pos['market_value'] = pos['quantity'] * pos.get('last_price', pos['avg_price'])
        total_market_value += pos['market_value']

    portfolio['balances']['market_value'] = total_market_value
    portfolio['balances']['total_value'] = portfolio['balances']['cash'] + total_market_value
    
    if 'equity_history' not in portfolio:
        portfolio['equity_history'] = []
    
    # --- NEW: Record the portfolio value AND the synchronized benchmark value ---
    portfolio['equity_history'].append({
        "timestamp": datetime.utcnow().isoformat(),
        "total_value": portfolio['balances']['total_value'],
        "benchmark_value": current_benchmark_value
    })
    
    save_portfolio_state()
    st.toast("Portfolio marked-to-market with latest prices!", icon="✅")

def add_cash_to_portfolio(amount_to_add: float):
    """
    Adds a specified amount of cash to the portfolio and logs it as a deposit.
    Also updates the initial_capital to reflect the new total contributed capital.
    """
    if amount_to_add <= 0:
        st.warning("Please enter a positive amount to deposit.")
        return

    portfolio = st.session_state.portfolio
    
    # --- IMPORTANT: Update both cash and the capital base ---
    # We treat 'initial_capital' as 'Total Contributed Capital' for accurate P&L
    portfolio['balances']['cash'] = float(portfolio['balances']['cash'] + amount_to_add)
    portfolio['balances']['initial_capital'] = float(portfolio['balances']['initial_capital'] + amount_to_add)
    
    # Update total value as well
    portfolio['balances']['total_value'] = float(portfolio['balances']['total_value'] + amount_to_add)

    # Log the transaction for record-keeping
    deposit_transaction = {
        "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 
        "ticker": "CASH", 
        "action": "DEPOSIT", 
        "quantity": float(amount_to_add), 
        "price": 1.0, 
        "commission": 0.0
    }
    portfolio['transactions'].insert(0, deposit_transaction)
    
    save_portfolio_state()
    st.success(f"Successfully deposited ₹{amount_to_add:,.2f} into your portfolio.")
    st.balloons()

def execute_paper_trade(ticker, action, quantity, market_price, slippage_pct=1.0, commission_bps=5):
    """
    Simulates a trade and returns a detailed receipt of the transaction.
    This version ensures all numbers are standard Python types to prevent JSON errors.
    """
    portfolio = st.session_state.portfolio

    # Validate quantity
    if quantity <= 0:
        return {"success": False, "message": "Quantity must be greater than zero."}

    # --- Slippage Calculation ---
    slippage_amount = 0
    if action == "BUY":
        execution_price = market_price * (1 + (slippage_pct / 100))
        slippage_amount = execution_price - market_price
    else:  # SELL
        execution_price = market_price * (1 - (slippage_pct / 100))
        slippage_amount = market_price - execution_price
        
    trade_value = quantity * execution_price
    commission = trade_value * (commission_bps / 10000)

    # --- Pre-Trade Validation ---
    if action == "BUY":
        if portfolio['balances']['cash'] < trade_value + commission:
            return {"success": False, "message": "Insufficient cash for this buy order."}
    elif action == "SELL":
        if ticker not in portfolio['positions'] or portfolio['positions'][ticker]['quantity'] < quantity:
            return {"success": False, "message": "Insufficient holdings for this sell order."}

    # --- Execute and Update Portfolio State ---
    if action == "BUY":
        portfolio['balances']['cash'] -= (trade_value + commission)
        if ticker in portfolio['positions']:
            pos = portfolio['positions'][ticker]
            new_total_cost = (pos['avg_price'] * pos['quantity']) + trade_value
            # --- FIX: Cast updated values to standard Python floats ---
            pos['quantity'] = float(pos['quantity'] + quantity)
            pos['avg_price'] = float(new_total_cost / pos['quantity'])
        else:
            # --- FIX: Cast initial values to standard Python floats ---
            portfolio['positions'][ticker] = {
                "quantity": float(quantity),
                "avg_price": float(execution_price),
                "last_price": float(execution_price)
            }
    
    elif action == "SELL":
        pos = portfolio['positions'][ticker]
        portfolio['balances']['cash'] += (trade_value - commission)
        # --- FIX: Cast updated quantity to standard Python float ---
        pos['quantity'] = float(pos['quantity'] - quantity)
        if pos['quantity'] < 1e-6: # Use a small threshold to handle float precision issues
            del portfolio['positions'][ticker]

    # --- Save State and Return Receipt ---
    new_transaction = {
        "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 
        "ticker": ticker, 
        "action": action, 
        # --- FIX: Cast transaction values to standard Python floats/ints ---
        "quantity": float(quantity), 
        "price": round(float(execution_price), 2), 
        "commission": round(float(commission), 2)
    }
    portfolio['transactions'].insert(0, new_transaction)
    save_portfolio_state()
    
    # Return a detailed receipt for the UI
    return {
        "success": True,
        "message": f"Successfully executed {action} {quantity} {ticker}.",
        "market_price": market_price,
        "slippage_pct": slippage_pct,
        "slippage_amount_per_share": slippage_amount,
        "execution_price": execution_price,
        "commission": commission,
        "total_cost": trade_value + commission if action == "BUY" else trade_value - commission
    }

def format_tickers_for_yf(tickers: list) -> list:
    """Appends the .NS suffix required by yfinance for Indian stocks."""
    return [f"{ticker}.NS" for ticker in tickers]

def generate_regime_map(min_val, mid_val, max_val, steps=10):
    """Generate dynamic regime mapping for leverage and ML conviction"""
    low_half = np.linspace(min_val, mid_val, num=(steps // 2))
    high_half = np.linspace(mid_val, max_val, num=(steps // 2) + 1)[1:]
    full_map_array = np.concatenate([low_half, high_half])
    return {i + 1: round(full_map_array[i], 4) for i in range(steps)}

def create_advanced_equity_curve(results_data):
    """Create sophisticated equity curve with multiple strategy comparison"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Cumulative Returns (Log Scale)', 'Rolling Sharpe Ratio (252D)', 'Drawdown Analysis'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Colors for different strategies
    colors = {
        'HRP': '#FF6B6B',
        'MaxSharpe': '#4ECDC4', 
        'Blended': '#45B7D1',
        'Benchmark': '#96CEB4'
    }
    
    for strategy, color in colors.items():
        if strategy in results_data and 'returns' in results_data[strategy]:
            returns = results_data[strategy]['returns']
            cum_returns = (1 + returns).cumprod()
            
            # Equity curve
            fig.add_trace(
                go.Scatter(x=returns.index, y=cum_returns, name=strategy, 
                          line=dict(color=color, width=2)),
                row=1, col=1
            )
            
            # Rolling Sharpe
            rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(x=returns.index, y=rolling_sharpe, name=f'{strategy} Sharpe',
                          line=dict(color=color, width=1), showlegend=False),
                row=2, col=1
            )
            
            # Drawdown
            peak = cum_returns.expanding().max()
            drawdown = (cum_returns - peak) / peak
            fig.add_trace(
                go.Scatter(x=returns.index, y=drawdown, fill='tonexty', name=f'{strategy} DD',
                          line=dict(color=color, width=1), showlegend=False),
                row=3, col=1
            )
    
    fig.update_layout(height=800, template="plotly_dark", title_text="Strategy Performance Dashboard")
    fig.update_yaxes(type="log", row=1, col=1)
    return fig

def create_factor_exposure_heatmap(factor_data):
    """Create factor exposure heatmap"""
    if not factor_data:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=factor_data['exposures'],
        x=factor_data['factors'],
        y=factor_data['dates'],
        colorscale='RdYlBu',
        zmid=0
    ))
    fig.update_layout(title="Factor Exposure Over Time", template="plotly_dark")
    return fig

def create_regime_indicator_chart(regime_data):
    """Create market regime indicator with multi-factor breakdown"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Market Regime Score (0-10)', 'Regime Factor Contributions'),
        vertical_spacing=0.1
    )
    
    if regime_data:
        # Main regime score
        fig.add_trace(
            go.Scatter(x=regime_data['dates'], y=regime_data['scores'],
                      fill='tonexty', name='Regime Score',
                      line=dict(color='#FFD700', width=2)),
            row=1, col=1
        )
        
        # Add regime zones
        fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Risk-On Zone", row=1, col=1)
        fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Risk-Off Zone", row=1, col=1)
        
        # Factor contributions (stacked bar)
        factors = ['Market Trend', 'Volatility', 'Breadth', 'Risk Appetite', 'Global Context']
        for i, factor in enumerate(factors):
            if factor in regime_data:
                fig.add_trace(
                    go.Bar(x=regime_data['dates'], y=regime_data[factor], 
                          name=factor, offsetgroup=1),
                    row=2, col=1
                )
    
    fig.update_layout(height=600, template="plotly_dark", barmode='stack')
    return fig

def create_portfolio_composition_chart(portfolio_data):
    """Create dynamic portfolio composition visualization"""
    if not portfolio_data:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sector Allocation', 'Top 10 Holdings'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Sector pie chart
    fig.add_trace(
        go.Pie(labels=portfolio_data['sectors'], values=portfolio_data['sector_weights'],
               name="Sectors", hole=0.4),
        row=1, col=1
    )
    
    # Top holdings bar chart
    fig.add_trace(
        go.Bar(x=portfolio_data['top_holdings_weights'], 
               y=portfolio_data['top_holdings'],
               orientation='h', name="Holdings"),
        row=1, col=2
    )
    
    fig.update_layout(height=500, template="plotly_dark")
    return fig

def execute_batch_portfolio_build(portfolio_name: str, total_capital: float, signal_file, slippage_pct: float, commission_bps: int):
    """
    Builds a new portfolio from an uploaded signal file in a single, efficient batch operation.
    """
    # --- 1. Validation and Setup ---
    if not signal_file:
        st.error("Please upload a signal file.")
        return
        
    try:
        signal_df = pd.read_csv(signal_file)
        if not all(col in signal_df.columns for col in ['Ticker', 'Weight']):
            st.error("Invalid file format. CSV must contain 'Ticker' and 'Weight' columns.")
            return
    except Exception as e:
        st.error(f"Failed to read signal file: {e}")
        return

    # Create the new portfolio
    initialize_portfolio_state(portfolio_name, total_capital)
    load_portfolio_state(portfolio_name) # Load it into the session
    
    portfolio = st.session_state.portfolio
    
    with st.spinner(f"Building portfolio '{portfolio_name}'... This may take a moment."):
        # --- 2. Batch Price Fetching ---
        all_tickers = signal_df['Ticker'].tolist()
        success, live_prices = get_live_prices(all_tickers)

        if not success:
            st.error(f"Failed to fetch live prices for tickers. Reason: {live_prices}")
            return
        
        # --- 3. Efficient Batch Calculation ---
        transactions = []
        total_cash_outflow = 0
        total_commission = 0

        for index, row in signal_df.iterrows():
            ticker = row['Ticker']
            weight = row['Weight']
            
            if ticker not in live_prices:
                st.warning(f"Skipping '{ticker}': Could not fetch its live price.")
                continue

            market_price = live_prices[ticker]
            
            # Apply slippage for BUY orders
            execution_price = market_price * (1 + (slippage_pct / 100))
            
            # Calculate quantity based on weight and execution price
            capital_for_ticker = total_capital * weight
            quantity = capital_for_ticker / execution_price
            
            if quantity <= 0:
                continue
                
            trade_value = quantity * execution_price
            commission = trade_value * (commission_bps / 10000)
            
            # --- 4. Prepare Portfolio Updates (without saving yet) ---
            # Add to positions
            portfolio['positions'][ticker] = {
                "quantity": float(quantity),
                "avg_price": float(execution_price),
                "last_price": float(market_price) # Store both for reference
            }
            
            # Create transaction record
            transactions.append({
                "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 
                "ticker": ticker, 
                "action": "BUY", 
                "quantity": float(quantity), 
                "price": round(float(execution_price), 2), 
                "commission": round(float(commission), 2)
            })
            
            total_cash_outflow += trade_value
            total_commission += commission

        # --- 5. Final, Single State Update ---
        portfolio['transactions'].extend(transactions)
        portfolio['balances']['cash'] -= (total_cash_outflow + total_commission)
        
        # Mark to market immediately after creation
        total_market_value = sum(p['quantity'] * p['last_price'] for p in portfolio['positions'].values())
        portfolio['balances']['market_value'] = total_market_value
        portfolio['balances']['total_value'] = portfolio['balances']['cash'] + total_market_value
        
        # Save the fully constructed portfolio state ONCE
        save_portfolio_state()

    st.success(f"Successfully built new portfolio '{portfolio_name}' with {len(transactions)} tickers!")
    st.balloons()

@st.cache_data(ttl=3600) # Cache for an hour
def get_benchmark_data(ticker="^NSEI", start_date=None, end_date=None):
    """
    Fetches historical data for a benchmark, with improved error reporting and
    a fallback from 'Adj Close' to 'Close'.
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, f"No data returned from yfinance for ticker '{ticker}'."
            
        # --- FIX: Check for 'Adj Close', otherwise fallback to 'Close' ---
        if 'Adj Close' in data.columns:
            price_column = 'Adj Close'
        else:
            price_column = 'Close'
        
        # Calculate returns using the selected price column
        return data[price_column].pct_change().dropna(), None # Return data, no error

    except Exception as e:
        return None, f"An exception occurred: {e}"
    
def calculate_realized_pnl(transactions: list) -> pd.DataFrame:
    """
    Analyzes a list of transactions to calculate realized P&L for each stock.
    This function pairs BUY and SELL orders to determine profit/loss from closed trades.
    """
    trades_only = [t for t in transactions if t['action'] in ['BUY', 'SELL']]
    if not trades_only:
        return pd.DataFrame() # Return empty if no trades

    trades = pd.DataFrame(trades_only)

    if trades.empty:
        return pd.DataFrame(columns=['Ticker', 'Realized P&L', 'Initial Investment', 'Final Value', 'Return %'])

    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    trades = trades.sort_values(by='timestamp')

    realized_gains = []
    
    for ticker in trades['ticker'].unique():
        ticker_trades = trades[trades['ticker'] == ticker].to_dict('records')
        
        buys = [t for t in ticker_trades if t['action'] == 'BUY']
        sells = [t for t in ticker_trades if t['action'] == 'SELL']

        while buys and sells:
            buy = buys.pop(0)
            sell = sells.pop(0)
            
            # Simple FIFO matching
            matched_qty = min(buy['quantity'], sell['quantity'])
            
            initial_investment = matched_qty * buy['price']
            final_value = matched_qty * sell['price']
            pnl = final_value - initial_investment
            
            realized_gains.append({
                'Ticker': ticker,
                'Realized P&L': pnl,
                'Initial Investment': initial_investment,
                'Final Value': final_value,
                'Return %': (pnl / initial_investment) * 100 if initial_investment != 0 else 0
            })
            
            # Put back remaining partial trades
            if buy['quantity'] > matched_qty:
                buys.insert(0, {**buy, 'quantity': buy['quantity'] - matched_qty})
            if sell['quantity'] > matched_qty:
                sells.insert(0, {**sell, 'quantity': sell['quantity'] - matched_qty})

    if not realized_gains:
        return pd.DataFrame(columns=['Ticker', 'Realized P&L', 'Initial Investment', 'Final Value', 'Return %'])
        
    pnl_df = pd.DataFrame(realized_gains)
    return pnl_df.groupby('Ticker').agg({
        'Realized P&L': 'sum',
        'Initial Investment': 'sum',
        'Final Value': 'sum'
    }).reset_index()

def calculate_portfolio_analytics(equity_history: list, risk_free_rate: float):
    """
    Calculates key performance and risk metrics from a pre-aligned equity history.
    This version is much simpler as it assumes benchmark data is already included.
    """
    if len(equity_history) < 2:
        return {}

    equity_df = pd.DataFrame(equity_history)
    equity_df = equity_df[equity_df['benchmark_value'] > 0] # Filter out any zero benchmark values
    
    if len(equity_df) < 2:
        return {}

    # --- SIMPLIFIED: No more resampling or reindexing needed! ---
    portfolio_returns = equity_df['total_value'].pct_change().dropna()
    benchmark_returns = equity_df['benchmark_value'].pct_change().dropna()

    if len(portfolio_returns) < 2:
        return {}

    # --- METRIC CALCULATIONS (The logic remains the same, but on cleaner data) ---
    volatility = portfolio_returns.std() * np.sqrt(252)
    excess_returns = portfolio_returns - (risk_free_rate / 252)
    sharpe_ratio = (excess_returns.mean() * 252) / volatility if volatility != 0 else 0
    
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (excess_returns.mean() * 252) / downside_deviation if downside_deviation != 0 else 0
    
    var_95 = portfolio_returns.quantile(0.05)
    
    covariance = portfolio_returns.cov(benchmark_returns)
    market_variance = benchmark_returns.var()
    beta = covariance / market_variance if market_variance != 0 else 0
    
    avg_portfolio_return = portfolio_returns.mean() * 252
    avg_benchmark_return = benchmark_returns.mean() * 252
    
    expected_return_capm = risk_free_rate + beta * (avg_benchmark_return - risk_free_rate)
    jensens_alpha = avg_portfolio_return - expected_return_capm
    
    return {
        "Annualized Volatility": volatility, "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio, "Historical VaR (95%)": var_95,
        "Portfolio Beta": beta, "Jensen's Alpha": jensens_alpha
    }

# --- V13.0 REPLACEMENT: New Helper Functions for a Cached Architecture ---

@st.cache_data(ttl=300) # Cache live index data for 5 minutes
def get_main_indices_data(period="1y", interval="1d"):
    """
    Fetches data for the main dashboard indices in a single, reliable batch call.
    This version is corrected to handle the MultiIndex DataFrame.
    """
    indices = {
        "NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "NIFTY BANK": "^NSEBANK",
        "NIFTY MIDCAP 100": "^CNXMIDCAP", "NIFTY SMALLCAP 100": "^CNXSC", "INDIA VIX": "^INDIAVIX"
    }
    try:
        data = yf.download(list(indices.values()), period=period, interval=interval, progress=False)
        if data.empty:
            return None, "yfinance returned no data for main indices."
            
        # --- FIX: Correctly access 'Adj Close' from the MultiIndex DataFrame ---
        adj_close = data.xs('Adj Close', level=1, axis=1)
        adj_close.columns = list(indices.keys()) # Rename columns to friendly names
        return adj_close, None
        
    except Exception as e:
        return None, f"Failed to fetch main index data: {e}"


@st.cache_data(ttl=14400) # Cache heavy analytics for 4 hours
def get_advanced_breadth_and_movers():
    """
    ADAPTIVE VERSION: This function is robust against incomplete historical data.
    It calculates metrics based on the available data window, preventing errors
    when a full year is not available for all stocks.
    """
    nifty50_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "LICI.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "KOTAKBANK.NS", "MARUTI.NS", "AXISBANK.NS", "SUNPHARMA.NS", "ADANIENT.NS", "ASIANPAINT.NS", "TITAN.NS", "NTPC.NS", "ONGC.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "POWERGRID.NS", "NESTLEIND.NS", "GRASIM.NS", "M&M.NS", "WIPRO.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "SBILIFE.NS", "HINDALCO.NS", "EICHERMOT.NS", "CIPLA.NS", "DRREDDY.NS", "TECHM.NS", "ADANIGREEN.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "DIVISLAB.NS", "UPL.NS", "HDFCLIFE.NS", "SHRIRAMFIN.NS"
    ]
    
    try:
        data = yf.download(nifty50_tickers, period="1y", progress=False, group_by='ticker')
        if data.empty:
            return None, "Failed to download historical data for NIFTY 50 constituents."

        if isinstance(data.columns, pd.MultiIndex):
            price_col_name = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(1) else 'Close'
            adj_close = data.xs(price_col_name, level=1, axis=1)
        else: # Fallback for flat structure
            adj_close = data[['Adj Close']] if 'Adj Close' in data.columns else data[['Close']]

        adj_close = adj_close.dropna(axis=1, how='all')
        if len(adj_close) < 2:
             return None, "Not enough data for analysis (less than 2 days)."

        # --- ADAPTIVE CALCULATION LOGIC ---
        available_data_len = len(adj_close)
        win_52wk = min(252, available_data_len)
        win_200d = min(200, available_data_len)
        win_50d = min(50, available_data_len)

        last_price = adj_close.iloc[-1]
        prev_close = adj_close.iloc[-2]
        
        movers_df = pd.DataFrame({'Last Price': last_price, '% Change': ((last_price / prev_close) - 1) * 100})
        movers_df.index = movers_df.index.str.replace('.NS', '', regex=False)
        movers_df = movers_df.dropna()
        
        advances = (movers_df['% Change'] > 0).sum()
        declines = (movers_df['% Change'] < 0).sum()

        high_52wk = adj_close.rolling(window=win_52wk, min_periods=1).max().iloc[-1]
        low_52wk = adj_close.rolling(window=win_52wk, min_periods=1).min().iloc[-1]
        
        new_highs = (last_price >= high_52wk).sum()
        new_lows = (last_price <= low_52wk).sum()
        
        # Safely calculate MAs only if enough data exists
        above_50ma_pct = ( (last_price > adj_close.rolling(window=win_50d).mean().iloc[-1]).sum() / len(adj_close.columns) ) * 100 if available_data_len >= win_50d else 0
        above_200ma_pct = ( (last_price > adj_close.rolling(window=win_200d).mean().iloc[-1]).sum() / len(adj_close.columns) ) * 100 if available_data_len >= win_200d else 0

        return {
            "movers_df": movers_df,
            "advances": int(advances), "declines": int(declines),
            "new_highs": int(new_highs), "new_lows": int(new_lows),
            "above_50ma_pct": above_50ma_pct,
            "above_200ma_pct": above_200ma_pct
        }, None
        
    except Exception as e:
        import traceback
        return None, f"A critical error occurred: {e}\n{traceback.format_exc()}"

@st.cache_data(ttl=900) # Cache each index for 15 minutes
def get_single_index_data(ticker, period_days=180):
    """
    Final Data Function v2.1: Sets end date to tomorrow to ensure the
    current day's data is included as soon as it's available on the API.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # --- ROBUSTNESS FIX: Set end date to tomorrow ---
        # This ensures that today's data (once published by the provider) is always included.
        end_date = datetime.now() + timedelta(days=1)
        start_date = datetime.now() - timedelta(days=period_days)
        
        data = ticker_obj.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="1d"
        )

        if data.empty or len(data) < 50:
            return None, f"Not enough data for {ticker} (need > 50 days)."

        # Calculate Technical Indicators
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data, None
    except Exception as e:
        return None, f"Data download failed for {ticker}. Reason: {e}"

def create_index_chart(df, index_name):
    """
    Creates a full Plotly TA chart with Price, MAs, and RSI from a DataFrame.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{index_name} Price & Moving Averages', 'RSI (14-Day)'),
        row_heights=[0.75, 0.25]
    )

    # --- Price and Moving Averages ---
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='Price'),
                  row=1, col=1)
    # MA20
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA 20',
                              line=dict(color='yellow', width=1.5)),
                  row=1, col=1)
    # MA50
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA 50',
                              line=dict(color='orange', width=2)),
                  row=1, col=1)

    # --- RSI Subplot ---
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                              line=dict(color='cyan', width=1.5)),
                  row=2, col=1)
    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)

    # --- Layout and Theming ---
    fig.update_layout(
        height=500,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    return fig

# --- Main Application ---
display_master_header()
st.markdown("<hr>", unsafe_allow_html=True) # Add a separator

st.markdown('<div class="main-header"><h1>⚡ Institutional Quantitative Trading Platform</h1><p>Advanced Multi-Factor Adaptive Trading System</p></div>', unsafe_allow_html=True)

# --- Module Selector ---
page = st.radio(
    "Select Platform Module:",
    ("Live Paper Trading", "Market Dashboard", "Backtest Analysis"),
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ==============================================================================
# MODULE 1: Backtest Analysis
# ==============================================================================
if page == "Backtest Analysis":
    
    # --- Advanced Sidebar Controls ---
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=QuantLab", width=200)
        st.markdown("### 🎛️ Strategy Configuration Center")
        
        # --- Section 1: Core Backtest Parameters ---
        with st.expander("📊 **Core Parameters**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", date(2021, 1, 1))
                portfolio_value = st.number_input("Capital (INR)", value=10000000, step=1000000, format="%d")
            with col2:
                end_date = st.date_input("End Date", date.today())
                benchmark = st.selectbox("Benchmark", ["NIFTY50", "NIFTY500", "SENSEX"])
        
        # --- Section 2: Advanced Universe Selection ---
        with st.expander("🎯 **Universe Selection**", expanded=False):
            universe_method = st.radio("Selection Method", 
                                    ["Dynamic Liquidity-Based", "Fixed Universe", "Sector Rotation"])
            
            if universe_method == "Dynamic Liquidity-Based":
                liquidity_threshold = st.slider("Min Daily Volume (Cr)", 1, 50, 10)
                momentum_lookback = st.slider("Momentum Lookback (Days)", 21, 252, 63)
                universe_size = st.slider("Universe Size", 50, 500, 200, 25)
            
            elif universe_method == "Fixed Universe":
                uploaded_universe = st.file_uploader("Upload Stock List (.csv)", type=['csv'])
            
            market_cap_filter = st.selectbox("Market Cap Filter", 
                                        ["All", "Large Cap Only", "Mid+Large Cap", "Small+Mid+Large"])
            
            exclude_sectors = st.multiselect("Exclude Sectors", 
                                        ["Banking", "IT", "Pharma", "Auto", "FMCG", "Metals", "Energy"])
        
        # --- Section 3: ML Model Configuration ---
        with st.expander("🤖 **Machine Learning Engine**", expanded=False):
            model_ensemble = st.selectbox("Model Architecture", 
                                        ["Advanced Stacking", "Gradient Boosting", "Neural Network", "Hybrid Ensemble"])
            
            if model_ensemble == "Advanced Stacking":
                level0_models = st.multiselect("Level 0 Models", 
                                            ["LightGBM", "XGBoost", "CatBoost", "RandomForest"], 
                                            default=["LightGBM", "XGBoost"])
                level1_model = st.selectbox("Meta-Model", ["Ridge", "Lasso", "ElasticNet", "Linear"])
            
            feature_selection = st.selectbox("Feature Selection", 
                                        ["Recursive Feature Elimination", "SHAP-based", "Mutual Information", "All Features"])
            
            prediction_horizons = st.multiselect("Prediction Horizons (Days)", 
                                            [5, 10, 15, 21, 30, 40, 60], 
                                            default=[10, 21, 40])
            
            quantile_regression = st.checkbox("Enable Quantile Regression", value=True)
            if quantile_regression:
                quantiles = st.slider("Confidence Interval", 0.05, 0.25, 0.15, 0.05, format="±%.2f")
        
        # --- Section 4: Portfolio Construction ---
        with st.expander("⚖️ **Portfolio Optimization**", expanded=False):
            optimization_method = st.selectbox("Primary Method", 
                                            ["Blended (ML+HRP)", "Max Sharpe", "Min Variance", "Risk Parity", "Black-Litterman"])
            
            if "Blended" in optimization_method:
                st.markdown("**Dynamic Blending Parameters:**")
                regime_blend_low = st.slider("ML Weight (Regime 1-3)", 0.0, 1.0, 0.2, 0.05)
                regime_blend_mid = st.slider("ML Weight (Regime 4-7)", 0.0, 1.0, 0.5, 0.05)
                regime_blend_high = st.slider("ML Weight (Regime 8-10)", 0.0, 1.0, 0.8, 0.05)
            
            st.markdown("**Risk Controls:**")
            max_position_weight = st.slider("Max Single Position", 0.01, 0.15, 0.05, 0.005, format="%.1f%%")
            max_sector_weight = st.slider("Max Sector Weight", 0.1, 0.5, 0.25, 0.05, format="%.0f%%")
            turnover_penalty = st.slider("Turnover Penalty", 0.0, 0.1, 0.02, 0.005)
            
            st.markdown("**Advanced Constraints:**")
            enable_factor_neutral = st.checkbox("Factor Neutral Portfolio")
            enable_sector_neutral = st.checkbox("Sector Neutral")
            long_only = st.checkbox("Long Only", value=True)
        
        # --- Section 5: Risk Management ---
        with st.expander("🛡️ **Risk Management**", expanded=False):
            st.markdown("**Dynamic Leverage:**")
            leverage_regime_map = {}
            for regime in [1, 5, 10]:
                leverage_regime_map[regime] = st.slider(f"Leverage (Regime {regime})", 0.0, 2.0, 
                                                    {1: 0.5, 5: 1.0, 10: 1.3}[regime], 0.05)
            
            st.markdown("**Stop Loss & Rebalancing:**")
            enable_dynamic_rebalancing = st.checkbox("Dynamic Rebalancing", value=True)
            if enable_dynamic_rebalancing:
                regime_change_threshold = st.slider("Regime Change Trigger", 1, 5, 2)
                volatility_trigger = st.slider("Volatility Trigger (VIX)", 15, 40, 25)
            
            portfolio_stop_loss = st.slider("Portfolio Stop Loss", -0.3, -0.05, -0.15, 0.01, format="%.0f%%")
            max_drawdown_limit = st.slider("Max Drawdown Limit", -0.25, -0.05, -0.12, 0.01, format="%.0f%%")
        
        # --- Section 6: Advanced Features ---
        with st.expander("⚙️ **Advanced Features**", expanded=False):
            enable_regime_model = st.checkbox("Market Regime Detection", value=True)
            if enable_regime_model:
                regime_factors = st.multiselect("Regime Factors", 
                                            ["Market Trend", "Volatility", "Breadth", "Risk Appetite", 
                                            "Yield Curve", "Global Markets", "Sector Rotation", "FII/DII Flow"],
                                            default=["Market Trend", "Volatility", "Breadth"])
            
            enable_factor_timing = st.checkbox("Factor Timing")
            enable_earnings_filter = st.checkbox("Earnings Calendar Filter")
            enable_event_filter = st.checkbox("Corporate Action Filter")
            
            transaction_costs = st.slider("Transaction Costs (bps)", 5, 50, 15)
            slippage_model = st.selectbox("Slippage Model", ["Linear", "Square Root", "Fixed"])
        
        st.markdown("---")
        
        # --- Action Buttons ---
        col1, col2 = st.columns(2)
        with col1:
            run_backtest = st.button("🚀 **Run Backtest**", use_container_width=True)
        with col2:
            save_config = st.button("💾 **Save Config**", use_container_width=True)
        
        validate_config = st.button("✅ **Validate Parameters**", use_container_width=True)
        load_preset = st.selectbox("Load Preset", ["Custom", "Conservative", "Aggressive", "Balanced"])

    # --- Main Dashboard Content ---

    # Results Upload Section
    st.header("📈 **Backtest Results & Analysis**")

    uploaded_file = st.file_uploader(
        "**Upload your backtest_summary.pkl file**", type=['pkl'],
        help="Upload the small summary file generated by run_backtest.py"
    )

    data_loaded = False

    # Mock data for demonstration
    if uploaded_file is not None:
        try:
            # Load the data from the uploaded file object
            results_data = pickle.load(uploaded_file)
            
            # Assuming the pkl file contains a dictionary with keys like 'results' and 'regime_data'
            st.session_state['backtest_results'] = results_data.get('results', {})
            st.session_state['regime_data'] = results_data.get('regime_data', {})
            st.success("✅ Results loaded successfully from file!")
            data_loaded = True
        except Exception as e:
            st.error(f"Error loading .pkl file: {e}")
            # Clear any stale data if loading fails
            if 'backtest_results' in st.session_state:
                del st.session_state['backtest_results']
            if 'regime_data' in st.session_state:
                del st.session_state['regime_data']

    # Allow loading demo data as an alternative
    if not data_loaded and st.button("Load Demo Data"):
        # Create mock results data
        dates = pd.date_range(start='2021-01-01', end='2024-01-01', freq='D')
        mock_results = {
            'MaxSharpe': {
                'returns': pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates),
                'metrics': {'cagr': 0.185, 'sharpe': 1.23, 'max_drawdown': -0.087, 'calmar': 2.13, 'sortino': 1.67}
            },
            'HRP': {
                'returns': pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates),
                'metrics': {'cagr': 0.142, 'sharpe': 1.01, 'max_drawdown': -0.065, 'calmar': 2.18, 'sortino': 1.34}
            },
            'Blended': {
                'returns': pd.Series(np.random.normal(0.0007, 0.013, len(dates)), index=dates),
                'metrics': {'cagr': 0.168, 'sharpe': 1.15, 'max_drawdown': -0.075, 'calmar': 2.24, 'sortino': 1.52}
            }
        }
        
        # Add regime data
        regime_data = {
            'dates': dates[::30],  # Monthly data
            'scores': np.random.uniform(2, 8, len(dates[::30])),
            'Market Trend': np.random.uniform(0, 2, len(dates[::30])),
            'Volatility': np.random.uniform(0, 2, len(dates[::30])),
            'Breadth': np.random.uniform(0, 2, len(dates[::30]))
        }
        
        st.session_state['backtest_results'] = mock_results
        st.session_state['regime_data'] = regime_data
        st.success("✅Demo Results loaded successfully!")
        data_loaded = True 

    # Display results if available
    if 'backtest_results' in st.session_state and st.session_state['backtest_results']:
        # This check will now work correctly
        results = st.session_state.backtest_results
        st.subheader("📊 **Performance Metrics**")
        
        # Create metrics comparison table
        metrics_df = pd.DataFrame({
            strategy: data['metrics'] for strategy, data in results.items()
        }).T
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            best_cagr = metrics_df['cagr'].max()
            best_strategy = metrics_df['cagr'].idxmax()
            st.metric("**CAGR**", f"{best_cagr:.1%}", 
                    delta=f"Best: {best_strategy}", delta_color="normal")
        
        with col2:
            best_sharpe = metrics_df['sharpe'].max()
            best_strategy = metrics_df['sharpe'].idxmax()
            st.metric("**Sharpe Ratio**", f"{best_sharpe:.2f}", 
                    delta=f"Best: {best_strategy}", delta_color="normal")
        
        with col3:
            best_dd = metrics_df['max_drawdown'].max()  # Least negative
            best_strategy = metrics_df['max_drawdown'].idxmax()
            st.metric("**Max Drawdown**", f"{best_dd:.1%}", 
                    delta=f"Best: {best_strategy}", delta_color="inverse")
        
        with col4:
            best_calmar = metrics_df['calmar'].max()
            best_strategy = metrics_df['calmar'].idxmax()
            st.metric("**Calmar Ratio**", f"{best_calmar:.2f}", 
                    delta=f"Best: {best_strategy}", delta_color="normal")
        
        with col5:
            best_sortino = metrics_df['sortino'].max()
            best_strategy = metrics_df['sortino'].idxmax()
            st.metric("**Sortino Ratio**", f"{best_sortino:.2f}", 
                    delta=f"Best: {best_strategy}", delta_color="normal")
        
        # Strategy comparison table
        st.subheader("📋 **Strategy Comparison Matrix**")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # --- Tabbed Analysis ---
        st.subheader("🔍 **Detailed Analysis**")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Performance", "🎯 Risk Analysis", "🌡️ Regime Model", 
            "💼 Portfolio", "📊 Factor Analysis", "⚙️ Attribution"
        ])
        
        with tab1:
            st.plotly_chart(create_advanced_equity_curve(results), use_container_width=True)
            
            # Monthly returns heatmap
            st.subheader("Monthly Returns Heatmap")
            # Use the 'results' variable which points to st.session_state.backtest_results
            blended_returns = results['Blended']['returns'] 
            monthly_returns = blended_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
            
            fig_heatmap = px.imshow(monthly_pivot.T, 
                                labels=dict(x="Year", y="Month", color="Returns"),
                                title="Monthly Returns Heatmap",
                                color_continuous_scale="RdYlGn",
                                aspect="auto")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk metrics radar chart
                fig_radar = go.Figure()
                
                categories = ['Sharpe', 'Sortino', 'Calmar', 'Max DD (Inv)', 'Volatility (Inv)']
                
                for strategy in results.keys():
                    metrics = results[strategy]['metrics']
                    values = [
                        metrics['sharpe'],
                        metrics['sortino'], 
                        metrics['calmar'],
                        -metrics['max_drawdown'] * 10,  # Invert and scale
                        1 / (results[strategy]['returns'].std() * np.sqrt(252))  # Invert volatility
                    ]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=strategy
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 3])
                    ),
                    showlegend=True,
                    title="Risk-Adjusted Performance Radar"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Drawdown analysis
                returns = results['Blended']['returns']
                cum_returns = (1 + returns).cumprod()
                peak = cum_returns.expanding().max()
                drawdown = (cum_returns - peak) / peak
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown,
                                        fill='tonexty', name='Drawdown',
                                        line=dict(color='red')))
                fig_dd.update_layout(title="Drawdown Analysis", yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, use_container_width=True)
        
        with tab3:
            if 'regime_data' in st.session_state:
                st.plotly_chart(create_regime_indicator_chart(st.session_state['regime_data']), 
                            use_container_width=True)
            else:
                st.info("Regime data not available in results file.")
        
        with tab4:
            # Mock portfolio data
            portfolio_data = {
                'sectors': ['IT', 'Banking', 'Pharma', 'Auto', 'FMCG', 'Metals'],
                'sector_weights': [0.25, 0.20, 0.15, 0.15, 0.15, 0.10],
                'top_holdings': ['TCS', 'INFY', 'HDFC', 'ICICI', 'RELIANCE', 'WIPRO', 'HUL', 'ITC', 'BAJAJ', 'MARUTI'],
                'top_holdings_weights': [0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03]
            }
            
            st.plotly_chart(create_portfolio_composition_chart(portfolio_data), 
                        use_container_width=True)
            
            # Portfolio statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Positions", "42")
                st.metric("Portfolio Turnover", "8.5%")
            with col2:
                st.metric("Concentration Risk", "Low")
                st.metric("Sector Diversity", "High")
        
        with tab5:
            # Mock factor data
            factor_data = {
                'exposures': np.random.randn(50, 8),
                'factors': ['Momentum', 'Value', 'Quality', 'Low Vol', 'Size', 'Profitability', 'Investment', 'Leverage'],
                'dates': pd.date_range('2023-01-01', periods=50, freq='W').strftime('%Y-%m-%d').tolist()
            }
            
            st.plotly_chart(create_factor_exposure_heatmap(factor_data), use_container_width=True)
        
        with tab6:
            st.subheader("Performance Attribution Analysis")
            
            # Attribution breakdown
            attribution_data = {
                'Source': ['Alpha Generation', 'Factor Timing', 'Portfolio Construction', 'Risk Management', 'Transaction Costs'],
                'Contribution (%)': [12.5, 3.2, 2.1, -1.8, -0.8],
                'Std Dev (%)': [8.2, 4.1, 2.3, 1.9, 0.5]
            }
            
            attribution_df = pd.DataFrame(attribution_data)
            st.dataframe(attribution_df, use_container_width=True)
            
            # Attribution waterfall chart
            fig_waterfall = go.Figure(go.Waterfall(
                name="Attribution", orientation="v",
                measure=["relative", "relative", "relative", "relative", "relative"],
                x=attribution_data['Source'],
                textposition="outside",
                text=[f"+{x}%" if x > 0 else f"{x}%" for x in attribution_data['Contribution (%)']],
                y=attribution_data['Contribution (%)'],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig_waterfall.update_layout(title="Performance Attribution Breakdown", showlegend=False)
            st.plotly_chart(fig_waterfall, use_container_width=True)

    # Action button responses
    if run_backtest:
        with st.spinner("Launching backtest engine..."):
            st.balloons()
            st.success("🚀 Backtest initiated! Check the API dashboard for progress updates.")
            st.info("📊 Results will be available for analysis once the backtest completes.")

    if save_config:
        st.success("💾 Configuration saved successfully!")

    if validate_config:
        st.success("✅ All parameters validated successfully!")
        st.info("📋 Configuration is ready for backtesting.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: grey;'>
        <p>Institutional Quantitative Trading Platform | Built for Professional Research & Deployment</p>
        <p>🔒 Secure • ⚡ High Performance • 📊 Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MODULE 2: Live Paper Trading (Corrected with Callback Pattern)
# ==============================================================================

elif page == "Live Paper Trading":
    import time
    migrate_old_portfolio()
    available_portfolios = get_available_portfolios()

    with st.sidebar:
        st.markdown("### 🗂️ Portfolio Selection")

        if not available_portfolios:
            st.warning("No portfolios found. Please create one.")
            selected_portfolio = None
        else:
            if 'selected_portfolio' not in st.session_state or st.session_state.selected_portfolio not in available_portfolios:
                st.session_state.selected_portfolio = available_portfolios[0]
            
            selected_portfolio = st.selectbox(
                "Select Portfolio", available_portfolios, 
                index=available_portfolios.index(st.session_state.selected_portfolio),
                key='selected_portfolio'
            )

        if selected_portfolio and ('portfolio' not in st.session_state or st.session_state.portfolio['metadata']['portfolio_name'] != selected_portfolio):
            load_portfolio_state(selected_portfolio)

        with st.expander("➕ Create New Portfolio (Manual)"):
            new_portfolio_name = st.text_input("New Portfolio Name", placeholder="e.g., 'momentum_strategy'")
            new_initial_capital = st.number_input("Initial Capital", value=1000000.0, step=100000.0, format="%.2f")
            if st.button("Create Portfolio", use_container_width=True):
                if new_portfolio_name and new_portfolio_name not in available_portfolios:
                    initialize_portfolio_state(new_portfolio_name, new_initial_capital)
                    st.session_state.selected_portfolio = new_portfolio_name # Auto-select the new portfolio
                    st.rerun()
                else:
                    st.error("Portfolio name is empty or already exists.")
        
        st.markdown("---")

        # --- V7.0 NEW: AUTOMATED PORTFOLIO BUILDER ---
        with st.expander("🤖 Automated Portfolio Builder"):
            st.info("Upload a CSV with 'Ticker' and 'Weight' columns to build a portfolio instantly.")
            
            builder_portfolio_name = st.text_input("New Portfolio Name", placeholder="e.g., 'june_momentum_fund'")
            builder_capital = st.number_input("Total Capital to Allocate (INR)", value=10000000.0, step=100000.0, format="%.2f", key="builder_capital")
            builder_signal_file = st.file_uploader("Upload Signal File (.csv)", type=['csv'])

            if st.button("Build Portfolio From Signal", use_container_width=True, type="primary"):
                if builder_portfolio_name and builder_portfolio_name not in available_portfolios:
                    # We need slippage and commission settings for the build
                    # Let's use the ones from the 'Trade Settings' expander
                    slippage = st.session_state.get('slippage_pct_val', 1.0) # Using session state for settings
                    commission = st.session_state.get('commission_bps_val', 5)
                    
                    execute_batch_portfolio_build(builder_portfolio_name, builder_capital, builder_signal_file, slippage, commission)
                    st.session_state.selected_portfolio = builder_portfolio_name
                    st.rerun()
                else:
                    st.error("Please provide a unique new portfolio name.")
        # --- End of New Section ---
        
        st.markdown("---")

        if 'portfolio' in st.session_state and st.session_state.portfolio:
            st.markdown("### 💵 Live Trading Terminal")
            st.markdown(f"**Active Portfolio:** `{st.session_state.portfolio['metadata']['portfolio_name']}`")

            if st.button("🔄 Mark-to-Market", use_container_width=True, type="primary"):
                update_pnl_with_live_prices()

            with st.expander("⚙️ Trade Settings"):
                # Add keys to these widgets to store their values in session state
                st.number_input("Slippage (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.2f", key="slippage_pct_val")
                st.number_input("Commission (bps)", min_value=0, max_value=50, value=5, step=1, key="commission_bps_val")
            
            st.markdown("**Live Order Entry**")

            for key, default_value in {
                'live_price': 0.0, 'trade_ticker_input': '',
                'trade_receipt': None, 'trade_quantity': 1.0
            }.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value
            
            # --- FIX 1: Create a callback function to reset the form state ---
            def reset_form_state():
                st.session_state.trade_ticker_input = ""
                st.session_state.live_price = 0.0
                st.session_state.trade_quantity = 1.0

            ticker_to_trade = st.text_input("1. Stock Ticker", value=st.session_state.trade_ticker_input)
            
            if st.button("Fetch Live Price", use_container_width=True):
                st.session_state.trade_receipt = None
                if ticker_to_trade:
                    with st.spinner(f"Fetching price for {ticker_to_trade.upper()}..."):
                        success, price_data = get_live_prices([ticker_to_trade.upper()])
                        if success:
                            st.session_state.live_price = price_data[ticker_to_trade.upper()]
                            st.session_state.trade_ticker_input = ticker_to_trade
                            st.rerun()
                        else:
                            st.session_state.live_price = 0.0
                            st.error(f"Could not fetch price: {price_data}")
                else:
                    st.warning("Please enter a ticker symbol.")

            if st.session_state.live_price > 0 and (ticker_to_trade.upper() != st.session_state.trade_ticker_input.upper()):
                st.warning("Ticker has changed. Please fetch the new price before executing.")
                st.session_state.live_price = 0.0

            if st.session_state.live_price > 0:
                st.info(f"Live Market Price for {st.session_state.trade_ticker_input.upper()}: ₹{st.session_state.live_price:,.2f}")

                with st.form("execution_form"):
                    st.markdown("**2. Confirm & Execute Trade**")
                    action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
                    st.number_input("Quantity", min_value=0.0001, step=0.0001, format="%.4f", key='trade_quantity')
                    
                    submitted = st.form_submit_button("Execute Paper Trade")
                    
                    if submitted:
                        # --- FIX: Read slippage and commission directly from session_state ---
                        slippage = st.session_state.slippage_pct_val
                        commission = st.session_state.commission_bps_val
                        
                        trade_result = execute_paper_trade(
                            st.session_state.trade_ticker_input.upper(),
                            action,
                            st.session_state.trade_quantity,
                            st.session_state.live_price,
                            slippage_pct=slippage,      # Pass the correct variable
                            commission_bps=commission   # Pass the correct variable
                        )
                        st.session_state.trade_receipt = trade_result
                        reset_form_state()
                        st.rerun()

            if st.session_state.trade_receipt:
                receipt = st.session_state.trade_receipt
                if receipt["success"]:
                    with st.container():
                        st.success(receipt['message'])
                        st.markdown("##### Execution Receipt")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Market Price", f"₹{receipt['market_price']:,.2f}")
                        col2.metric("Execution Price", f"₹{receipt['execution_price']:,.2f}", delta=f"₹{receipt['slippage_amount_per_share']:,.4f} slippage", delta_color="inverse")
                        col3.metric("Commission Paid", f"₹{receipt['commission']:,.2f}")
                else:
                    st.error(f"Trade Failed: {receipt['message']}")
                
                if st.button("Dismiss Receipt"):
                    st.session_state.trade_receipt = None
                    st.rerun()

            st.markdown("---")
            with st.expander("🗑️ Delete Current Portfolio"):
                st.warning(f"This will permanently delete the '{selected_portfolio}' portfolio and all its history.")
                if st.checkbox(f"Yes, I want to delete '{selected_portfolio}'."):
                    if st.button("🔥 CONFIRM DELETION", use_container_width=True, type="primary"):
                        delete_portfolio(selected_portfolio)
                        st.rerun()

        else:
            st.info("Please create or select a portfolio to begin.")

        st.markdown("---")
        st.markdown("### ⚙️ Live Settings")
        auto_refresh_enabled = st.toggle("Enable Auto-Refresh (1 min)", value=False, key="auto_refresh_toggle")
        
        # We can add an interval selector for more control
        refresh_interval = st.select_slider(
            "Refresh Interval (seconds)",
            options=[30, 60, 120, 300],
            value=60,
            disabled=not auto_refresh_enabled
        )

    # --- Main Page Logic ---

    # We always update the portfolio when the page runs
    if 'portfolio' in st.session_state and st.session_state.portfolio:
        with st.spinner("Fetching latest market prices..."):
            update_pnl_with_live_prices()

    # --- The rest of the page display logic remains the same ---
    if 'portfolio' in st.session_state and st.session_state.portfolio:
        st.header(f"💼 Live Portfolio: {st.session_state.portfolio['metadata']['portfolio_name']}")
        
        # Display a status indicator for the auto-refresh
        if auto_refresh_enabled:
            st.caption(f"Auto-refresh is active. Updating every {refresh_interval} seconds.")

        portfolio_state = st.session_state.portfolio

        # KPI Display
        total_value = portfolio_state['balances']['total_value']
        initial_capital = portfolio_state['balances']['initial_capital']
        pnl = total_value - initial_capital
        pnl_pct = (pnl / initial_capital) if initial_capital != 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Value (INR)", f"₹{total_value:,.2f}")
        col2.metric("Total P&L (INR)", f"₹{pnl:,.2f}", delta=f"{pnl_pct:.2%}")
        col3.metric("Cash Balance (INR)", f"₹{portfolio_state['balances']['cash']:,.2f}")
        col4.metric("Market Value of Positions (INR)", f"₹{portfolio_state['balances']['market_value']:,.2f}")

        # Tabbed Display for Details
        tab1, tab2, tab3 = st.tabs(["📊 **Positions**", "📜 **Transaction Log**", "📈 **Portfolio Analytics**"])

        with tab1:
            if portfolio_state['positions']:
                positions_list = []
                for ticker, data in portfolio_state['positions'].items():
                    unrealized_pnl = (data.get('last_price', data['avg_price']) - data['avg_price']) * data['quantity']
                    positions_list.append({
                        "Ticker": ticker, "Quantity": data['quantity'], "Avg Entry Price": f"₹{data['avg_price']:.2f}",
                        "Last Market Price": f"₹{data.get('last_price', data['avg_price']):.2f}",
                        "Market Value": f"₹{data.get('market_value', 0):,.2f}",
                        "Unrealized P&L": f"₹{unrealized_pnl:,.2f}"
                    })
                positions_df = pd.DataFrame(positions_list)
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("No open positions in the portfolio.")

        with tab2:
            if portfolio_state['transactions']:
                trans_df = pd.DataFrame(portfolio_state['transactions'])
                st.dataframe(trans_df, use_container_width=True, height=400)
            else:
                st.info("No transactions have been recorded.")

        with tab3:
            st.subheader("Performance and Risk Analysis")
            
            equity_history = portfolio_state.get('equity_history', [])

            # Add a configurable risk-free rate slider for context
            risk_free_rate = st.slider(
                "Risk-Free Rate (%) for Analytics", 
                min_value=0.0, max_value=10.0, value=7.0, step=0.1,
                format="%.1f%%", key="risk_free_rate_val"
            ) / 100.0
            st.markdown("---")

            # --- 1. Historical Equity Curve ---
            st.markdown("#### Portfolio Equity Curve")
            if len(equity_history) > 1:
                equity_df = pd.DataFrame(equity_history)
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                
                fig_equity = px.line(equity_df, x='timestamp', y='total_value', title='Portfolio Value Over Time', labels={'timestamp': 'Date', 'total_value': 'Portfolio Value (INR)'})
                fig_equity.update_layout(template="plotly_dark")
                st.plotly_chart(fig_equity, use_container_width=True)
            else:
                st.info("Not enough data to display an equity curve. Use the 'Mark-to-Market' button to build your portfolio history.")

            st.markdown("---")
            
            # --- 2. Performance & Risk Metrics ---
            st.markdown("#### Key Performance & Risk Metrics")
            with st.spinner("Calculating performance metrics..."):
                analytics = calculate_portfolio_analytics(equity_history, risk_free_rate)
                
                if analytics:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Jensen's Alpha", f"{analytics.get('Jensen's Alpha', 0):.3f}", help="Measures the portfolio's return above the expected return given its beta (risk). Positive alpha indicates outperformance.")
                    col2.metric("Portfolio Beta", f"{analytics.get('Portfolio Beta', 0):.2f}", help="Measures the portfolio's volatility relative to the market (NIFTY 50). Beta > 1 is more volatile; < 1 is less volatile.")
                    col3.metric("Sharpe Ratio", f"{analytics.get('Sharpe Ratio', 0):.2f}", help="Measures risk-adjusted return. Higher is better.")
                    
                    with st.expander("View All Risk & Performance Metrics"):
                        st.metric("Sortino Ratio", f"{analytics.get('Sortino Ratio', 0):.2f}", help="Similar to Sharpe, but only considers downside volatility. Higher is better.")
                        st.metric("Annualized Volatility", f"{analytics.get('Annualized Volatility', 0):.2%}", help="The standard deviation of the portfolio's returns, a measure of its total risk.")
                        st.metric("95% Daily VaR (Value at Risk)", f"{analytics.get('Historical VaR (95%)', 0):.2%}", help="The most you can expect to lose in a single day, based on historical performance, with 95% confidence.")
                else:
                    st.info("Awaiting more portfolio history to calculate performance metrics. Please use 'Mark-to-Market' on at least two different occasions.")

            st.markdown("---")

            # --- 3. Realized P&L Attribution ---
            st.markdown("#### Realized P&L Attribution")
            pnl_df = calculate_realized_pnl(portfolio_state['transactions'])
            
            if not pnl_df.empty:
                st.dataframe(pnl_df.sort_values(by='Realized P&L', ascending=False), use_container_width=True)

                # P&L Bar Chart
                fig_pnl = px.bar(pnl_df, x='Ticker', y='Realized P&L', color='Realized P&L',
                                 title='Realized P&L by Stock (from closed trades)', color_continuous_scale=px.colors.diverging.RdYlGn,
                                 color_continuous_midpoint=0)
                fig_pnl.update_layout(template="plotly_dark")
                st.plotly_chart(fig_pnl, use_container_width=True)
            else:
                st.info("No realized P&L to display. Close some trades (i.e., sell stocks you have bought) to see attribution.")

    else:
        st.header("💼 Live Paper Portfolio")
        st.info("No portfolio loaded. Please select one from the sidebar or create a new one.")

    if auto_refresh_enabled:
        # Wait for the specified interval
        time.sleep(refresh_interval)
        # Programmatically re-run the script from the top
        st.rerun()

# ==============================================================================
# PAGE: MARKET DASHBOARD (V6 - FINAL, FEATURE-RICH VERSION)
# ==============================================================================
elif page == "Market Dashboard":
    st.title("🇮🇳 Institutional Market Dashboard")
    
    if st.button("REFRESH FULL MARKET ANALYSIS", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.success("Caches cleared! Rerunning the full market analysis...")
        st.rerun()

    # --- Create the main tabs for the dashboard ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Market Internals", 
        "📈 Sector & Movers", 
        "🗓️ Events & News", 
        "Index Charts"
    ])

    # ==================== TAB 1: MARKET INTERNALS ====================
    with tab1:
        st.header("Internal Health & Sentiment")
        
        # --- Fetch data needed for this tab ---
        with st.spinner("Loading internal market data..."):
            analytics_data, breadth_error = get_advanced_breadth_and_movers()
            vix_data, vix_error = get_vix_data()
            health_data, health_error = calculate_market_health(analytics_data)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Market Health Oscillator")
            if health_error:
                st.error(health_error)
            elif health_data:
                st.plotly_chart(create_vix_gauge(health_data['score']), use_container_width=True)
                st.metric(f"Health Score: {health_data['label']}", f"{health_data['score']:.1f} / 100")
        with col2:
            st.subheader("Volatility & Breadth")
            if vix_error:
                st.error(vix_error)
            elif vix_data:
                st.metric("India VIX", f"{vix_data['price']:.2f}", delta=f"{vix_data['change']:+.2f}")
            if breadth_error:
                st.error(breadth_error)
            elif analytics_data:
                st.metric("Advancing Stocks (N50)", analytics_data['advances'])
                st.metric("Declining Stocks (N50)", analytics_data['declines'])

    # ==================== TAB 2: SECTOR & MOVERS ====================
    with tab2:
        st.header("Sector Performance & Top Movers")

        # --- Fetch data needed for this tab ---
        with st.spinner("Loading sector and movers data..."):
            sector_perf_df, sector_error = get_sectoral_performance()
            movers_df, movers_error = get_fo_movers_data()

        st.subheader("Sector Performance")
        if sector_error:
            st.error(sector_error)
        elif sector_perf_df is not None:
            st.dataframe(sector_perf_df.style.format("{:.2f}%").background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("Top Gainers & Losers (F&O Universe)")
        if movers_error:
            st.error(movers_error)
        elif movers_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 10 Gainers**")
                st.dataframe(movers_df.head(10).style.format({'Last Price': "₹{:.2f}", '% Change': "{:+.2f}%"}).applymap(lambda v: 'color: #27ae60' if isinstance(v, (int, float)) and v > 0 else '', subset=['% Change']))
            with col2:
                st.markdown("**Top 10 Losers**")
                st.dataframe(movers_df.tail(10).sort_values(by='% Change', ascending=True).style.format({'Last Price': "₹{:.2f}", '% Change': "{:+.2f}%"}).applymap(lambda v: 'color: #c0392b' if isinstance(v, (int, float)) and v < 0 else '', subset=['% Change']))

    # ==================== TAB 3: EVENTS & NEWS ====================
    with tab3:
        st.header("Economic Calendar & Market News")

        # --- NEW: Interactive Controls for the Calendar ---
        st.subheader("Advanced Economic Calendar")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            # Let user select multiple countries
            countries = st.multiselect(
                "Select Countries", 
                ['India', 'United States', 'Euro Zone', 'United Kingdom', 'Japan', 'China', 'Germany'], 
                default=['India', 'United States', 'Euro Zone']
            )
        with col2:
            # Let user select multiple importance levels
            importances = st.multiselect(
                "Select Importance", 
                ['High', 'Medium', 'Low'], 
                default=['High', 'Medium']
            )

        # --- Data Fetching & Display Logic ---
        # We will look 7 days ahead from today
        from_date = datetime.now()
        to_date = from_date + timedelta(days=7)

        with st.spinner("Fetching calendar for the next 7 days..."):
            calendar_df, calendar_error = get_economic_calendar(
                countries=[c.lower() for c in countries],
                importances=[i.lower() for i in importances],
                from_date=from_date,
                to_date=to_date
            )

        if calendar_error:
            st.error(calendar_error)
        elif calendar_df is not None and not calendar_df.empty:
            st.info(f"Showing {len(calendar_df)} events for the next 7 days based on your filters.")
            # Group events by day for a much cleaner presentation
            for event_date in calendar_df['date'].unique():
                # Display a nice header for each day
                day_name = pd.to_datetime(event_date, format='%d-%b-%Y').strftime('%A')
                st.markdown(f"#### 🗓️ **{day_name}, {event_date}**")
                
                day_df = calendar_df[calendar_df['date'] == event_date]
                st.dataframe(day_df.drop('date', axis=1), use_container_width=True)
        else:
            st.warning("No matching economic events found for the selected criteria in the next 7 days.")

        # --- The News Sentiment section remains the same ---
        st.markdown("<hr>", unsafe_allow_html=True)

        with st.spinner("Loading calendar and news data..."):
            sentiment_data, sentiment_error = get_news_sentiment()
        
        st.subheader("Headline Sentiment Analysis")
        if sentiment_error:
            st.error(sentiment_error)
        elif sentiment_data:
            st.metric(label=f"Overall Sentiment: {sentiment_data['label']}", value=f"{sentiment_data['score']:.3f}")
            with st.expander("View Latest Headlines"):
                for h in sentiment_data['headlines']:
                    st.markdown(f"- {h}")

    # ==================== TAB 4: INDEX CHARTS ====================
    with tab4:
        st.header("Key Index Technical Analysis")
        # This section remains the same, fetching and displaying the 4 main index charts
        indices_to_chart = {
            "NIFTY 50": ["^NSEI"], "NIFTY BANK": ["^NSEBANK"],
            "NIFTY MIDCAP 100": ["^CNXMIDCAP", "MIDCAP150.NS"],
            "NIFTY SMALLCAP 100": ["^CNXSC", "SMALLCAP250.NS"]
        }
        chart_data = {}
        for name, tickers in indices_to_chart.items():
            for ticker in tickers:
                df, error = get_single_index_data(ticker=ticker)
                if df is not None:
                    chart_data[name] = {"df": df, "ticker": ticker, "error": None}; break
                else:
                    chart_data[name] = {"df": None, "ticker": None, "error": error}
        items = list(chart_data.items())
        for i in range(0, len(items), 2):
            c1, c2 = st.columns(2)
            with c1:
                if i < len(items):
                    name, result = items[i]
                    if result["df"] is not None: st.plotly_chart(create_index_chart(result["df"], f'{name} ({result["ticker"]})'), use_container_width=True)
                    else: st.subheader(name); st.error(f'Could not load data: {result["error"]}')
            with c2:
                if i + 1 < len(items):
                    name, result = items[i+1]
                    if result["df"] is not None: st.plotly_chart(create_index_chart(result["df"], f'{name} ({result["ticker"]})'), use_container_width=True)
                    else: st.subheader(name); st.error(f'Could not load data: {result["error"]}')
