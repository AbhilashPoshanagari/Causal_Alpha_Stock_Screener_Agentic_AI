import os
import json
from datetime import datetime, timedelta, date
import time
from openai import OpenAI
import pathlib
import sys
import locale
import re
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict, Optional
import hashlib
from itertools import combinations
from dotenv import load_dotenv
import os
from pathlib import Path

# ===== Configuration Parameters =====
CACHE_FRESHNESS_HOURS = 1000  # hours before price cache is stale
# # CACHE_FILE_PATH = "./content/cache.json"
# # PRICE_CACHE_PATH = "./content/price_cache.json"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CONTENT_DIR = os.path.join(BASE_DIR, "..", "content")

# CACHE_FILE_PATH = os.path.join(CONTENT_DIR, "cache.json")
# PRICE_CACHE_PATH = os.path.join(CONTENT_DIR, "price_cache.json")
# NIFTY_50_PATH = os.path.join(CONTENT_DIR, "NIFTY_50_latest.csv")
# # Define full paths
# EVALUATION_LOG_PATH = os.path.join(CONTENT_DIR, "evaluation_log.csv")
# SIMILARITY_LOG_PATH = os.path.join(CONTENT_DIR, "similarity_log.csv")

# Resolve absolute path to the 'content' folder
CONTENT_DIR = Path(__file__).resolve().parent / "content"

# File paths
CACHE_FILE_PATH = CONTENT_DIR / "cache.json"
PRICE_CACHE_PATH = CONTENT_DIR / "price_cache.json"
NIFTY_50_PATH = CONTENT_DIR / "NIFTY_50_latest.csv"
EVALUATION_LOG_PATH = CONTENT_DIR / "evaluation_log.csv"
SIMILARITY_LOG_PATH = CONTENT_DIR / "similarity_log.csv"
MODEL_NAME = "gpt-4o"
DEFAULT_LOOKBACK_YEARS = 3
ROLLING_WINDOW_DAYS = 252

# from google.colab import userdata
# openai_key = userdata.get('OPENAI_API_KEY')
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", openai_key)
client = OpenAI()

# Force UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# List of reliable/trusted sites which the Agents can rely on - to do web research
trusted_urls_macro = [
    "rbi.org.in/",
    "finmin.gov.in/",
    "mospi.gov.in/",
    "sebi.gov.in/",
    "nseindia.com",
    "bseindia.com",
    "data.worldbank.org/",
    "data.oecd.org/",
    "tradingeconomics.com/",
    "fred.stlouisfed.org/",
    "ceicdata.com/",
    "investing.com/",  # /commodities/brent-oil
    "tradingeconomics.com/",  # india/currency
    "imf.org/",  # en/Publications/WEO
    "rsisinternational.org/",
    "brokerage-free.in/",
    "commerce.gov.in/",  # Export-import data by sector
    "finmin.nic.in/",  # Budget allocations by sector
    "indiabudget.gov.in/economicsurvey/",  # Macro + Sector performance annually
    "data.worldbank.org/country/india",  # Sectoral development data (infra, finance, health)
    "cmie.com/",  # Premium sectoral data
]

trusted_urls_sector = [
    "wikipedia.org/",
    "reuters.com/",
    "ticker.finology.in/",
    "ft.com/",
    "ibef.org/industry",  # Official industry and sector reports (India-specific)
    "moneycontrol.com/",
    "statista.com/markets/",  # Global sector data, statistics, industry trends
    "mckinsey.com/industries",  # Deep strategic sector reports (global + India)
    "www2.deloitte.com/",  # /global/en/insights/industry.html - Research on financial services, energy, consumer goods
    "home.kpmg/",  # /xx/en/home/insights.html - Sector-specific trends and analyses
    "rbi.org.in/",  # /Scripts/Publications.aspx?head=Reports - Sectoral credit growth, sector stress analysis
    "sebi.gov.in/",  # /sebiweb/home/HomeAction.do?doListing=yes - Financial sector outlook, regulatory updates
    "nseindia.com/",  # /market-data/live-equity-market - Sector index performance (Auto, FMCG, Pharma, etc.)
    "reuters.com/business/",  # Global sector news
    "bloomberg.com/markets/sectors",  # Premium sector analysis
    "ft.com/companies",  # Comprehensive sector coverage
    "bseindia.com/markets/Indices/Indices.aspx?expandable=0",  # Sector-wise performance metrics
    "finance.yahoo.com/sectors/",  # Broad sector movements (Energy, Tech, Financials)
    "tradingeconomics.com/india/indicators",  # Industry statistics (energy, services, agriculture)
    "economictimes.indiatimes.com/industry",  # Latest sector-specific news for Indian industries
    "business-standard.com/industry",  # Indian company & sector updates
    "cnbc.com/sectors/",  # Sector analysis for US, EU, Asia
    "stats.oecd.org/",  # Global sector data (manufacturing, finance, services)
    "data.imf.org/",  # Financial health of sectors globally
    "morningstar.in/default.aspx",
    "in.investing.com/indices/major-indices",
    "web.stockedge.com/sectors",
    "tradingview.com/markets/stocks-india/sectorandindustry-sector/",
]

trusted_urls_technical = [
    "nseindia.com",
    "bseindia.com",
    "moneycontrol.com",
    # "twelvedata.com",
    # "alphavantage.co",
    # "upstox.com/",
    "finance.yahoo.com",
    # "tradingview.com",
    "in.investing.com/",
    "investtech.com/",
    "finviz.com",
    # "chartink.com/",
    # "stockcharts.com",
    # "marketsmithindia.com/",
    "barchart.com",
    "web.stockedge.com",
    # "trendlyne.com",
]



# 1. Collapse each list into a comma-separated string
trusted_urls_macro_str = ", ".join(trusted_urls_macro)
trusted_urls_sector_str = ", ".join(trusted_urls_sector)
trusted_urls_technical_str = ", ".join(trusted_urls_technical)

# 2. Guardrail template
GUARDRAIL_TEMPLATE = (
    "Use only these trusted sources for factual data: {sources_list}. "
    "Rely exclusively on them when conducting this analysis."
)

# 3. Toggle for experimental guardrail usage
USE_TRUSTED_SOURCES = False  # default OFF; set True to enable guardrail in prompts




"""
# Example: iterating over the list
for site in trusted_urls[0:3]:
    print(f"Accessing data from: {site}")

# Undoing all usage of this variable in LLM calls - need to experiment with other architectures
"""

"""## 2. Price Cache Utilities
Functions to load, save, and check freshness of the price cache.
"""

def load_price_cache() -> Dict:
    try:
        if pathlib.Path(PRICE_CACHE_PATH).exists():
            return json.load(open(PRICE_CACHE_PATH))
    except Exception as e:
        print(f"Error loading price cache: {e}")
    return {}
#
def save_price_cache(cache: Dict):
    try:
        json.dump(cache, open(PRICE_CACHE_PATH, 'w'), indent=2)
    except Exception as e:
        print(f"Error saving price cache: {e}")

def is_price_cache_fresh(ticker: str, freshness_hours: int = CACHE_FRESHNESS_HOURS) -> bool:
    cache = load_price_cache()
    entry = cache.get(ticker, {})
    ts = entry.get("last_updated")
    if not ts:
        return False
    try:
        last = datetime.fromisoformat(ts)
    except:
        return False
    return (datetime.utcnow() - last) < timedelta(hours=freshness_hours)

"""## 3. Fetching Historical Data & Indicators
Function to download stock and benchmark data, compute returns, moving averages, Bollinger Bands, RSI, MACD, etc.
"""

def get_stock_data(ticker: str, start_date=None, end_date=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Set default dates
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365 * DEFAULT_LOOKBACK_YEARS)

    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError(f"No data for ticker {ticker}")
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    # Download benchmark data
    nifty_data = yf.download("^NSEI", start=start_date, end=end_date)
    if isinstance(nifty_data.columns, pd.MultiIndex):
        nifty_data.columns = nifty_data.columns.get_level_values(0)

    # Align indices
    nifty_data = nifty_data.reindex(stock_data.index, method='ffill')

    # Calculate daily returns
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    nifty_data['Daily_Return'] = nifty_data['Close'].pct_change()

    # Relative performance
    stock_data['Rel_Performance'] = stock_data['Daily_Return'] - nifty_data['Daily_Return']
    stock_data['Cum_Rel_Performance'] = (1 + stock_data['Rel_Performance']).cumprod() - 1

    # Volatility (returns std)
    stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=20).std()

    # Moving averages
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()

    # Bollinger Bands (price std)
    price_std = stock_data['Close'].rolling(window=20).std()
    stock_data['Upper_Band'] = stock_data['MA_20'] + 2 * price_std
    stock_data['Lower_Band'] = stock_data['MA_20'] - 2 * price_std

    # RSI
    delta = stock_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    return stock_data, nifty_data

"""## 4. Serialization of DataFrame
Convert a DataFrame of indicators into a JSON-serializable dictionary with summary metrics.
"""

def serialize_dataframe(df: pd.DataFrame) -> Dict:
    """Convert DataFrame to JSON-serializable dict including indicators and summary metrics."""
    n = len(df)
    last_date = df.index[-1]
    current_price = float(df['Close'].iloc[-1])
    # Year-to-date return
    try:
        start_of_year = datetime(last_date.year, 1, 1)
        ytd_slice = df.loc[df.index >= start_of_year, 'Close']
        if not ytd_slice.empty:
            ytd_start = float(ytd_slice.iloc[0])
            ytd_return = (current_price / ytd_start - 1) * 100
        else:
            ytd_return = 0.0
    except Exception:
        ytd_return = 0.0
    # Multi-year returns annualized
    return_1y = return_3y = return_5y = 0.0
    days = n
    if days > ROLLING_WINDOW_DAYS:
        price_1y = float(df['Close'].iloc[-ROLLING_WINDOW_DAYS])
        return_1y = (current_price / price_1y - 1) * 100
    if days > ROLLING_WINDOW_DAYS * 3:
        price_3y = float(df['Close'].iloc[-ROLLING_WINDOW_DAYS * 3])
        return_3y = ((current_price / price_3y) ** (1/3) - 1) * 100
    if days > ROLLING_WINDOW_DAYS * 5:
        price_5y = float(df['Close'].iloc[-ROLLING_WINDOW_DAYS * 5])
        return_5y = ((current_price / price_5y) ** (1/5) - 1) * 100
    current_rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df else 0.0
    current_volatility = float(df['Volatility'].iloc[-1] * 100) if 'Volatility' in df else 0.0

    return {
        'dates': df.index.strftime('%Y-%m-%d').tolist(),
        'open': df['Open'].tolist(),
        'high': df['High'].tolist(),
        'low': df['Low'].tolist(),
        'close': df['Close'].tolist(),
        'volume': df['Volume'].tolist() if 'Volume' in df.columns else [0]*n,
        'indicators': {
            'daily_return': df['Daily_Return'].fillna(0).tolist(),
            'volatility': df['Volatility'].fillna(0).tolist(),
            'ma_20': df['MA_20'].fillna(0).tolist(),
            'ma_50': df['MA_50'].fillna(0).tolist(),
            'ma_200': df['MA_200'].fillna(0).tolist(),
            'rsi': df['RSI'].fillna(0).tolist(),
            'macd': df['MACD'].fillna(0).tolist(),
            'upper_band': df['Upper_Band'].fillna(0).tolist(),
            'lower_band': df['Lower_Band'].fillna(0).tolist(),
        },
        'summary_metrics': {
            'current_price': current_price,
            'ytd_return': ytd_return,
            'return_1y': return_1y,
            'return_3y': return_3y,
            'return_5y': return_5y,
            'current_rsi': current_rsi,
            'current_volatility': current_volatility
        }
    }

"""## 5. Cache-Backed Fetch Function
Fetch price data with cache support, falling back to live fetch if cache is stale or force_refresh=True.
"""

def get_ticker_price_data(
    ticker: str,
    start_date=None,
    end_date=None,
    force_refresh: bool = False
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    ticker = ticker.upper()
    cache = load_price_cache()

    if not force_refresh and is_price_cache_fresh(ticker):
        entry = cache[ticker]['price_history']
        dates = pd.to_datetime(entry['dates'])
        df = pd.DataFrame({
            'Open': entry['open'],
            'High': entry['high'],
            'Low': entry['low'],
            'Close': entry['close'],
            'Volume': entry['volume'],
            'Daily_Return': entry['indicators']['daily_return'],
            'Volatility': entry['indicators']['volatility'],
            'MA_20': entry['indicators']['ma_20'],
            'MA_50': entry['indicators']['ma_50'],
            'MA_200': entry['indicators']['ma_200'],
            'RSI': entry['indicators']['rsi'],
            'MACD': entry['indicators']['macd'],
            'Upper_Band': entry['indicators']['upper_band'],
            'Lower_Band': entry['indicators']['lower_band'],
        }, index=dates)
        nifty = yf.download("^NSEI", start=dates[0], end=dates[-1] + timedelta(days=1))
        nifty = nifty.reindex(df.index, method='ffill')
        return df, entry, nifty

    # Fetch fresh
    df, nifty = get_stock_data(ticker, start_date, end_date)
    history = serialize_dataframe(df)
    # nifty_history = serialize_dataframe(nifty)
    # cache.setdefault("^NSEI",{})['price_history'] = nifty_history
    cache.setdefault(ticker, {})['price_history'] = history
    cache[ticker]['last_updated'] = datetime.utcnow().isoformat()
    # cache["^NSEI"]['last_updated'] = datetime.utcnow().isoformat()
    save_price_cache(cache)
    return df, history, nifty

"""## 6. Plotting Functions
Functions to plot price, moving averages, Bollinger Bands, RSI, volatility, MACD with Plotly.
"""

def plot_price_charts(stock_data: pd.DataFrame, ticker: str, nifty_data: Optional[pd.DataFrame]=None, benchmark: str="^NSEI"):
    # Price + MAs + benchmark
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name=f'{ticker} Close'))
    fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_50'], mode='lines', name='50-day MA', line=dict(dash='dash')))
    fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_200'], mode='lines', name='200-day MA', line=dict(dash='dash')))
    if nifty_data is not None and not nifty_data.empty:
        sf = stock_data['Close'].iloc[0] / nifty_data['Close'].iloc[0]
        fig1.add_trace(go.Scatter(x=nifty_data.index, y=nifty_data['Close']*sf, mode='lines', name=f'{benchmark} (scaled)'))
    fig1.update_layout(title=f'{ticker} Price & MAs', xaxis_title='Date', yaxis_title='Price')
    # fig1.show()

    # Bollinger Bands
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
    fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_Band'], mode='lines', name='Upper Band', line=dict(width=0.5)))
    fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_Band'], mode='lines', name='Lower Band', fill='tonexty', line=dict(width=0.5)))
    fig2.update_layout(title=f'{ticker} Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
    # fig2.show()

    # RSI
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'))
    fig3.add_hline(y=70, line_dash='dash', annotation_text='Overbought')
    fig3.add_hline(y=30, line_dash='dash', annotation_text='Oversold')
    fig3.update_layout(title=f'{ticker} RSI', xaxis_title='Date', yaxis_title='RSI')
    fig3.update_yaxes(range=[0,100])
    # fig3.show()

    # Volatility
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Volatility']*100, mode='lines', name=f'{ticker} Volatility (%)'))
    if nifty_data is not None and not nifty_data.empty:
        nifty_data['Daily_Return'] = nifty_data['Close'].pct_change()
        nv = nifty_data['Daily_Return'].rolling(window=20).std()*100
        fig4.add_trace(go.Scatter(x=nifty_data.index, y=nv, mode='lines', name=f'{benchmark} Volatility'))
        fig4.update_layout(title=f'{ticker} vs {benchmark} Volatility', xaxis_title='Date', yaxis_title='Volatility (%)')
    else:
        fig4.update_layout(title=f'{ticker} Volatility', xaxis_title='Date', yaxis_title='Volatility (%)')
    # fig4.show()

    # MACD
    # print(f'MACD : {stock_data.MACD_Signal}')
    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD'))
    fig5.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD_Signal'], mode='lines', name='Signal'))
    hist = stock_data['MACD'] - stock_data['MACD_Signal']
    colors = ['green' if v>=0 else 'red' for v in hist]
    fig5.add_trace(go.Bar(x=stock_data.index, y=hist, name='Histogram', marker_color=colors))
    fig5.update_layout(title=f'{ticker} MACD', xaxis_title='Date')
    fig5.show()

    return {
        "price_chart": fig1.to_plotly_json(),
        "bollinger_chart": fig2.to_plotly_json(),
        "rsi_chart": fig3.to_plotly_json(),
        "volatility_chart": fig4.to_plotly_json(),
        "macd_chart": fig5.to_plotly_json()
    }

"""## 7. LLM Cache Implementation
Functions to load, save, and validate freshness of the LLM result cache.
"""

def load_cache():
    """Load the cache from the cache file or create a new one if it doesn't exist."""
    try:
        if pathlib.Path(CACHE_FILE_PATH).exists():
            with open(CACHE_FILE_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
    return {}

def save_cache(cache):
    """Save the cache to the cache file."""
    try:
        with open(CACHE_FILE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")

def is_cache_fresh(timestamp, freshness_hours=CACHE_FRESHNESS_HOURS):
    """Check if the cached data is fresh based on its timestamp."""
    if not timestamp:
        return False

    cache_time = datetime.fromisoformat(timestamp)
    current_time = datetime.now()
    return (current_time - cache_time) < timedelta(hours=freshness_hours)

def get_cache_entry(ticker, domain):
    """Get a cache entry for a specific ticker and domain."""
    cache = load_cache()
    if ticker not in cache:
        return None
    if domain not in cache[ticker]:
        return None
    entry = cache[ticker][domain]
    if isinstance(entry.get("result"), str) and "error" in entry.get("result", "").lower():
        return None
    if is_cache_fresh(entry.get("last_updated")):
        return entry.get("result")
    return None

def update_cache_entry(ticker, domain, result):
    """Update a cache entry for a specific ticker and domain."""
    cache = load_cache()
    timestamp = datetime.now().isoformat()
    if ticker not in cache:
        cache[ticker] = {}
    cache[ticker][domain] = {
        "result": result,
        "last_updated": timestamp
    }
    cache[ticker]["ticker_last_updated"] = timestamp
    save_cache(cache)
    return result

"""##8. Utility Functions for LLM Interaction
Text sanitization, prompt formatting, similarity computations, etc.
"""

def extract_text_from_response(response):
    """
    Extract text content from the OpenAI Responses API response.
    """
    try:
        if hasattr(response, 'output') and isinstance(response.output, list):
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'message':
                    if hasattr(item, 'content') and isinstance(item.content, list):
                        text_parts = []
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                text_parts.append(content_item.text)
                        if text_parts:
                            return "\n".join(text_parts)
        if hasattr(response, 'model_dump'):
            dump = response.model_dump()
            if 'output' in dump and isinstance(dump['output'], list):
                for item in dump['output']:
                    if item.get('type') == 'message' and 'content' in item:
                        for content in item['content']:
                            if 'text' in content:
                                return content['text']
        return str(response)
    except Exception as e:
        print(f"Error extracting text from response: {e}")
        return str(response)

def sanitize_utf8(text):
    """
    Removes characters that are not valid UTF-8 and surrogate pairs that break serialization.
    """
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[\uD800-\uDFFF]', '', text)
    text = ''.join(c for c in text if c.isprintable() or c in "\n\t ")
    return text

def format_prompt_with_hallucination_control(task_prompt):
    """Add hallucination control instructions to a prompt."""
    return f"{task_prompt}\n\nUse domain knowledge for standard definitions, but do not guess or invent real-time facts not present in the search results. If missing data, disclaim it: 'No data found.'"

def compute_similarity(text1, text2):
    """Compute cosine similarity between two text snippets."""
    emb1 = embedding_model.encode(text1, convert_to_tensor=True)
    emb2 = embedding_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def chunk_text(text, max_words=100):
    """Split text into chunks of max_words."""
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def average_cosine_similarity(base_output, comparison_output):
    """Compute average cosine similarity between two texts split into chunks."""
    base_chunks = chunk_text(base_output)
    comp_chunks = chunk_text(comparison_output)
    if not base_chunks or not comp_chunks:
        return 0.0
    sims = [compute_similarity(a, b) for a, b in zip(base_chunks, comp_chunks)]
    return float(np.mean(sims))


def strip_non_ascii(text):
    """Strip non-ASCII characters from text."""
    if not isinstance(text, str):
        text = str(text)
    return ''.join(ch for ch in text if ord(ch) < 128)

def query_llm(prompt, tools=None):
    """
    Query the OpenAI Responses API and return the cleaned text.
    """
    try:
        # Sanitize to valid UTF-8
        prompt_clean = sanitize_utf8(prompt)
        # (optionally strip non-ASCII if you want even stricter sanitization)
        prompt_clean = strip_non_ascii(prompt_clean)

        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt_clean,
            tools=tools or [],
            temperature=0.2
        )
        return extract_text_from_response(response)
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return {"error": str(e)}

def serialize_obj(o):
    """Custom serializer: use to_dict() if available, else str()."""
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)

import re

"""## 9. BaseAgent Class Definition
Implements methods to initialize agent, maintain memory, run sub-agents, and produce final synthesis.
"""

class BaseAgent:
    def __init__(self, ticker, cache_freshness_hours=CACHE_FRESHNESS_HOURS):
        self.ticker = ticker
        self.memory = []  # Simple memory list for context
        self.cache_freshness_hours = cache_freshness_hours
        try:
            stock_data, price_history, nifty_data = get_ticker_price_data(ticker)
            self.price_data = price_history
            self.add_to_memory(f"Retrieved price data for {ticker} with {len(stock_data)} data points.")
        except Exception as e:
            print(f"Error retrieving price data for {ticker}: {e}")
            self.price_data = None
            self.add_to_memory(f"Failed to retrieve price data for {ticker}: {e}")

    def add_to_memory(self, text):
        """Add text to the agent's memory."""
        if not isinstance(text, str):
            try:
                text = json.dumps(text, default=lambda o: str(o))
            except:
                text = str(text)
        self.memory.append(text)

    def get_context(self):
        """Get the current context from memory."""
        return "\n".join(self.memory[-5:])  # Last 5 messages for context

    def run_sequence(self, tasks, weights=None):
        """Run a sequence of sub-agent tasks."""
        results = {}
        for name, task_prompt in tasks.items():
            domain = name.lower().replace(" ", "_")
            cached_result = get_cache_entry(self.ticker, domain)
            if cached_result:
                print(f"Using cached data for {name} of {self.ticker}")
                results[name] = cached_result
                self.add_to_memory(f"{name} (Cached): {json.dumps(cached_result, indent=2)}")
                continue
            print(f"Fetching fresh data for {name} of {self.ticker}")
            price_context = ""
            if self.price_data:
                metrics = self.price_data.get('summary_metrics', {})
                price_context = f"""
                Price Data Context for {self.ticker}:
                - Current Price: ₹{metrics.get('current_price', 'N/A')}
                - YTD Return: {metrics.get('ytd_return', 'N/A'):.2f}%
                - 1Y Return: {metrics.get('return_1y', 'N/A'):.2f}%
                - 3Y Annualized Return: {metrics.get('return_3y', 'N/A'):.2f}%
                - 5Y Annualized Return: {metrics.get('return_5y', 'N/A'):.2f}%
                - Current RSI: {metrics.get('current_rsi', 'N/A'):.2f}
                - Current Volatility: {metrics.get('current_volatility', 'N/A'):.2f}%
                """
            context = self.get_context()
            task_with_ticker = task_prompt.replace("[TICKER]", self.ticker)
            controlled_prompt = format_prompt_with_hallucination_control(task_with_ticker)
            full_prompt = f"Context:\n{context}\nPrice Data:\n{price_context}\nTask: {controlled_prompt}"
            result = query_llm(full_prompt, tools=[{"type": "web_search"}])
            results[name] = result
            self.add_to_memory(f"{name}: {json.dumps(result, indent=2)}")
            update_cache_entry(self.ticker, domain, result)
            time.sleep(1)
        return results

    def final_synthesis(self, sub_agent_outputs, weights=None):
        """
        Generate a final synthesis based on sub-agent outputs.
        """
        cached_result = get_cache_entry(self.ticker, "final_synthesis")
        if cached_result:
            print(f"Using cached final synthesis for {self.ticker}")
            return cached_result
        print(f"Generating fresh final synthesis for {self.ticker}")
        outputs_json = json.dumps(sub_agent_outputs, indent=2, default=serialize_obj)
        cache = load_cache()
        timestamps = {}
        if self.ticker in cache:
            for domain in cache[self.ticker]:
                if domain != "ticker_last_updated" and "last_updated" in cache[self.ticker][domain]:
                    timestamps[domain] = cache[self.ticker][domain]["last_updated"]
        timestamps_json = json.dumps(timestamps, indent=2)
        price_context = ""
        if self.price_data:
            price_context = f"""
            Historical Price Data Summary for {self.ticker}:
            Current Price: ₹{self.price_data['summary_metrics']['current_price']:.2f}
            YTD Return: {self.price_data['summary_metrics']['ytd_return']:.2f}%
            1-Year Return: {self.price_data['summary_metrics']['return_1y']:.2f}%
            3-Year Annualized Return: {self.price_data['summary_metrics']['return_3y']:.2f}%
            5-Year Annualized Return: {self.price_data['summary_metrics']['return_5y']:.2f}%
            Current RSI: {self.price_data['summary_metrics']['current_rsi']:.2f}
            Current Volatility (20-day): {self.price_data['summary_metrics']['current_volatility']:.2f}%
            """
        weighting_instructions = ""
        if weights:
            weights_json = json.dumps(weights, indent=2)
            weighting_instructions = f"""
            Apply the following MANUAL WEIGHTS to each domain:
            {weights_json}
            """
        else:
            weighting_instructions = """
            Apply adaptive and temporal weighting based on recency and importance factors.
            1. More recent data weighted more heavily.
            2. Consider dominance_factor and confidence from each sub-agent.
            3. For long-term projections, stable fundamentals matter more.
            """
        synthesis_prompt = f"""
        Given the following sub-agent outputs for {self.ticker}:
        {outputs_json}
        And the following timestamps indicating recency:
        {timestamps_json}
        {price_context}
        {weighting_instructions}
        Generate a final structured JSON with keys:
        - short_term_range
        - medium_term_volatility
        - long_term_growth
        - risks
        - recommendation
        - rationale
        - final_research_summary
        Include disclaimer at end: "This outlook is based on publicly available web search results, historical price data, and LLM interpretation. For educational use only, not financial advice."
        Return ONLY the JSON object."
        """
        result = query_llm(synthesis_prompt, tools=[{"type": "web_search"}])
        cache = load_cache()
        if self.ticker not in cache:
            cache[self.ticker] = {}
        cache[self.ticker]["weights_used"] = weights or "dynamic weighting"
        save_cache(cache)
        update_cache_entry(self.ticker, "final_synthesis", result)
        return result

    def run(self, tasks, weights=None):
        print(f"\n--- Starting analysis for {self.ticker} ---\n")
        ticker_tasks = {name: prompt.replace("[TICKER]", self.ticker) for name, prompt in tasks.items()}
        sub_outputs = self.run_sequence(ticker_tasks)
        final_result = self.final_synthesis(sub_outputs, weights)
        print(f"\n--- Completed analysis for {self.ticker} ---\n")
        return {"sub_agent_outputs": sub_outputs, "final_synthesis": final_result, "price_data": self.price_data}

"""##10. Define Default Sub-Agent Tasks
Predefined tasks for Macro, Sector & Company, and Technical analysis.
"""

def get_default_tasks():
    """Get the default tasks for the sub-agents."""
    return {
        "Macro Analysis": """
        Analyze the current Indian macroeconomic factors affecting [TICKER]. Focus on:
        1. RBI monetary policies and interest rates
        2. Inflation trends in India
        3. GDP growth projections
        4. Global trade tensions affecting India
        5. Any recent government policies affecting the stock

        Conduct a web search to find the most recent information available.

        Return your analysis as a JSON object with structure:
        {
            "summary": "...",
            "risks": [...],
            "dominance_factor": "0-10",
            "confidence": "0-1",
            "disclaimers": "..."
        }
        """,

        "Sector and Company Analysis": """
        Analyze sector & company fundamentals for [TICKER]. Focus on:
        1. Industry/sector performance
        2. Financial health
        3. Valuation metrics
        4. Recent earnings
        5. Management & strategy
        6. Competitive positioning

        Conduct a web search for the latest info.

        Return JSON structured as:
        {
            "summary": "...",
            "risks": [...],
            "dominance_factor": "0-10",
            "confidence": "0-1",
            "disclaimers": "..."
        }
        """,

        "Technical Analysis": """
        Perform technical analysis for [TICKER]. Focus on:
        1. Price action & trends
        2. Key indicators (RSI, MACD, etc.)
        3. Support & resistance
        4. Volume analysis
        5. Short-term volatility
        6. Chart patterns

        Conduct a web search for latest info.

        Return JSON structured as:
        {
            "summary": "...",
            "risks": [...],
            "dominance_factor": "0-10",
            "confidence": "0-1",
            "disclaimers": "..."
        }
        """
    }

"""##11. Stock Outlook Function
Convenience wrapper `run_stock_outlook` to manage caching and BaseAgent execution.
"""

def run_stock_outlook(
    ticker: str,
    use_cache: bool = True,
    weights: Optional[Dict[str, float]] = None,
    start_date=None,
    end_date=None
) -> dict:
    """
    Returns the final_synthesis for the ticker using cache if available and fresh,
    otherwise triggers BaseAgent and refreshes the cache.
    """
    ticker = ticker.upper()
    cache = load_cache()

    # Ensure a cache entry exists for this ticker
    if ticker not in cache:
        cache[ticker] = {}

    # Price data cache management
    if not use_cache or not is_price_cache_fresh(ticker):
        try:
            print(f"Fetching fresh price data for {ticker}")
            get_ticker_price_data(ticker, start_date, end_date, force_refresh=True)
        except Exception as e:
            print(f"Error fetching price data for {ticker}: {e}")
            raise Exception(f"Failed to retrieve price data for {ticker}. Analysis cannot proceed without price data.")

    # If we have a fresh final_synthesis in cache, return it
    if use_cache and ticker in cache and "final_synthesis" in cache[ticker]:
        entry = cache[ticker]["final_synthesis"]
        if is_cache_fresh(entry.get("last_updated")):
            result = entry.get("result")

            # Display the cached plots
            try:
                price_cache = load_price_cache()
                if ticker in price_cache and "price_history" in price_cache[ticker]:
                    dates = pd.to_datetime(price_cache[ticker]["price_history"]["dates"])
                    stock_data = pd.DataFrame({
                        'Open': price_cache[ticker]["price_history"]["open"],
                        'High': price_cache[ticker]["price_history"]["high"],
                        'Low': price_cache[ticker]["price_history"]["low"],
                        'Close': price_cache[ticker]["price_history"]["close"],
                        'Volume': price_cache[ticker]["price_history"]["volume"],
                        'Daily_Return': price_cache[ticker]["price_history"]["indicators"]["daily_return"],
                        'Volatility': price_cache[ticker]["price_history"]["indicators"]["volatility"],
                        'MA_20': price_cache[ticker]["price_history"]["indicators"]["ma_20"],
                        'MA_50': price_cache[ticker]["price_history"]["indicators"]["ma_50"],
                        'MA_200': price_cache[ticker]["price_history"]["indicators"]["ma_200"],
                        'RSI': price_cache[ticker]["price_history"]["indicators"]["rsi"],
                        'MACD': price_cache[ticker]["price_history"]["indicators"]["macd"],
                        'Upper_Band': price_cache[ticker]["price_history"]["indicators"]["upper_band"],
                        'Lower_Band': price_cache[ticker]["price_history"]["indicators"]["lower_band"]
                    }, index=dates)

                    # nifty_data = yf.download("^NSEI", start=dates[0], end=dates[-1] + timedelta(days=1))
                    nifty_data = pd.read_csv(NIFTY_50_PATH)
                    print(f"Displaying charts for {ticker} based on cached data")
                    plot_price_charts(stock_data, ticker, nifty_data)
            except Exception as e:
                print(f"Error displaying charts from cached data: {e}")

            return cache

    # No fresh cache — generate result using BaseAgent
    print(f"No fresh cache found for {ticker}, running BaseAgent...")

    # Initialize agent (which will also fetch price data)
    agent = BaseAgent(ticker)
    tasks = get_default_tasks()
    # Pass weights to the run method if provided
    full_result = agent.run(tasks, weights)
    # Store in cache
    print(f'full result : {full_result}')
    cache[ticker]["final_synthesis"] = {
        "result": full_result["final_synthesis"],
        "last_updated": datetime.utcnow().isoformat()
    }
    print(f'Micro analysis data : {full_result["sub_agent_outputs"]["Macro Analysis"]}')
    cache[ticker]["macro_analysis"] = {
        "result": full_result["sub_agent_outputs"]["Macro Analysis"],
        "last_updated": datetime.utcnow().isoformat()
    }
    cache[ticker]["technical_analysis"] = {
        "result": full_result["sub_agent_outputs"]["Technical Analysis"],
        "last_updated": datetime.utcnow().isoformat()
    }
    cache[ticker]["sector_and_company_analysis"] = {
        "result": full_result["sub_agent_outputs"]["Sector and Company Analysis"],
        "last_updated": datetime.utcnow().isoformat()
    }
    cache[ticker]["ticker_last_updated"] = datetime.utcnow().isoformat()
    print(f"cache data : ", cache[ticker])
    save_cache(cache)

    # Plot the newly fetched data
    try:
        price_cache = load_price_cache()
        if ticker in price_cache and "price_history" in price_cache[ticker]:
            dates = pd.to_datetime(price_cache[ticker]["price_history"]["dates"])
            stock_data = pd.DataFrame({
                'Open': price_cache[ticker]["price_history"]["open"],
                'High': price_cache[ticker]["price_history"]["high"],
                'Low': price_cache[ticker]["price_history"]["low"],
                'Close': price_cache[ticker]["price_history"]["close"],
                'Volume': price_cache[ticker]["price_history"]["volume"],
                'Daily_Return': price_cache[ticker]["price_history"]["indicators"]["daily_return"],
                'Volatility': price_cache[ticker]["price_history"]["indicators"]["volatility"],
                'MA_20': price_cache[ticker]["price_history"]["indicators"]["ma_20"],
                'MA_50': price_cache[ticker]["price_history"]["indicators"]["ma_50"],
                'MA_200': price_cache[ticker]["price_history"]["indicators"]["ma_200"],
                'RSI': price_cache[ticker]["price_history"]["indicators"]["rsi"],
                'MACD': price_cache[ticker]["price_history"]["indicators"]["macd"],
                'Upper_Band': price_cache[ticker]["price_history"]["indicators"]["upper_band"],
                'Lower_Band': price_cache[ticker]["price_history"]["indicators"]["lower_band"]
            }, index=dates)

            nifty_data = yf.download("^NSEI", start=dates[0], end=dates[-1] + timedelta(days=1))
            print(f"Displaying charts for {ticker}")
            plot_price_charts(stock_data, ticker, nifty_data)
    except Exception as e:
        print(f"Error displaying charts: {e}")

    return cache

"""## 12. Similarity & Evaluation Functions
Functions to log runs, compute and log similarities, flag runs, and summarize evaluation logs.
"""

def log_run(ticker, prompt_id, agent, output_text, recommendation):
    """Log a run in the evaluation log."""
    log_path = EVALUATION_LOG_PATH
    run_id = hashlib.sha1((ticker + prompt_id + agent + output_text + str(time.time())).encode()).hexdigest()[:10]
    entry = {
        "run_id": run_id,
        "ticker": ticker,
        "prompt_id": prompt_id,
        "agent": agent,
        "recommendation": recommendation,
        "output_text": output_text,
        "timestamp": datetime.now().isoformat()
    }
    df_entry = pd.DataFrame([entry])
    try:
        df_existing = pd.read_csv(log_path)
        df_existing = pd.concat([df_existing, df_entry], ignore_index=True)
    except FileNotFoundError:
        df_existing = df_entry
    print(f'eval log path : {log_path}')
    df_existing.to_csv(log_path, index=False)
    print(f"Logged run: {run_id}")
    return run_id

def log_similarity_pairwise(ticker, field, run_id_1, run_id_2, similarity, threshold=0.9):
    """Log similarity between a pair of runs."""
    log_path = SIMILARITY_LOG_PATH

    entry = {
        "ticker": ticker,
        "field": field,
        "run_id_1": run_id_1,
        "run_id_2": run_id_2,
        "similarity": round(similarity, 4),
        "threshold": threshold,
        "below_threshold": similarity < threshold
    }
    df_entry = pd.DataFrame([entry])
    try:
        df_existing = pd.read_csv(log_path)
        df_existing = pd.concat([df_existing, df_entry], ignore_index=True)
    except FileNotFoundError:
        df_existing = df_entry
    df_existing.to_csv(log_path, index=False)

def flag_run(run_id: str, flag: str):
    """Flag a run in evaluation_log.csv."""
    log_path = EVALUATION_LOG_PATH
    try:
        df = pd.read_csv(log_path)
        df.loc[df["run_id"] == run_id, "flags"] = flag
        df.to_csv(log_path, index=False)
        print(f"⚠️ Flagged run {run_id} with flag: {flag}")
    except Exception as e:
        print(f"❌ Failed to flag run: {e}")

def compare_and_log_pairwise(ticker: str, outputs: List[dict], run_ids: List[str], fields: List[str], threshold: float = 0.9) -> Tuple[Dict[str, List[float]], bool]:
    """Compare and log similarity between multiple outputs."""
    similarities = {field: [] for field in fields}
    below_threshold = False
    for field in fields:
        for (i, j) in combinations(range(len(outputs)), 2):
            val1 = str(outputs[i].get(field, ""))
            val2 = str(outputs[j].get(field, ""))
            if val1 and val2:
                sim = average_cosine_similarity(val1, val2)
                similarities[field].append(sim)
                log_similarity_pairwise(
                    ticker=ticker,
                    field=field,
                    run_id_1=run_ids[i],
                    run_id_2=run_ids[j],
                    similarity=sim,
                    threshold=threshold
                )
                if sim < threshold:
                    below_threshold = True
                    print(f"⚠️ Flagged pair: {run_ids[i]} vs {run_ids[j]} | Field: {field} | Similarity: {sim:.4f} < {threshold}")
    return similarities, below_threshold

def predict(
    stock_name: str,
    n: int = 1,
    check_similarity: bool = False,
    similarity_threshold: float = 0.90,
    compare_fields: Optional[List[str]] = None,
    use_cache: bool = True,
    weights: Optional[Dict[str, float]] = None,
    start_date=None,
    end_date=None
) -> Tuple[str, str, str]:
    """
    Run stock analysis with optional similarity checking.
    Returns: (results_markdown, evaluation_log_path, similarity_log_path)
    """
    if check_similarity:
        if n <= 1:
            return ("❌ Error: Similarity check requires number of runs (n) > 1.", None, None)
        if use_cache:
            return ("❌ Error: Please disable 'Use Cache if Available' when running similarity check.", None, None)
    else:
        if n > 1:
            return ("❌ Error: When similarity check is disabled, Number of Runs (n) should be 1.", None, None)
    stock_name = stock_name.upper()
    outputs, run_ids = [], []
    for i in range(n):
        prompt_id = f"predict_v2_run{i+1}"
        result = run_stock_outlook(
            stock_name,
            use_cache=use_cache,
            weights=weights,
            start_date=start_date,
            end_date=end_date
        )
        if isinstance(result, dict) and 'result' in result:
            result = result['result']
        if isinstance(result, str) and result.strip().startswith('```json'):
            result = json.loads(result.strip().replace('```json', '').replace('```','').strip())
        outputs.append(result)
        text = result.get('final_research_summary', json.dumps(result))
        rec = result.get('recommendation', 'UNKNOWN')
        run_id = log_run(
            ticker=stock_name,
            prompt_id=prompt_id,
            agent="Final Synthesis",
            output_text=text,
            recommendation=rec
        )
        run_ids.append(run_id)
    similarity_log = ""
    if check_similarity and n > 1:
        compare_fields = compare_fields or ["final_research_summary"]
        similarities, below_threshold = compare_and_log_pairwise(
            ticker=stock_name,
            outputs=outputs,
            run_ids=run_ids,
            fields=compare_fields,
            threshold=similarity_threshold
        )
        similarity_log += "\n\n🔍 **Field-Level Similarities**:\n"
        for field, sims in similarities.items():
            if sims:
                avg_sim = np.mean(sims)
                flag = "⚠️ Below threshold!" if avg_sim < similarity_threshold else "✅"
                similarity_log += f"- `{field}`: **{avg_sim:.4f}** {flag}\n"
            else:
                similarity_log += f"- `{field}`: No valid pairs.\n"
    final = outputs[-1]
    display_parts = []
    if 'short_term_range' in final:
        display_parts.append(f"**Short-Term Price Range**: {final['short_term_range']}")
    if 'medium_term_volatility' in final:
        display_parts.append(f"**Medium-Term Volatility**: {final['medium_term_volatility']}")
    if 'long_term_growth' in final:
        display_parts.append(f"**Long-Term Growth Estimate**: {final['long_term_growth']}")
    if 'recommendation' in final:
        display_parts.append(f"**Recommendation**: `{final['recommendation']}`")
    if 'rationale' in final:
        display_parts.append(f"**Rationale**: {final['rationale']}")
    if 'final_research_summary' in final:
        display_parts.append(f"**Summary**:\n{final['final_research_summary']}")
    if weights:
        display_parts.append(f"**Manual Weights Applied**: {json.dumps(weights, indent=2)}")

    return "\n\n".join(display_parts) + similarity_log, EVALUATION_LOG_PATH, SIMILARITY_LOG_PATH

"""## 14. Batch Runner & Log Summaries
Functions to run batch predictions, summarize evaluation logs, and initialize/clear logs.
"""

def run_batch_predict(
    tickers: List[str],
    n: int = 3,
    check_similarity: bool = True,
    similarity_threshold: float = 0.9,
    compare_fields: Optional[List[str]] = None,
    use_cache: bool = True,
    weights: Optional[Dict[str, float]] = None
):
    """Run prediction on multiple tickers as a batch."""
    for ticker in tickers:
        try:
            md, eval_log, sim_log = predict(
                ticker,
                n=n,
                check_similarity=check_similarity,
                similarity_threshold=similarity_threshold,
                compare_fields=compare_fields,
                use_cache=use_cache,
                weights=weights
            )
            print(md)
        except Exception as e:
            print(f"❌ Failed for {ticker}: {e}")

def summarize_evaluation_logs(show_only_flagged=False, last_n_runs=None):
    """
    Summary of past runs from evaluation_log.csv.
    """
    try:
        df = pd.read_csv(EVALUATION_LOG_PATH)
    except FileNotFoundError:
        print("⚠️ Log file not found.")
        return
    if last_n_runs is not None and last_n_runs < len(df):
        df = df.tail(last_n_runs)
    print("📊 Summary of Logged Evaluations\n")
    if "recommendation" in df.columns and "ticker" in df.columns:
        summary = df.groupby(["ticker", "recommendation"]).size().unstack(fill_value=0)
        display(summary)
    else:
        print("⚠️ Missing required columns in the log.")
    if "flags" in df.columns:
        if show_only_flagged:
            df = df[df["flags"].notnull() & (df["flags"] != "")]
            print("\n📌 Showing only flagged runs.\n")
        flagged = df[df["flags"].notnull() & (df["flags"] != "")]
        if not flagged.empty:
            print("\n🚩 Runs flagged with issues:")
            cols_to_show = [c for c in ["run_id", "ticker", "prompt_id", "agent", "recommendation", "flags"] if c in df.columns]
            display(flagged[cols_to_show])
            issues = []
            for i, row in flagged.iterrows():
                for field in str(row["flags"]).split(","):
                    issues.append({"ticker": row["ticker"], "field": field})
            if issues:
                df_issues = pd.DataFrame(issues)
                summary_issues = df_issues.groupby(["ticker", "field"]).size().reset_index(name="low_sim_pairs")
                print("\n🧠 Similarity Issues by Ticker & Field:")
                display(summary_issues)
        else:
            print("\n✅ No runs flagged with similarity issues.")

def initialize_logs():
    """Initialize log files with headers if they don't exist."""
    if not os.path.exists(EVALUATION_LOG_PATH):
        pd.DataFrame(columns=["run_id", "ticker", "prompt_id", "agent", "recommendation", "output_text", "timestamp"]).to_csv(EVALUATION_LOG_PATH, index=False)
    if not os.path.exists(SIMILARITY_LOG_PATH):
        pd.DataFrame(columns=["ticker", "field", "run_id_1", "run_id_2", "similarity", "below_threshold", "threshold"]).to_csv(SIMILARITY_LOG_PATH, index=False)

def clear_cache():
    """Clear the entire LLM result cache."""
    if pathlib.Path(CACHE_FILE_PATH).exists():
        os.remove(CACHE_FILE_PATH)
        print("LLM cache cleared successfully.")
    else:
        print("No LLM cache file found.")

def clear_price_cache():
    """Clear the entire price data cache."""
    if pathlib.Path(PRICE_CACHE_PATH).exists():
        os.remove(PRICE_CACHE_PATH)
        print("Price cache cleared successfully.")
    else:
        print("No price cache file found.")

def clear_ticker_cache(ticker):
    """Clear the cache for a specific ticker."""
    ticker = ticker.upper()
    cache = load_cache()
    if ticker in cache:
        del cache[ticker]
        save_cache(cache)
        print(f"LLM cache for {ticker} cleared successfully.")
    else:
        print(f"No LLM cache found for {ticker}.")
    price_cache = load_price_cache()
    if ticker in price_cache:
        del price_cache[ticker]
        save_price_cache(price_cache)
        print(f"Price cache for {ticker} cleared successfully.")
    else:
        print(f"No price cache found for {ticker}.")

"""Main vs Evaluation tabs separated"""

# Display the cached plots
import plotly.io as pio

def display_plots_from_price_cache(ticker):
            ticker = ticker.upper()
            try:
                price_cache = load_price_cache()
                if ticker in price_cache and "price_history" in price_cache[ticker]:
                    dates = pd.to_datetime(price_cache[ticker]["price_history"]["dates"])

                    stock_data = pd.DataFrame({
                        'Open': price_cache[ticker]["price_history"]["open"],
                        'High': price_cache[ticker]["price_history"]["high"],
                        'Low': price_cache[ticker]["price_history"]["low"],
                        'Close': price_cache[ticker]["price_history"]["close"],
                        'Volume': price_cache[ticker]["price_history"]["volume"],
                        'Daily_Return': price_cache[ticker]["price_history"]["indicators"]["daily_return"],
                        'Volatility': price_cache[ticker]["price_history"]["indicators"]["volatility"],
                        'MA_20': price_cache[ticker]["price_history"]["indicators"]["ma_20"],
                        'MA_50': price_cache[ticker]["price_history"]["indicators"]["ma_50"],
                        'MA_200': price_cache[ticker]["price_history"]["indicators"]["ma_200"],
                        'RSI': price_cache[ticker]["price_history"]["indicators"]["rsi"],
                        'MACD': price_cache[ticker]["price_history"]["indicators"]["macd"],
                        'Upper_Band': price_cache[ticker]["price_history"]["indicators"]["upper_band"],
                        'Lower_Band': price_cache[ticker]["price_history"]["indicators"]["lower_band"]
                    }, index=dates)
                    # try:
                    #     nifty_data = yf.download("^NSEI", start=dates[0], end=dates[-1] + timedelta(days=1))
                    # except Exception as e:
                    nifty_data = pd.read_csv(NIFTY_50_PATH)
                    # nifty_data = pd.read_csv("../../nifty_data_cleaned.csv")
                    # print(f"Displaying charts for {ticker} based on cached data")
                    plots = plot_price_charts(stock_data, ticker)
                    # return plots["price_chart"], plots["bollinger_chart"], plots["rsi_chart"], plots["volatility_chart"]
                    return {
                            "price_chart": pio.to_json(plots["price_chart"]),
                            "bollinger_chart": pio.to_json(plots["bollinger_chart"]),
                            "rsi_chart": pio.to_json(plots["rsi_chart"]),
                            "volatility_chart": pio.to_json(plots["volatility_chart"]),
                            "macd_chart": pio.to_json(plots["macd_chart"])
                        }
            except Exception as e:
                print(f"Error displaying charts from cached data: {e}")
                return None, None, None, None, None
# display_plots_from_price_cache("TCS.NS")

import pandas as pd
import json
import re


def clean_json_string(raw_string):
    """Clean markdown artifacts and parse JSON if possible."""
    if isinstance(raw_string, dict):
        return raw_string  # Already a dictionary, no need to clean
    if not raw_string:
        return {}
    cleaned = raw_string.replace('```json', '').replace('```', '').strip()
    if not cleaned:
        return {}
    try:
        # print(f'Cleaned data : {cleaned}')
        return json.loads(cleaned)
    except Exception:
        # print(f"JSON parse failed. Attempting to extract bolded sections. {Exception}")
        return extract_bolded_sections(raw_string)

def extract_bolded_sections(content):
    """Extract **bolded** sections if JSON parse fails."""
    result = {}
    pattern = r'\*\*(.*?)\*\*:?\s*([\s\S]*?)(?=(\n\s*\*\*|$))'
    matches = re.findall(pattern, content)
    # print(f'Matches : {matches}')
    for key, value, _ in matches:
        result[key.strip()] = value.strip()
    return result

analysis_tabs = [
    'macro_analysis',
    'sector_and_company_analysis',
    'technical_analysis',
    'final_synthesis'
]

import json
import re

def extract_json_from_string(text):
    """Extract JSON object from a string, if present."""
    if not isinstance(text, str):
        return None
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None

def get_analysis_content(analysis_data):
    """Return analysis content dynamically with embedded JSON handled."""
    raw_result = analysis_data.get('result', '')
    last_updated = analysis_data.get('last_updated', '')
    content = ""

    # Start with last updated
    if last_updated:
        content += f"**Last Updated**: {last_updated}\n\n"

    parsed = {}
    if isinstance(raw_result, str):
        embedded_json = extract_json_from_string(raw_result)
        if embedded_json:
            parsed.update(embedded_json)
            pre_text = raw_result.split('{')[0].strip()
            if pre_text:
                parsed["description"] = pre_text
        else:
            parsed["description"] = raw_result.strip()

    elif isinstance(raw_result, dict):
        parsed.update(raw_result)
        for key, value in list(parsed.items()):
            if isinstance(value, str):
                embedded_json = extract_json_from_string(value)
                if embedded_json:
                    del parsed[key]  # remove the old string value
                    parsed.update(embedded_json)  # merge extracted JSON

    # Now format everything in parsed
    for key, value in parsed.items():
        title = key.replace('_', ' ').title()
        if isinstance(value, list):
            content += f"**{title}**:\n"
            for item in value:
                content += f"- {item}\n"
            content += "\n"
        elif isinstance(value, dict):
            content += f"**{title}**:\n"
            for sub_key, sub_val in value.items():
                content += f"- {sub_key.title()}: {sub_val}\n"
            content += "\n"
        else:
            content += f"**{title}**: {value}\n\n"

    return content.strip() if content.strip() else "**No data available.**"

"""Multiple tabs for various analyses"""

def predict_2(
    stock_name,
    analysis_type="final_synthesis",
    n=1,
    check_similarity=False,
    similarity_threshold=0.9,
    compare_fields=False,
    use_cache=True,
    show_similarity_summary=False,
    weights=None,
    start_date=None,
    end_date=None
    ):
    """
    Run stock analysis with optional similarity checking.
    Returns: (results_markdown, evaluation_log_path, similarity_log_path)
    """
    if check_similarity:
        if n <= 1:
            print("❌ Error: Similarity check requires number of runs (n) > 1.", None, None)
        if use_cache:
            print("❌ Error: Please disable 'Use Cache if Available' when running similarity check.", None, None)
    else:
        if n > 1:
            print("❌ Error: When similarity check is disabled, Number of Runs (n) should be 1.", None, None)

    stock_name = stock_name.upper()
    outputs, run_ids = [], []

    for i in range(n):
        prompt_id = f"predict_v2_run{i+1}"
        # print("Weights : ", weights)
        result = run_stock_outlook(
            stock_name,
            use_cache=use_cache,
            weights=weights,
            start_date=start_date,
            end_date=end_date
        )
        try:
          result[stock_name][analysis_type]["result"] = clean_json_string(result[stock_name][analysis_type]["result"])
        except:
          return "No data found"
        outputs.append(result)
        text = result[stock_name][analysis_type]["result"].get('final_research_summary')
        rec = result[stock_name][analysis_type]["result"].get('recommendation', 'UNKNOWN')
        run_id = log_run(
            ticker=stock_name,
            prompt_id=prompt_id,
            agent="Final Synthesis",
            output_text=text,
            recommendation=rec
        )
        run_ids.append(run_id)

    similarity_log = ""
    if check_similarity and n > 1:
        compare_fields = compare_fields or ["final_research_summary"]
        similarities, below_threshold = compare_and_log_pairwise(
            ticker=stock_name,
            outputs=outputs,
            run_ids=run_ids,
            fields=compare_fields,
            threshold=similarity_threshold
        )
        similarity_log += "Field-Level Similarities:"
        for field, sims in similarities.items():
            if sims:
                avg_sim = np.mean(sims)
                flag = "Below threshold!" if avg_sim < similarity_threshold else " "
                similarity_log += f"- `{field}`: {avg_sim:.4f}  {flag}\n"
            else:
                similarity_log += f"- `{field}`: No valid pairs.\n"

    return outputs, EVALUATION_LOG_PATH, SIMILARITY_LOG_PATH

# predict_2("TCS.NS")

def update_all_tabs_2(results, ticker):
  outputs = []
  for analysis_type in analysis_tabs:
      outputs.append(get_analysis_content(results[ticker][analysis_type]))
  return outputs


FIELDS_TO_COMPARE = ["short_term_range", "long_term_growth", "final_research_summary"]

def gradio_predict(stock_ticker, analysis_type, n_runs, check_similarity, similarity_threshold,
                   compare_fields, use_cache, show_similarity_summary, add_weights,
                   macro_weight_val, sector_weight_val, tech_weight_val
                   ):
    try:
      initialize_logs()
      if add_weights == False:
          custom_weights = None
      else:
          custom_weights = {
                "Macro Analysis": float(macro_weight_val),
                "Sector and Company Analysis": float(sector_weight_val),
                "Technical Analysis": float(tech_weight_val)
            }
      md, eval_log_path, sim_log_path = predict_2(
          stock_name=stock_ticker.upper().strip(),
          n=n_runs,
          check_similarity=check_similarity,
          similarity_threshold=similarity_threshold,
          compare_fields=compare_fields or [],
          use_cache=use_cache,
          weights=custom_weights
      )

      stock_results = update_all_tabs_2(md[0], stock_ticker )
      plots = display_plots_from_price_cache(stock_ticker)

    #   print(f"Plots : {plots}")
      return (*stock_results, eval_log_path, sim_log_path, plots["price_chart"], plots["bollinger_chart"], plots["rsi_chart"], plots["volatility_chart"], plots["macd_chart"] )
    #   return (*stock_results, eval_log_path, sim_log_path )
    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        print(error_msg)
        # return {tab: error_msg for tab in analysis_tabs}, None, None
        return [error_msg] + [None, None, None, None, None, None]
