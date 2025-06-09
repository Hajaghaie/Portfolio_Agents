"""
Data fetching utilities through yfinance and tavily. Yahoo! Finance package is used to fetch the financial data and tavily is used to fetch the news. 
Make sure to set the api keys in the config.py file. 
"""
import logging
import time
import datetime
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from .config import BENCHMARK_TICKER


def fetch_financial_data(tickers: List[str], period: str = "1y", start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Fetches historical stock data using yfinance."""
    fetch_method_info = f"Start: {start_date}, End: {end_date}" if start_date and end_date else f"Period: {period}"
    logging.info(f"Fetching Financial Data for: {tickers} ({fetch_method_info})")
    data = {}
    if not tickers:
        logging.warning("No tickers provided for fetching data.")
        return data
    try:
        for ticker in tickers:
            ticker = ticker.upper().strip()
            tkr = yf.Ticker(ticker)
            hist = pd.DataFrame()
            if start_date and end_date:
                hist = tkr.history(start=start_date, end=end_date)
                if hist.empty:
                    logging.warning(f"No history data found for ticker: {ticker} between {start_date} and {end_date}.")
            else:
                hist = tkr.history(period=period)
                if hist.empty:
                    logging.warning(f"No history data found for ticker: {ticker} for period: {period}.")
            if hist.empty:
                try:
                    info = tkr.info
                    if not info or ('symbol' not in info and 'longName' not in info):
                        logging.warning(f"Ticker {ticker} might be invalid or delisted (no info found). Skipping.")
                        continue
                    else:
                        logging.info(f"Info found for {ticker}, but no history for the period. Skipping.")
                        continue
                except Exception as info_e:
                    logging.warning(f"Could not verify ticker {ticker} info: {info_e}. Skipping.")
                    continue
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist.columns = hist.columns.str.lower()
            if 'close' not in hist.columns:
                logging.warning(f"'close' column missing for {ticker}. Skipping.")
                continue
            data[ticker] = hist
            logging.info(f"Successfully fetched data for {ticker}.")
            time.sleep(0.5)
        logging.info(f"Successfully fetched data for: {list(data.keys())}")
        if not data:
            logging.warning("Failed to fetch valid data for ALL requested tickers.")
        return data
    except Exception as e:
        logging.error(f"Error during financial data fetching process: {e}")
        return data


def fetch_market_news(state) -> Dict:
    """Fetches relevant market news using Tavily based on user profile."""
    from langchain_community.tools.tavily_search.tool import TavilySearchResults
    from .config import TAVILY_API_KEY
    tavily_tool = TavilySearchResults(max_results=3) if TAVILY_API_KEY else None
    logging.info("Fetching Market News...")
    if not tavily_tool:
        logging.warning("Tavily tool not available. Skipping market news.")
        return {"market_news": "N/A (Tool not configured)"}
    user_profile = state.get('user_profile', {})
    query = f"Recent market news relevant to investment goal '{user_profile.get('goal', 'general investing')}' with risk tolerance '{user_profile.get('risk_tolerance', 'medium')}'"
    if state.get('asset_universe'):
        query += f" focusing on potential assets like {', '.join(state['asset_universe'])}"
    try:
        news_results = tavily_tool.invoke({"query": query})
        formatted_news = "\n".join([f"- {item['content']}" for item in news_results]) if news_results else "No specific news found."
        logging.info(f"News Query: {query}\nNews Found: {formatted_news[:200]}...")
        return {"market_news": formatted_news}
    except Exception as e:
        logging.error(f"Error fetching market news: {e}")
        return {"market_news": f"Failed to fetch news: {e}"}


def fetch_data_node(state) -> Dict:
    """Node to call the financial data fetching function, ensuring benchmark data is included."""
    asset_universe = state.get('asset_universe')
    user_profile = state.get('user_profile', {})
    if not asset_universe:
        asset_universe = user_profile.get('suggested_assets')
        if not asset_universe:
            logging.error("No asset universe identified to fetch data for.")
            return {"error_message": "Cannot fetch data: No assets were identified in the user request or previous steps."}
    benchmark_ticker = BENCHMARK_TICKER
    tickers_to_fetch = list(asset_universe)
    if benchmark_ticker not in [t.upper() for t in tickers_to_fetch]:
        tickers_to_fetch.append(benchmark_ticker)
        logging.info(f"Added benchmark ticker {benchmark_ticker} for Beta calculation.")
    start_date_str = user_profile.get('start_date')
    end_date_str = user_profile.get('end_date')
    data = {}
    data_fetched_with_range = False
    if start_date_str and end_date_str:
        try:
            datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
            datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
            logging.info(f"Using specific date range for data fetching: {start_date_str} to {end_date_str}")
            data = fetch_financial_data(tickers=tickers_to_fetch, start_date=start_date_str, end_date=end_date_str)
            data_fetched_with_range = True
        except ValueError:
            logging.warning(f"Invalid start_date ('{start_date_str}') or end_date ('{end_date_str}') format. Falling back to time_horizon-based period.")
    if not data_fetched_with_range:
        time_horizon_input = user_profile.get('time_horizon')
        period = "5y"
        years = None
        if isinstance(time_horizon_input, int):
            years = time_horizon_input
        elif isinstance(time_horizon_input, str):
            if 'year' in time_horizon_input.lower():
                try:
                    relevant_part = time_horizon_input.lower().split('year')[0].strip()
                    if '-' in relevant_part:
                        years_str = relevant_part.split('-')[-1].strip()
                    else:
                        years_str = relevant_part.split()[-1].strip()
                    years = int(years_str)
                except Exception as e:
                    logging.warning(f"Could not parse years from time_horizon string '{time_horizon_input}' (Error: {e}). Using default period {period}.")
            elif 'long-term' in time_horizon_input.lower():
                years = 10
            elif 'medium-term' in time_horizon_input.lower():
                years = 5
            elif 'short-term' in time_horizon_input.lower():
                years = 1
            else:
                try:
                    years = int(time_horizon_input.strip())
                    logging.info(f"Parsed time_horizon string '{time_horizon_input}' as {years} years.")
                except ValueError:
                    logging.warning(f"Could not parse time_horizon string '{time_horizon_input}' as years. Using default period {period}.")
        if years is not None and years > 0:
            period = f"{years}y"
            logging.info(f"Setting data fetching period to: {period} based on {years} years.")
        else:
            logging.info(f"Could not determine positive years from time_horizon '{time_horizon_input}'. Using default period {period}.")
        logging.info(f"Using period-based data fetching: {period}")
        data = fetch_financial_data(tickers=tickers_to_fetch, period=period)
    if benchmark_ticker not in data or data[benchmark_ticker].empty:
        logging.error(f"Failed to fetch benchmark data ({benchmark_ticker}). Manual Beta calculation will be disabled.")
    if not data:
        return {"financial_data": data, "error_message": f"Failed to fetch ANY valid data for the identified assets or benchmark. Cannot proceed."}
    valid_assets_in_universe = [t for t in asset_universe if t.upper() in data and t.upper() != benchmark_ticker]
    original_count = len(asset_universe)
    valid_count = len(valid_assets_in_universe)
    if valid_count < original_count:
        removed_assets = set(a.upper() for a in asset_universe) - set(data.keys()) - {benchmark_ticker}
        logging.warning(f"Only fetched data for {valid_count} out of {original_count} requested assets (excluding benchmark). Missing: {removed_assets if removed_assets else 'None'}")
    return {"financial_data": data, "asset_universe": valid_assets_in_universe} 