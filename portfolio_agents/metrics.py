"""
This file contains the functions to calculate the financial metrics. as mentioned comprehensiveley in the paper, the
metrics are calculated using the CAPM model and the SMAs. Also, the manual beta calculation is included if the data is not available in yfinance. 
(the yfinance package cannot get the beta value for certain tickers, like ETFs, so we use the manual beta calculation for those tickers based
on the historical data and benchmark returns)
The validation function is used to validate the portfolio and the metrics. 
"""
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf


def calculate_financial_metrics(data: Dict[str, pd.DataFrame], portfolio: Optional[Dict[str, float]] = None) -> Dict:
    """Calculates financial metrics, including CAPM (with manual Beta calc) and SMAs."""
    logging.info("Calculating Financial Metrics (including CAPM & SMAs with manual Beta)")
    metrics_results = {}
    if not data:
        logging.warning("No financial data provided for metric calculation.")
        return {"error": "No financial data available"}

    RISK_FREE_RATE = 0.045
    EXPECTED_MARKET_RETURN = 0.09
    BENCHMARK_TICKER = "^GSPC"
    MIN_PERIODS_FOR_BETA = 60
    logging.info(f"Using CAPM assumptions: Rf={RISK_FREE_RATE:.1%}, E(Rm)={EXPECTED_MARKET_RETURN:.1%}")
    logging.info(f"Using Benchmark: {BENCHMARK_TICKER}")

    benchmark_data = data.get(BENCHMARK_TICKER)
    benchmark_returns = None
    if benchmark_data is not None and 'close' in benchmark_data.columns:
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        if benchmark_returns.empty:
            logging.warning(f"Could not calculate returns for benchmark {BENCHMARK_TICKER}. Manual Beta calculation disabled.")
            benchmark_returns = None
    else:
        logging.warning(f"Benchmark data ({BENCHMARK_TICKER}) missing or invalid. Manual Beta calculation disabled.")

    asset_betas = {}
    logging.info("Fetching/Calculating betas...")
    asset_tickers = [t for t in data.keys() if t != BENCHMARK_TICKER]
    for ticker in asset_tickers:
        beta = None
        source = "N/A"
        try:
            tkr_info = None
            try:
                tkr_info = yf.Ticker(ticker).info
            except Exception as e:
                logging.warning(f"Could not fetch info for {ticker}: {e}")
            beta_info = tkr_info.get('beta') if tkr_info else None
            if beta_info is not None:
                beta = float(beta_info)
                source = "yf.info"
                logging.info(f"  {ticker}: Beta = {beta:.2f} (Source: {source})")
            elif benchmark_returns is not None:
                asset_df = data.get(ticker)
                if asset_df is not None and 'close' in asset_df.columns:
                    asset_returns = asset_df['close'].pct_change().dropna()
                    if not asset_returns.empty:
                        aligned_df = pd.merge(asset_returns.rename('asset'), benchmark_returns.rename('benchmark'), left_index=True, right_index=True, how='inner')
                        if len(aligned_df) >= MIN_PERIODS_FOR_BETA:
                            cov_matrix = aligned_df.cov()
                            covariance = cov_matrix.loc['asset', 'benchmark']
                            benchmark_variance = aligned_df['benchmark'].var()
                            if benchmark_variance != 0:
                                calculated_beta = covariance / benchmark_variance
                                beta = float(calculated_beta)
                                source = "Calculated"
                                logging.info(f"  {ticker}: Beta = {beta:.2f} (Source: {source})")
                            else:
                                logging.warning(f"  {ticker}: Benchmark variance is zero, cannot calculate Beta.")
                        else:
                            logging.warning(f"  {ticker}: Insufficient overlapping data ({len(aligned_df)} days) with benchmark to calculate Beta.")
                    else:
                        logging.warning(f"  {ticker}: No return data available to calculate Beta.")
                else:
                    logging.warning(f"  {ticker}: Price data missing, cannot calculate Beta.")
            if beta is None:
                logging.warning(f"  {ticker}: Beta not available (from yf.info or calculation)." )
        except Exception as e:
            logging.error(f"  Error processing beta for {ticker}: {e}")
            beta = None
        asset_betas[ticker] = beta

    try:
        if portfolio:
            logging.info(f"Calculating metrics for portfolio: {list(portfolio.keys())}")
            portfolio_df = pd.DataFrame()
            valid_tickers_in_portfolio = []
            weights_list = []
            individual_capm_returns = {}
            aligned_data = {}
            common_index = None
            for ticker, weight in portfolio.items():
                ticker = ticker.upper()
                if ticker in asset_tickers and ticker in data and not data[ticker].empty and 'close' in data[ticker].columns:
                    df_ticker = data[ticker][['close']].copy()
                    df_ticker.rename(columns={'close': ticker}, inplace=True)
                    aligned_data[ticker] = df_ticker
                    if common_index is None:
                        common_index = df_ticker.index
                    else:
                        common_index = common_index.intersection(df_ticker.index)
                else:
                    logging.warning(f"Data missing or invalid for ticker {ticker} in portfolio. Excluding from calculation.")
            if not aligned_data or common_index is None or common_index.empty:
                return {"error": "No valid/aligned data found for tickers in the portfolio."}
            portfolio_df = pd.DataFrame(index=common_index)
            for ticker, df_ticker in aligned_data.items():
                portfolio_df = pd.merge(portfolio_df, df_ticker.loc[common_index], left_index=True, right_index=True, how='inner')
                valid_tickers_in_portfolio.append(ticker)
                weights_list.append(portfolio[ticker])
            if portfolio_df.empty:
                return {"error": "Could not align data for portfolio calculation."}
            valid_weight_sum = sum(weights_list)
            if abs(valid_weight_sum) < 1e-6:
                return {"error": "Portfolio weights for available assets sum to zero."}
            normalized_weights = [w / valid_weight_sum for w in weights_list]
            returns = portfolio_df.pct_change().dropna()
            if returns.empty:
                return {"error": "Could not calculate returns (e.g., only one data point)."}
            portfolio_return = (returns * normalized_weights).sum(axis=1)
            cumulative_return = (1 + portfolio_return).cumprod() - 1
            total_return = cumulative_return.iloc[-1] if not cumulative_return.empty else 0.0
            trading_days_per_year = 252
            num_days = len(portfolio_return)
            if num_days < 5:
                annualized_return = total_return
                volatility = portfolio_return.std() * (trading_days_per_year ** 0.5) if num_days > 1 else 0.0
            else:
                annualized_return = (1 + total_return) ** (trading_days_per_year / num_days) - 1
                volatility = portfolio_return.std() * (trading_days_per_year ** 0.5)
            sharpe_ratio = annualized_return / volatility if volatility != 0 else 0.0
            rolling_max = (1 + cumulative_return).cummax()
            daily_drawdown = (1 + cumulative_return) / rolling_max - 1.0
            max_drawdown = daily_drawdown.min() if not daily_drawdown.empty else 0.0
            portfolio_expected_return_capm = None
            weighted_capm_sum = 0.0
            weight_sum_for_capm = 0.0
            for ticker in valid_tickers_in_portfolio:
                beta = asset_betas.get(ticker)
                if beta is not None:
                    individual_capm_returns[ticker] = RISK_FREE_RATE + beta * (EXPECTED_MARKET_RETURN - RISK_FREE_RATE)
                else:
                    individual_capm_returns[ticker] = None
            for i, ticker in enumerate(valid_tickers_in_portfolio):
                capm_ret = individual_capm_returns.get(ticker)
                if capm_ret is not None:
                    weight = normalized_weights[i]
                    weighted_capm_sum += weight * capm_ret
                    weight_sum_for_capm += weight
            if weight_sum_for_capm > 1e-6:
                portfolio_expected_return_capm = weighted_capm_sum / weight_sum_for_capm
            else:
                logging.warning("Could not calculate portfolio CAPM (no valid betas/weights).")
            portfolio_value = (1 + portfolio_return).cumprod()
            portfolio_sma_50 = None
            portfolio_sma_200 = None
            portfolio_momentum_outlook = "Neutral"
            if len(portfolio_value) >= 50:
                portfolio_sma_50 = portfolio_value.rolling(window=50).mean().iloc[-1]
            if len(portfolio_value) >= 200:
                portfolio_sma_200 = portfolio_value.rolling(window=200).mean().iloc[-1]
            if portfolio_sma_50 is not None and portfolio_sma_200 is not None:
                if portfolio_sma_50 > portfolio_sma_200:
                    portfolio_momentum_outlook = "Bullish (50d > 200d SMA)"
                else:
                    portfolio_momentum_outlook = "Bearish (50d < 200d SMA)"
            elif portfolio_sma_50 is not None:
                portfolio_momentum_outlook = "Neutral (Insufficient history for 200d SMA)"
            metrics_results['portfolio'] = {
                'total_return': round(total_return, 4),
                'annualized_return': round(annualized_return, 4),
                'annualized_volatility': round(volatility, 4),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 4),
                'expected_return_capm': round(portfolio_expected_return_capm, 4) if portfolio_expected_return_capm is not None else None,
                'portfolio_sma_50': round(portfolio_sma_50, 2) if portfolio_sma_50 is not None else None,
                'portfolio_sma_200': round(portfolio_sma_200, 2) if portfolio_sma_200 is not None else None,
                'portfolio_momentum_outlook': portfolio_momentum_outlook,
                'included_assets': valid_tickers_in_portfolio,
                'period_days': num_days,
                'original_weight_sum': round(sum(portfolio.values()), 4),
                'included_weight_sum': round(valid_weight_sum, 4),
                'capm_calculation_weight_coverage': round(weight_sum_for_capm, 4)
            }
        logging.info("Calculating metrics for individual assets...")
        for ticker in asset_tickers:
            df = data.get(ticker)
            if df is None or df.empty or 'close' not in df.columns:
                logging.warning(f"Skipping individual metrics for {ticker}: Empty or no close price.")
                continue
            returns = df['close'].pct_change().dropna()
            if returns.empty or len(returns) < 2:
                logging.warning(f"Skipping individual metrics for {ticker}: Not enough return data ({len(returns)} points).")
                continue
            total_return = (1 + returns).prod() - 1
            num_days = len(returns)
            trading_days_per_year = 252
            if num_days < 5:
                annualized_return = total_return
                volatility = returns.std() * (trading_days_per_year ** 0.5)
            else:
                annualized_return = (1 + total_return) ** (trading_days_per_year / num_days) - 1
                volatility = returns.std() * (trading_days_per_year ** 0.5)
            sharpe_ratio = annualized_return / volatility if volatility != 0 else 0.0
            cumulative_return = (1 + returns).cumprod() - 1
            rolling_max = (1 + cumulative_return).cummax()
            daily_drawdown = (1 + cumulative_return) / rolling_max - 1.0
            max_drawdown = daily_drawdown.min() if not daily_drawdown.empty else 0.0
            beta = asset_betas.get(ticker)
            expected_return_capm = None
            if beta is not None:
                expected_return_capm = RISK_FREE_RATE + beta * (EXPECTED_MARKET_RETURN - RISK_FREE_RATE)
            sma_50 = None
            sma_200 = None
            close_prices = df['close']
            if len(close_prices) >= 50:
                sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
            if len(close_prices) >= 200:
                sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
            metrics_results[ticker.upper()] = {
                'total_return': round(total_return, 4),
                'annualized_return': round(annualized_return, 4),
                'annualized_volatility': round(volatility, 4),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 4),
                'beta': round(beta, 2) if beta is not None else None,
                'expected_return_capm': round(expected_return_capm, 4) if expected_return_capm is not None else None,
                'sma_50': round(sma_50, 2) if sma_50 is not None else None,
                'sma_200': round(sma_200, 2) if sma_200 is not None else None,
                'period_days': num_days
            }
        if portfolio is None:
            logging.info("(Portfolio metrics not requested, only individual metrics calculated)")
        logging.info("Metrics Calculation Complete (CAPM, SMAs, Manual Beta included)")
        return metrics_results
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        metrics_results["error"] = f"Calculation failed: {str(e)}"
        return metrics_results

def validate_portfolio_calculations(portfolio: Optional[Dict[str, float]], metrics: Optional[Dict]) -> Dict:
    """Performs validation checks on the portfolio and its metrics."""
    logging.info("Validating Portfolio Calculations")
    errors = []
    status = 'pass'
    if not isinstance(portfolio, dict) or not portfolio:
        errors.append("Portfolio allocation is missing or not a dictionary.")
        status = 'fail'
        return {"status": status, "errors": errors}
    if not isinstance(metrics, dict):
        errors.append("Metrics data is missing or not a dictionary.")
        status = 'fail'
        metrics = {}
    total_weight = sum(portfolio.values())
    if not abs(total_weight - 1.0) < 0.01:
        errors.append(f"Portfolio weights sum to {total_weight:.4f}, significantly different from 1.0.")
        status = 'fail'
    if any(w < 0 for w in portfolio.values()):
        logging.info("Portfolio contains negative weights (potential short positions).")
    portfolio_metrics = metrics.get('portfolio', None)
    if not isinstance(portfolio_metrics, dict) or not portfolio_metrics:
        if portfolio:
            errors.append("Portfolio metrics dictionary is missing within the metrics results.")
            status = 'fail'
    elif 'error' in portfolio_metrics:
        errors.append(f"Portfolio metrics calculation reported an error: {portfolio_metrics['error']}")
        status = 'fail'
    logging.info(f"Validation Complete: Status={status}")
    return {"status": status, "errors": errors}

def calculate_metrics_node(state) -> Dict:
    """Node to call the metrics calculation function."""
    logging.info("Calculating Metrics Node")
    financial_data = state.get('financial_data')
    if not financial_data:
        logging.error("Financial data missing, cannot calculate metrics.")
        return {"error_message": "Cannot calculate metrics: Financial data is missing."}
    metrics = calculate_financial_metrics(data=financial_data, portfolio=None)
    if 'error' in metrics:
        return {"metrics": metrics, "error_message": f"Metrics calculation failed: {metrics['error']}"}
    proposed_portfolio = state.get('proposed_portfolio')
    if proposed_portfolio:
        portfolio_metrics = calculate_financial_metrics(data=financial_data, portfolio=proposed_portfolio)
        if 'portfolio' in portfolio_metrics:
            metrics['portfolio'] = portfolio_metrics['portfolio']
        elif 'error' in portfolio_metrics:
            metrics['portfolio'] = {'error': portfolio_metrics['error']}
    return {"metrics": metrics} 