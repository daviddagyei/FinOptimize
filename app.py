import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional
# Import utils functions for calculations
from utils import calc_performance_metrics, calc_return_metrics, calc_risk_metrics, calc_univariate_regression
from utils import calc_tangency_portfolio, calc_gmv_portfolio, calc_mv_portfolio, plot_mv_frontier

# Configure Streamlit page
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Popular tickers for autocomplete suggestions
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'BND', 'GLD', 'SLV',
    'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA', 'PYPL', 'SQ',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'BMY', 'GILD', 'AMGN',
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'OKE', 'EPD'
]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker: str, start_date: str, end_date: str, data_type: str = "Close", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance with caching.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    data_type : str
        Type of data to return ('Close', 'Adj Close', 'OHLC', 'All')
    interval : str
        Data interval ('1d', '1wk', '1mo')
    
    Returns:
    --------
    pd.DataFrame
        Historical stock data
    """
    try:
        # Download data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        
        # Return specific data type
        if data_type == "Close":
            return data[['Close']].rename(columns={'Close': ticker})
        elif data_type == "Adj Close":
            return data[['Close']].rename(columns={'Close': ticker})  # yfinance returns adjusted close by default
        elif data_type == "OHLC":
            return data[['Open', 'High', 'Low', 'Close']]
        elif data_type == "All":
            return data
        else:
            return data[['Close']].rename(columns={'Close': ticker})
            
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_multiple_tickers(tickers: List[str], start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch data for multiple tickers and combine into a single DataFrame.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    interval : str
        Data interval ('1d', '1wk', '1mo')
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with closing prices for all tickers
    """
    try:
        # Use yfinance to download multiple tickers at once
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval, group_by='ticker')
        
        if data.empty:
            return pd.DataFrame()
        
        if len(tickers) == 1:
            # For single ticker, yfinance returns different structure
            if 'Close' in data.columns:
                return data[['Close']].rename(columns={'Close': tickers[0]})
            elif hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                # Handle MultiIndex columns - flatten them
                ticker = tickers[0]
                if (ticker, 'Close') in data.columns:
                    result = data[(ticker, 'Close')].to_frame()
                    result.columns = [ticker]
                    return result
                else:
                    # Take the first column and rename it
                    result = data.iloc[:, :1].copy()
                    result.columns = [ticker]
                    return result
            else:
                # Sometimes the structure is different, try to get the close price
                close_data = data.filter(like='Close')
                if not close_data.empty:
                    return close_data.rename(columns={close_data.columns[0]: tickers[0]})
                else:
                    # Last resort - take the last column which is usually Close
                    return data.iloc[:, -1:].rename(columns={data.columns[-1]: tickers[0]})
        
        # Extract closing prices for multiple tickers
        close_prices = pd.DataFrame()
        for ticker in tickers:
            if hasattr(data.columns, 'levels') and ticker in data.columns.levels[0]:
                close_prices[ticker] = data[(ticker, 'Close')]
            elif (ticker, 'Close') in data.columns:
                close_prices[ticker] = data[(ticker, 'Close')]
        
        return close_prices.dropna()
        
    except Exception as e:
        st.error(f"Error fetching data for tickers {tickers}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ticker_info(ticker: str) -> dict:
    """
    Get basic information about a ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    
    Returns:
    --------
    dict
        Basic information about the ticker
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'currency': info.get('currency', 'USD')
        }
    except:
        return {'name': ticker, 'sector': 'N/A', 'industry': 'N/A', 'market_cap': 'N/A', 'currency': 'USD'}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_risk_free_rate(start_date: str, end_date: str, rate_type: str = "^TNX", interval: str = "1d", annualization_factor: int = 252) -> pd.DataFrame:
    """
    Fetch risk-free rate data using Treasury yields.
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    rate_type : str
        Risk-free rate proxy ticker. Default is 10Y Treasury (^TNX)
        Options: ^IRX (3-month), ^FVX (5-year), ^TNX (10-year), ^TYX (30-year)
    interval : str
        Data interval ('1d', '1wk', '1mo')
    annualization_factor : int
        Number of periods per year for the given interval
    
    Returns:
    --------
    pd.DataFrame
        Risk-free rate data
    """
    try:
        # Fetch Treasury data
        treasury = yf.Ticker(rate_type)
        data = treasury.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            # Fallback to a constant risk-free rate if data unavailable
            st.warning(f"Could not fetch {rate_type} data. Using 2% constant risk-free rate.")
            
            # Create date range based on interval
            if interval == "1d":
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
            elif interval == "1wk":
                dates = pd.date_range(start=start_date, end=end_date, freq='W')
            elif interval == "1mo":
                dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start
            else:
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Convert annual 2% to period returns using compound formula
            period_rf_constant = (1 + 0.02) ** (1/annualization_factor) - 1
            rf_rate = pd.DataFrame({'Risk_Free_Rate': [period_rf_constant] * len(dates)}, index=dates)
            return rf_rate
        
        # Convert annual yield to period returns using compound formula
        # R_f,t = (1 + R_f,ann)^(Δt) - 1, where Δt = 1/annualization_factor
        annual_rf = data['Close'] / 100  # Convert percentage to decimal
        period_rf = (1 + annual_rf) ** (1/annualization_factor) - 1
        rf_rate = pd.DataFrame({'Risk_Free_Rate': period_rf})
        
        # Remove timezone information to match other data
        rf_rate.index = rf_rate.index.tz_localize(None)
        
        return rf_rate
        
    except Exception as e:
        st.warning(f"Error fetching risk-free rate data: {str(e)}. Using 2% constant rate.")
        
        # Create date range based on interval
        if interval == "1d":
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        elif interval == "1wk":
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
        elif interval == "1mo":
            dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Convert annual 2% to period returns using compound formula
        period_rf_constant = (1 + 0.02) ** (1/annualization_factor) - 1
        rf_rate = pd.DataFrame({'Risk_Free_Rate': [period_rf_constant] * len(dates)}, index=dates)
        return rf_rate

def calculate_excess_returns(returns_df: pd.DataFrame, rf_rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate excess returns by subtracting risk-free rate from asset returns.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of asset returns
    rf_rate_df : pd.DataFrame
        DataFrame with risk-free rate
    
    Returns:
    --------
    pd.DataFrame
        Excess returns (ri - rf)
    """
    try:
        # Ensure both dataframes have compatible datetime indices
        returns_clean = returns_df.copy()
        rf_clean = rf_rate_df.copy()
        
        # Remove timezone info from both if present to ensure compatibility
        if returns_clean.index.tz is not None:
            returns_clean.index = returns_clean.index.tz_localize(None)
        if rf_clean.index.tz is not None:
            rf_clean.index = rf_clean.index.tz_localize(None)
        
        # Align dates using forward fill for missing risk-free rate dates
        aligned_rf = rf_clean.reindex(returns_clean.index, method='ffill')
        
        # Calculate excess returns for each asset
        excess_returns = returns_clean.subtract(aligned_rf['Risk_Free_Rate'], axis=0)
        
        return excess_returns
        
    except Exception as e:
        st.error(f"Error calculating excess returns: {str(e)}")
        return returns_df  # Return original returns if calculation fails

def calculate_comprehensive_metrics(data: pd.DataFrame, start_date: str, end_date: str, 
                                  rf_rate_type: str = "^TNX", interval: str = "1d", 
                                  annualization_factor: int = 252) -> pd.DataFrame:
    """
    Calculate comprehensive performance and risk metrics including excess returns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data DataFrame
    start_date : str
        Start date for risk-free rate
    end_date : str
        End date for risk-free rate
    rf_rate_type : str
        Risk-free rate ticker
    interval : str
        Data interval ('1d', '1wk', '1mo')
    annualization_factor : int
        Number of periods per year for the given interval
    
    Returns:
    --------
    pd.DataFrame
        Comprehensive metrics DataFrame
    """
    try:
        # Calculate returns
        returns = data.pct_change().dropna()
        
        if returns.empty:
            return pd.DataFrame()
        
        # Ensure returns index is timezone-naive for compatibility
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        
        # Fetch risk-free rate
        rf_data = fetch_risk_free_rate(start_date, end_date, rf_rate_type, interval, annualization_factor)
        
        # Calculate excess returns
        excess_returns = calculate_excess_returns(returns, rf_data)
        
        # Calculate basic metrics using existing functions with correct annualization factor
        basic_metrics = calc_performance_metrics(returns, adj=annualization_factor, var=0.05)
        
        # Calculate excess return metrics
        excess_metrics = {}
        for col in excess_returns.columns:
            col_excess = excess_returns[[col]].dropna()
            if not col_excess.empty:
                excess_metrics[f"{col}_Excess_Return_Ann"] = col_excess.mean().iloc[0] * annualization_factor
                excess_metrics[f"{col}_Excess_Sharpe"] = (col_excess.mean().iloc[0] * annualization_factor) / (col_excess.std().iloc[0] * np.sqrt(annualization_factor))
        
        # Add risk-free rate information
        rf_avg = rf_data['Risk_Free_Rate'].mean() * annualization_factor  # Annualized
        
        # Enhanced metrics DataFrame
        enhanced_df = basic_metrics.copy()
        enhanced_df['Risk_Free_Rate_Ann'] = rf_avg
        
        # Add excess return metrics
        for col in enhanced_df.index:
            if f"{col}_Excess_Return_Ann" in excess_metrics:
                enhanced_df.loc[col, 'Excess_Return_Ann'] = excess_metrics[f"{col}_Excess_Return_Ann"]
                enhanced_df.loc[col, 'Excess_Sharpe_Ratio'] = excess_metrics[f"{col}_Excess_Sharpe"]
        
        return enhanced_df
        
    except Exception as e:
        st.error(f"Error calculating comprehensive metrics: {str(e)}")
        return pd.DataFrame()

def calculate_rolling_correlations(returns_df: pd.DataFrame, window: int = 30) -> dict:
    """
    Calculate rolling correlations between assets.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of asset returns
    window : int
        Rolling window size in days
    
    Returns:
    --------
    dict
        Dictionary containing rolling correlation data
    """
    try:
        rolling_corrs = {}
        asset_pairs = []
        
        # Get all unique pairs of assets
        assets = returns_df.columns.tolist()
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                pair = f"{assets[i]} vs {assets[j]}"
                asset_pairs.append((assets[i], assets[j], pair))
                
                # Calculate rolling correlation
                rolling_corr = returns_df[assets[i]].rolling(window=window).corr(returns_df[assets[j]])
                rolling_corrs[pair] = rolling_corr.dropna()
        
        return {
            'rolling_correlations': rolling_corrs,
            'asset_pairs': asset_pairs,
            'window': window
        }
        
    except Exception as e:
        st.error(f"Error calculating rolling correlations: {str(e)}")
        return {}

def calculate_diversification_metrics(corr_matrix: pd.DataFrame) -> dict:
    """
    Calculate diversification and correlation insights.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix of assets
    
    Returns:
    --------
    dict
        Dictionary with diversification metrics
    """
    try:
        # Extract upper triangular correlations (exclude diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()
        
        metrics = {
            'avg_correlation': correlations.mean(),
            'max_correlation': correlations.max(),
            'min_correlation': correlations.min(),
            'median_correlation': correlations.median(),
            'std_correlation': correlations.std(),
            'highly_correlated_pairs': correlations[correlations > 0.7].sort_values(ascending=False),
            'negatively_correlated_pairs': correlations[correlations < -0.3].sort_values(),
            'diversification_ratio': 1 - correlations.abs().mean(),
            'correlation_distribution': correlations
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating diversification metrics: {str(e)}")
        return {}

def calculate_capm_analysis(asset_returns: pd.DataFrame, market_returns: pd.DataFrame, 
                          rf_rate_df: pd.DataFrame, market_ticker: str = "SPY", 
                          annualization_factor: int = 252) -> dict:
    """
    Calculate comprehensive CAPM analysis for assets against market benchmark.
    
    Parameters:
    -----------
    asset_returns : pd.DataFrame
        DataFrame of asset returns
    market_returns : pd.DataFrame  
        DataFrame of market benchmark returns
    rf_rate_df : pd.DataFrame
        DataFrame with risk-free rate
    market_ticker : str
        Market benchmark ticker symbol
    annualization_factor : int
        Number of periods per year for the given frequency
    
    Returns:
    --------
    dict
        Dictionary containing CAPM analysis results
    """
    try:
        # Ensure all dataframes have compatible datetime indices
        asset_returns_clean = asset_returns.copy()
        market_returns_clean = market_returns.copy()
        rf_rate_clean = rf_rate_df.copy()
        
        # Remove timezone info from all if present to ensure compatibility
        if asset_returns_clean.index.tz is not None:
            asset_returns_clean.index = asset_returns_clean.index.tz_localize(None)
        if market_returns_clean.index.tz is not None:
            market_returns_clean.index = market_returns_clean.index.tz_localize(None)
        if rf_rate_clean.index.tz is not None:
            rf_rate_clean.index = rf_rate_clean.index.tz_localize(None)
        
        # Align all data to same dates
        common_dates = asset_returns_clean.index.intersection(market_returns_clean.index).intersection(rf_rate_clean.index)
        
        if len(common_dates) < 30:  # Need sufficient data
            st.warning("Insufficient overlapping data for CAPM analysis (minimum 30 observations required)")
            return {}
        
        # Filter to common dates
        asset_returns_aligned = asset_returns_clean.loc[common_dates]
        market_returns_aligned = market_returns_clean.loc[common_dates]
        rf_rate_aligned = rf_rate_clean.loc[common_dates]
        
        # Calculate excess returns (asset - risk free rate)
        asset_excess_returns = calculate_excess_returns(asset_returns_aligned, rf_rate_aligned)
        market_excess_returns = calculate_excess_returns(market_returns_aligned, rf_rate_aligned)
        
        capm_results = {}
        
        # Calculate CAPM metrics for each asset
        for asset in asset_excess_returns.columns:
            try:
                # Prepare data for regression (asset excess return vs market excess return)
                asset_excess = asset_excess_returns[[asset]].dropna()
                market_excess_common = market_excess_returns.reindex(asset_excess.index, method='ffill')
                
                if len(asset_excess) < 20:  # Need minimum observations
                    continue
                
                # Use utils function for regression analysis
                regression_results = calc_univariate_regression(
                    y=asset_excess, 
                    X=market_excess_common, 
                    intercept=True, 
                    adj=annualization_factor  # Use frequency-specific annualization factor
                )
                
                if not regression_results.empty:
                    # Extract CAPM metrics from regression
                    alpha_annual = regression_results.loc[asset, 'Alpha (Annualized)']
                    alpha_raw = regression_results.loc[asset, 'Alpha (Raw)']
                    beta = regression_results.loc[asset, 'Beta']
                    r_squared = regression_results.loc[asset, 'R-Squared']
                    treynor_ratio = regression_results.loc[asset, 'Treynor Ratio']
                    info_ratio = regression_results.loc[asset, 'Information Ratio']
                    tracking_error = regression_results.loc[asset, 'Tracking Error']
                    downside_beta = regression_results.loc[asset, 'Downside Beta']
                    
                    # Calculate expected return using CAPM
                    market_premium = market_excess_common.mean().iloc[0] * annualization_factor  # Annualized market premium
                    rf_annual = rf_rate_aligned['Risk_Free_Rate'].mean() * annualization_factor
                    expected_return_capm = rf_annual + beta * market_premium
                    
                    # Calculate actual return
                    actual_return = asset_returns_aligned[asset].mean() * annualization_factor
                    
                    # Calculate Jensen's Alpha (actual - expected)
                    jensen_alpha = actual_return - expected_return_capm
                    
                    # Store results
                    capm_result = {
                        'asset': asset,
                        'alpha_annualized': alpha_annual,
                        'alpha_raw': alpha_raw,
                        'beta': beta,
                        'r_squared': r_squared,
                        'treynor_ratio': treynor_ratio,
                        'information_ratio': info_ratio,
                        'tracking_error': tracking_error,
                        'downside_beta': downside_beta,
                        'expected_return_capm': expected_return_capm,
                        'actual_return': actual_return,
                        'jensen_alpha': jensen_alpha,
                        'market_premium': market_premium,
                        'risk_free_rate': rf_annual,
                        'observations': len(asset_excess),
                        'asset_excess_returns': asset_excess,
                        'market_excess_returns': market_excess_common
                    }
                    
                    capm_results[asset] = capm_result
                    
            except Exception as e:
                st.warning(f"Could not calculate CAPM for {asset}: {str(e)}")
                continue
        
        # Add market benchmark info
        capm_summary = {
            'market_ticker': market_ticker,
            'analysis_period': f"{common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}",
            'total_observations': len(common_dates),
            'market_return_annual': market_returns_aligned.mean().iloc[0] * annualization_factor,
            'market_volatility_annual': market_returns_aligned.std().iloc[0] * np.sqrt(annualization_factor),
            'market_sharpe': (market_returns_aligned.mean().iloc[0] * annualization_factor - rf_annual) / (market_returns_aligned.std().iloc[0] * np.sqrt(annualization_factor)),
            'assets_analyzed': list(capm_results.keys()),
            'capm_results': capm_results
        }
        
        return capm_summary
        
    except Exception as e:
        st.error(f"Error in CAPM analysis: {str(e)}")
        return {}

def calculate_optimal_portfolios(returns_df: pd.DataFrame, annualization_factor: int = 252) -> dict:
    """
    Calculate optimal portfolios using mean-variance optimization.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of asset returns
    annualization_factor : int
        Number of periods per year for the given frequency
    
    Returns:
    --------
    dict
        Dictionary containing optimal portfolio results
    """
    try:
        if returns_df.empty or len(returns_df.columns) < 2:
            return {}
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean() * annualization_factor  # Annualized
        cov_matrix = returns_df.cov() * annualization_factor      # Annualized
        
        # Calculate portfolio weights
        tangency_weights = calc_tangency_portfolio(mean_returns.values, cov_matrix.values)
        gmv_weights = calc_gmv_portfolio(cov_matrix.values)
        
        # Convert to Series with asset names
        tangency_weights = pd.Series(tangency_weights, index=returns_df.columns, name='Tangency')
        gmv_weights = pd.Series(gmv_weights, index=returns_df.columns, name='GMV')
        
        # Calculate portfolio metrics
        portfolios = {}
        
        # Tangency Portfolio
        tangency_return = (mean_returns * tangency_weights).sum()
        tangency_volatility = np.sqrt(tangency_weights.T @ cov_matrix @ tangency_weights)
        tangency_sharpe = tangency_return / tangency_volatility
        
        portfolios['Tangency'] = {
            'weights': tangency_weights,
            'expected_return': tangency_return,
            'volatility': tangency_volatility,
            'sharpe_ratio': tangency_sharpe
        }
        
        # Global Minimum Variance Portfolio
        gmv_return = (mean_returns * gmv_weights).sum()
        gmv_volatility = np.sqrt(gmv_weights.T @ cov_matrix @ gmv_weights)
        gmv_sharpe = gmv_return / gmv_volatility
        
        portfolios['GMV'] = {
            'weights': gmv_weights,
            'expected_return': gmv_return,
            'volatility': gmv_volatility,
            'sharpe_ratio': gmv_sharpe
        }
        
        # Equally Weighted Portfolio
        n_assets = len(returns_df.columns)
        equal_weights = pd.Series([1.0/n_assets] * n_assets, index=returns_df.columns, name='Equal')
        equal_return = (mean_returns * equal_weights).sum()
        equal_volatility = np.sqrt(equal_weights.T @ cov_matrix @ equal_weights)
        equal_sharpe = equal_return / equal_volatility
        
        portfolios['Equal Weight'] = {
            'weights': equal_weights,
            'expected_return': equal_return,
            'volatility': equal_volatility,
            'sharpe_ratio': equal_sharpe
        }
        
        return {
            'portfolios': portfolios,
            'mean_returns': mean_returns,
            'cov_matrix': cov_matrix,
            'individual_assets': {
                'returns': mean_returns,
                'volatilities': np.sqrt(np.diag(cov_matrix)),
                'sharpe_ratios': mean_returns / np.sqrt(np.diag(cov_matrix))
            }
        }
        
    except Exception as e:
        st.error(f"Error calculating optimal portfolios: {str(e)}")
        return {}

def create_efficient_frontier_data(mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
                                 n_portfolios: int = 100) -> pd.DataFrame:
    """
    Generate efficient frontier data points showing the complete "C" shaped curve.
    
    Parameters:
    -----------
    mean_returns : pd.Series
        Mean returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix
    n_portfolios : int
        Number of portfolios to generate for the frontier
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with complete frontier data (both efficient and inefficient)
    """
    try:
        # Calculate GMV portfolio to get the minimum risk point
        gmv_weights = calc_gmv_portfolio(cov_matrix.values)
        gmv_return = np.sum(gmv_weights * mean_returns.values)
        gmv_volatility = np.sqrt(gmv_weights.T @ cov_matrix.values @ gmv_weights)
        
        # Calculate tangency portfolio for reference
        tangency_weights = calc_tangency_portfolio(mean_returns.values, cov_matrix.values)
        tangency_return = np.sum(tangency_weights * mean_returns.values)
        
        # Create a wider range of target returns to capture the full "C" curve
        # Include returns both above and below the GMV return
        min_ret = min(mean_returns.min() * 0.8, gmv_return * 0.3)  # Go well below GMV
        max_ret = max(mean_returns.max() * 1.3, tangency_return * 1.1)  # Go above max return
        
        # Create target returns with higher density
        target_returns = np.linspace(min_ret, max_ret, n_portfolios * 2)
        
        all_portfolios = []
        
        for target_ret in target_returns:
            try:
                # Calculate mean-variance portfolio for target return
                weights = calc_mv_portfolio(mean_returns.values, cov_matrix.values, 
                                          excess=False, target=target_ret)
                
                # Check if weights are valid (sum to approximately 1)
                if abs(np.sum(weights) - 1.0) > 0.05:  # More lenient for the full curve
                    continue
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(weights * mean_returns.values)
                portfolio_volatility = np.sqrt(weights.T @ cov_matrix.values @ weights)
                
                # Include all valid portfolios to show the complete curve
                if portfolio_volatility > 0 and not np.isnan(portfolio_volatility):
                    # Determine if this is on the efficient or inefficient frontier
                    is_efficient = portfolio_return >= gmv_return * 0.995
                    
                    all_portfolios.append({
                        'return': portfolio_return,
                        'volatility': portfolio_volatility,
                        'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0,
                        'weights': weights,
                        'is_efficient': is_efficient
                    })
                
            except Exception:
                continue  # Skip problematic portfolios
        
        # Add GMV portfolio explicitly (this is the turning point of the "C")
        all_portfolios.append({
            'return': gmv_return,
            'volatility': gmv_volatility,
            'sharpe_ratio': gmv_return / gmv_volatility if gmv_volatility > 0 else 0,
            'weights': gmv_weights,
            'is_efficient': True
        })
        
        # Add tangency portfolio explicitly
        tangency_volatility = np.sqrt(tangency_weights.T @ cov_matrix.values @ tangency_weights)
        all_portfolios.append({
            'return': tangency_return,
            'volatility': tangency_volatility,
            'sharpe_ratio': tangency_return / tangency_volatility if tangency_volatility > 0 else 0,
            'weights': tangency_weights,
            'is_efficient': True
        })
        
        # Convert to DataFrame and sort by volatility to create smooth curve
        frontier_df = pd.DataFrame(all_portfolios)
        if not frontier_df.empty:
            frontier_df = frontier_df.sort_values('volatility').drop_duplicates(subset=['volatility'], keep='first')
            
            # Remove any extreme outliers that might distort the curve
            vol_q99 = frontier_df['volatility'].quantile(0.99)
            vol_q01 = frontier_df['volatility'].quantile(0.01)
            frontier_df = frontier_df[
                (frontier_df['volatility'] >= vol_q01) & 
                (frontier_df['volatility'] <= vol_q99)
            ]
        
        return frontier_df
        
    except Exception as e:
        st.error(f"Error creating efficient frontier: {str(e)}")
        return pd.DataFrame()

def create_efficient_frontier_plot(optimal_data: dict, returns_df: pd.DataFrame) -> go.Figure:
    """
    Create efficient frontier visualization showing the complete "C" shaped curve.
    
    Parameters:
    -----------
    optimal_data : dict
        Optimal portfolio data
    returns_df : pd.DataFrame
        Returns data for calculating efficient frontier
    
    Returns:
    --------
    go.Figure
        Plotly figure with complete efficient frontier
    """
    try:
        fig = go.Figure()
        
        if not optimal_data or 'mean_returns' not in optimal_data:
            return fig
        
        mean_returns = optimal_data['mean_returns']
        cov_matrix = optimal_data['cov_matrix']
        portfolios = optimal_data['portfolios']
        individual_assets = optimal_data['individual_assets']
        
        # Generate complete frontier with more points for smoother "C" curve
        frontier_data = create_efficient_frontier_data(mean_returns, cov_matrix, n_portfolios=150)
        
        if not frontier_data.empty and len(frontier_data) > 1:
            # Sort by volatility to ensure proper line drawing
            frontier_data = frontier_data.sort_values('volatility')
            
            # Separate efficient and inefficient portions
            efficient_data = frontier_data[frontier_data['is_efficient'] == True]
            inefficient_data = frontier_data[frontier_data['is_efficient'] == False]
            
            # Plot the complete frontier as one continuous line first
            fig.add_trace(go.Scatter(
                x=frontier_data['volatility'],
                y=frontier_data['return'],
                mode='lines',
                name='Complete Frontier',
                line=dict(color='lightblue', width=2, dash='solid'),
                hovertemplate='Complete Frontier<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{customdata:.3f}<extra></extra>',
                customdata=frontier_data['sharpe_ratio'],
                showlegend=True
            ))
            
            # Overlay the efficient portion with a different style
            if not efficient_data.empty:
                efficient_data_sorted = efficient_data.sort_values('volatility')
                fig.add_trace(go.Scatter(
                    x=efficient_data_sorted['volatility'],
                    y=efficient_data_sorted['return'],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='navy', width=4),
                    hovertemplate='Efficient Frontier<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{customdata:.3f}<extra></extra>',
                    customdata=efficient_data_sorted['sharpe_ratio']
                ))
            
            # Add markers to show the distinction more clearly
            if not inefficient_data.empty:
                fig.add_trace(go.Scatter(
                    x=inefficient_data['volatility'],
                    y=inefficient_data['return'],
                    mode='markers',
                    name='Inefficient Region',
                    marker=dict(size=4, color='lightcoral', symbol='circle', opacity=0.6),
                    hovertemplate='Inefficient Portfolio<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{customdata:.3f}<extra></extra>',
                    customdata=inefficient_data['sharpe_ratio']
                ))
            
            # Add markers on efficient frontier for key points
            if not efficient_data.empty:
                n_markers = min(8, len(efficient_data))
                if n_markers > 0:
                    marker_indices = np.linspace(0, len(efficient_data)-1, n_markers, dtype=int)
                    marker_data = efficient_data.iloc[marker_indices]
                    
                    fig.add_trace(go.Scatter(
                        x=marker_data['volatility'],
                        y=marker_data['return'],
                        mode='markers',
                        name='Efficient Points',
                        marker=dict(size=6, color='darkblue', symbol='circle'),
                        showlegend=False,
                        hovertemplate='Efficient Point<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{customdata:.3f}<extra></extra>',
                        customdata=marker_data['sharpe_ratio']
                    ))
        else:
            st.warning("Unable to generate complete frontier with current data. This may occur with highly correlated assets or insufficient data.")
        
        # Add individual assets
        fig.add_trace(go.Scatter(
            x=individual_assets['volatilities'],
            y=individual_assets['returns'],
            mode='markers+text',
            name='Individual Assets',
            marker=dict(size=14, color='lightgray', symbol='circle', line=dict(width=2, color='black')),
            text=individual_assets['returns'].index,
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertemplate='%{text}<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{customdata:.3f}<extra></extra>',
            customdata=individual_assets['sharpe_ratios']
        ))
        
        # Add optimal portfolios with enhanced styling
        colors = {'Tangency': 'red', 'GMV': 'green', 'Equal Weight': 'orange'}
        symbols = {'Tangency': 'star', 'GMV': 'diamond', 'Equal Weight': 'square'}
        sizes = {'Tangency': 20, 'GMV': 18, 'Equal Weight': 16}
        
        for portfolio_name, portfolio_data in portfolios.items():
            fig.add_trace(go.Scatter(
                x=[portfolio_data['volatility']],
                y=[portfolio_data['expected_return']],
                mode='markers+text',
                name=f'{portfolio_name} Portfolio',
                marker=dict(
                    size=sizes.get(portfolio_name, 18),
                    color=colors.get(portfolio_name, 'purple'),
                    symbol=symbols.get(portfolio_name, 'circle'),
                    line=dict(width=3, color='white')
                ),
                text=[portfolio_name],
                textposition='top center',
                textfont=dict(size=10, color='black', family='Arial Black'),
                hovertemplate=f'{portfolio_name} Portfolio<br>' +
                             f'Volatility: {portfolio_data["volatility"]:.2%}<br>' +
                             f'Return: {portfolio_data["expected_return"]:.2%}<br>' +
                             f'Sharpe: {portfolio_data["sharpe_ratio"]:.3f}<extra></extra>'
            ))
        
        # Update layout with improved styling
        fig.update_layout(
            title=dict(
                text='Complete Efficient Frontier - "C" Shaped Curve',
                font=dict(size=18, color='black'),
                x=0.5
            ),
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            height=650,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Format axes as percentages
        fig.update_layout(
            xaxis=dict(
                tickformat='.1%',
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True
            ),
            yaxis=dict(
                tickformat='.1%',
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True
            )
        )
        
        # Add annotations for better understanding
        if portfolios:
            gmv_portfolio = portfolios.get('GMV')
            tangency_portfolio = portfolios.get('Tangency')
            
            if gmv_portfolio:
                fig.add_annotation(
                    x=gmv_portfolio['volatility'],
                    y=gmv_portfolio['expected_return'],
                    text="Minimum Risk<br>(Turning Point)",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor="green",
                    ax=30,
                    ay=-40,
                    font=dict(size=10, color="green")
                )
            
            if tangency_portfolio:
                fig.add_annotation(
                    x=tangency_portfolio['volatility'],
                    y=tangency_portfolio['expected_return'],
                    text="Max Sharpe Ratio<br>(Optimal)",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor="red",
                    ax=30,
                    ay=40,
                    font=dict(size=10, color="red")
                )
        
        # Add explanatory annotation about the curve shape
        fig.add_annotation(
            x=0.02, y=0.02,
            xref="paper", yref="paper",
            text="<b>Curve Explanation:</b><br>" +
                 "• Complete 'C' shaped frontier shown<br>" +
                 "• Dark blue: Efficient frontier (upper curve)<br>" +
                 "• Light blue: Complete frontier including inefficient region<br>" +
                 "• Red dots: Inefficient portfolios (lower curve)",
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating efficient frontier plot: {str(e)}")
        return go.Figure()

def create_portfolio_weights_chart(optimal_data: dict) -> go.Figure:
    """
    Create portfolio weights comparison chart.
    
    Parameters:
    -----------
    optimal_data : dict
        Optimal portfolio data
    
    Returns:
    --------
    go.Figure
        Plotly figure with portfolio weights
    """
    try:
        fig = go.Figure()
        
        if not optimal_data or 'portfolios' not in optimal_data:
            return fig
        
        portfolios = optimal_data['portfolios']
        
        # Extract weights data
        weights_data = []
        portfolio_names = []
        
        for portfolio_name, portfolio_data in portfolios.items():
            weights = portfolio_data['weights']
            weights_data.append(weights.values)
            portfolio_names.append(portfolio_name)
        
        # Create stacked bar chart
        assets = list(portfolios[list(portfolios.keys())[0]]['weights'].index)
        
        for i, asset in enumerate(assets):
            asset_weights = [weights[i] for weights in weights_data]
            
            fig.add_trace(go.Bar(
                name=asset,
                x=portfolio_names,
                y=asset_weights,
                hovertemplate=f'{asset}<br>Weight: %{{y:.1%}}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='Portfolio Weights Comparison',
            xaxis_title='Portfolio Type',
            yaxis_title='Weight Allocation',
            barmode='stack',
            height=500,
            yaxis=dict(tickformat='.0%')
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating portfolio weights chart: {str(e)}")
        return go.Figure()

def create_risk_return_comparison(optimal_data: dict) -> pd.DataFrame:
    """
    Create risk-return comparison table.
    
    Parameters:
    -----------
    optimal_data : dict
        Optimal portfolio data
    
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    try:
        if not optimal_data or 'portfolios' not in optimal_data:
            return pd.DataFrame()
        
        portfolios = optimal_data['portfolios']
        individual_assets = optimal_data['individual_assets']
        
        comparison_data = []
        
        # Add individual assets
        returns_series = individual_assets['returns']
        volatilities_array = individual_assets['volatilities']
        sharpe_ratios_array = individual_assets['sharpe_ratios']
        
        for i, asset in enumerate(returns_series.index):
            comparison_data.append({
                'Portfolio/Asset': asset,
                'Type': 'Individual Asset',
                'Expected Return': returns_series.iloc[i],
                'Volatility': volatilities_array[i],
                'Sharpe Ratio': sharpe_ratios_array[i]
            })
        
        # Add optimal portfolios
        for portfolio_name, portfolio_data in portfolios.items():
            comparison_data.append({
                'Portfolio/Asset': portfolio_name,
                'Type': 'Optimal Portfolio',
                'Expected Return': portfolio_data['expected_return'],
                'Volatility': portfolio_data['volatility'],
                'Sharpe Ratio': portfolio_data['sharpe_ratio']
            })
        
        return pd.DataFrame(comparison_data)
        
    except Exception as e:
        st.error(f"Error creating risk-return comparison: {str(e)}")
        return pd.DataFrame()

def calculate_portfolio_metrics_over_time(returns_df: pd.DataFrame, weights: pd.Series, 
                                        rolling_window: int = 60) -> pd.DataFrame:
    """
    Calculate rolling portfolio metrics over time.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Asset returns
    weights : pd.Series
        Portfolio weights
    rolling_window : int
        Rolling window size
    
    Returns:
    --------
    pd.DataFrame
        Rolling portfolio metrics
    """
    try:
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate rolling metrics
        rolling_metrics = pd.DataFrame(index=returns_df.index)
        rolling_metrics['Portfolio_Return'] = portfolio_returns
        rolling_metrics['Rolling_Return'] = portfolio_returns.rolling(rolling_window).mean()
        rolling_metrics['Rolling_Volatility'] = portfolio_returns.rolling(rolling_window).std()
        rolling_metrics['Rolling_Sharpe'] = rolling_metrics['Rolling_Return'] / rolling_metrics['Rolling_Volatility']
        
        # Calculate cumulative returns
        rolling_metrics['Cumulative_Return'] = (1 + portfolio_returns).cumprod()
        
        return rolling_metrics.dropna()
        
    except Exception as e:
        st.error(f"Error calculating portfolio metrics over time: {str(e)}")
        return pd.DataFrame()
    """
    Create Security Market Line visualization for CAPM analysis.
    
    Parameters:
    -----------
    capm_data : dict
        CAPM analysis results
    rf_rate : float
        Risk-free rate (annualized)
    
    Returns:
    --------
    go.Figure
        Plotly figure with Security Market Line
    """
    try:
        fig = go.Figure()
        
        if not capm_data or 'capm_results' not in capm_data:
            return fig
        
        capm_results = capm_data['capm_results']
        market_premium = capm_data.get('market_return_annual', 0) - rf_rate
        
        # Extract data for plotting
        betas = []
        expected_returns = []
        actual_returns = []
        asset_names = []
        
        for asset, results in capm_results.items():
            betas.append(results['beta'])
            expected_returns.append(results['expected_return_capm'])
            actual_returns.append(results['actual_return'])
            asset_names.append(asset)
        
        # Create Security Market Line (theoretical)
        beta_range = np.linspace(0, max(betas + [1.5]), 100)
        sml_returns = rf_rate + beta_range * market_premium
        
        # Add Security Market Line
        fig.add_trace(go.Scatter(
            x=beta_range,
            y=sml_returns,
            mode='lines',
            name='Security Market Line (SML)',
            line=dict(color='blue', width=3, dash='dash'),
            hovertemplate='Beta: %{x:.2f}<br>Expected Return: %{y:.2%}<extra></extra>'
        ))
        
        # Add market portfolio point (beta = 1)
        fig.add_trace(go.Scatter(
            x=[1.0],
            y=[rf_rate + market_premium],
            mode='markers',
            name=f'Market ({capm_data.get("market_ticker", "Market")})',
            marker=dict(size=15, color='red', symbol='diamond'),
            hovertemplate=f'Market Portfolio<br>Beta: 1.00<br>Expected Return: {rf_rate + market_premium:.2%}<extra></extra>'
        ))
        
        # Add risk-free asset point (beta = 0)
        fig.add_trace(go.Scatter(
            x=[0.0],
            y=[rf_rate],
            mode='markers',
            name='Risk-Free Asset',
            marker=dict(size=12, color='green', symbol='square'),
            hovertemplate=f'Risk-Free Asset<br>Beta: 0.00<br>Return: {rf_rate:.2%}<extra></extra>'
        ))
        
        # Add individual assets
        colors = px.colors.qualitative.Set1
        
        for i, (asset, results) in enumerate(capm_results.items()):
            beta = results['beta']
            expected_ret = results['expected_return_capm']
            actual_ret = results['actual_return']
            alpha = results['jensen_alpha']
            
            # Determine position relative to SML
            above_sml = actual_ret > expected_ret
            
            fig.add_trace(go.Scatter(
                x=[beta],
                y=[actual_ret],
                mode='markers',
                name=f'{asset}',
                marker=dict(
                    size=12,
                    color=colors[i % len(colors)],
                    symbol='circle',
                    line=dict(width=2, color='black' if above_sml else 'gray')
                ),
                hovertemplate=f'{asset}<br>' +
                             f'Beta: {beta:.3f}<br>' +
                             f'Actual Return: {actual_ret:.2%}<br>' +
                             f'Expected Return: {expected_ret:.2%}<br>' +
                             f'Alpha: {alpha:.2%}<br>' +
                             f'{"Above SML" if above_sml else "Below SML"}<extra></extra>'
            ))
            
            # Add arrow showing alpha (distance from SML)
            if abs(alpha) > 0.01:  # Only show significant alphas
                fig.add_annotation(
                    x=beta,
                    y=actual_ret,
                    ax=beta,
                    ay=expected_ret,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='green' if above_sml else 'red',
                    showarrow=True
                )
        
        # Update layout
        fig.update_layout(
            title='Security Market Line (SML) - CAPM Analysis',
            xaxis_title='Beta (Systematic Risk)',
            yaxis_title='Expected Return',
            height=600,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Format y-axis as percentage
        fig.update_layout(yaxis=dict(tickformat='.1%'))
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Security Market Line: {str(e)}")
        return go.Figure()

def create_beta_analysis_chart(capm_data: dict) -> go.Figure:
    """
    Create beta analysis visualization showing beta distribution and interpretation.
    
    Parameters:
    -----------
    capm_data : dict
        CAPM analysis results
    
    Returns:
    --------
    go.Figure
        Plotly figure with beta analysis
    """
    try:
        fig = go.Figure()
        
        if not capm_data or 'capm_results' not in capm_data:
            return fig
        
        capm_results = capm_data['capm_results']
        
        # Extract beta values and asset names
        assets = []
        betas = []
        colors = []
        
        for asset, results in capm_results.items():
            beta = results['beta']
            assets.append(asset)
            betas.append(beta)
            
            # Color code based on beta value
            if beta > 1.2:
                colors.append('red')  # High beta (aggressive)
            elif beta > 0.8:
                colors.append('orange')  # Moderate beta
            elif beta > 0.5:
                colors.append('green')  # Low beta (defensive)  
            else:
                colors.append('blue')  # Very low beta
        
        # Create bar chart
        fig.add_trace(go.Bar(
            x=assets,
            y=betas,
            marker_color=colors,
            name='Beta Values',
            hovertemplate='%{x}<br>Beta: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add reference lines
        fig.add_hline(y=1.0, line_dash="dash", line_color="black", 
                     annotation_text="Market Beta (β = 1.0)")
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray",
                     annotation_text="Risk-Free (β = 0)")
        
        # Update layout
        fig.update_layout(
            title='Beta Analysis - Systematic Risk Comparison',
            xaxis_title='Assets',
            yaxis_title='Beta (β)',
            height=500,
            showlegend=False
        )
        
        # Add beta interpretation annotations
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text="<b>Beta Interpretation:</b><br>" +
                 "β > 1.2: High Risk/High Return<br>" +
                 "0.8 < β < 1.2: Market-like<br>" +
                 "0.5 < β < 0.8: Defensive<br>" +
                 "β < 0.5: Very Conservative",
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating beta analysis chart: {str(e)}")
        return go.Figure()

def create_ticker_input() -> List[str]:
    """
    Create dynamic ticker input with autocomplete functionality.
    
    Returns:
    --------
    List[str]
        List of selected tickers
    """
    st.sidebar.header("📊 Ticker Selection")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Single Ticker", "Multiple Tickers", "Upload CSV"]
    )
    
    tickers = []
    
    if input_method == "Single Ticker":
        # Single ticker input with suggestions
        ticker_input = st.sidebar.text_input(
            "Enter Ticker Symbol:",
            value="AAPL",
            help="Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)\nFor cryptocurrencies, use format: BTC-USD, ETH-USD, ADA-USD"
        ).upper().strip()
        
        if ticker_input:
            tickers = [ticker_input]
        
        # Show ticker format examples
        with st.sidebar.expander("📋 Ticker Examples"):
            st.write("**Stocks:** AAPL, MSFT, GOOGL, TSLA")
            st.write("**ETFs:** SPY, QQQ, VTI, VOO")
            st.write("**Crypto:** BTC-USD, ETH-USD, ADA-USD")
            st.write("**International:** NESN.SW, ASML.AS")
    
    elif input_method == "Multiple Tickers":
        # Multiple ticker input
        ticker_text = st.sidebar.text_area(
            "Enter Ticker Symbols (one per line):",
            value="AAPL\nMSFT\nGOOGL",
            help="Enter multiple ticker symbols, one per line"
        )
        
        if ticker_text:
            tickers = [t.strip().upper() for t in ticker_text.split('\n') if t.strip()]
    
    elif input_method == "Upload CSV":
        # CSV upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV with tickers:",
            type=['csv'],
            help="CSV should have a column named 'ticker' or 'symbol'"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # Try to find ticker column
                ticker_col = None
                for col in ['ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER', 'SYMBOL']:
                    if col in df.columns:
                        ticker_col = col
                        break
                
                if ticker_col:
                    tickers = df[ticker_col].astype(str).str.upper().str.strip().tolist()
                    st.sidebar.success(f"Loaded {len(tickers)} tickers from CSV")
                else:
                    st.sidebar.error("CSV must contain a column named 'ticker' or 'symbol'")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {str(e)}")
    
    return tickers

def create_date_range_input() -> tuple:
    """
    Create date range input controls.
    
    Returns:
    --------
    tuple
        (start_date, end_date) as strings
    """
    st.sidebar.header("📅 Date Range")
    
    # Preset date ranges
    preset = st.sidebar.selectbox(
        "Select preset range:",
        ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "10 Years"]
    )
    
    end_date = datetime.now()
    
    if preset == "1 Month":
        start_date = end_date - timedelta(days=30)
    elif preset == "3 Months":
        start_date = end_date - timedelta(days=90)
    elif preset == "6 Months":
        start_date = end_date - timedelta(days=180)
    elif preset == "1 Year":
        start_date = end_date - timedelta(days=365)
    elif preset == "2 Years":
        start_date = end_date - timedelta(days=730)
    elif preset == "5 Years":
        start_date = end_date - timedelta(days=1825)
    elif preset == "10 Years":
        start_date = end_date - timedelta(days=3650)
    else:  # Custom
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input(
            "Start Date:",
            value=datetime.now() - timedelta(days=365)
        )
        end_date = col2.date_input(
            "End Date:",
            value=datetime.now()
        )
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def display_ticker_info(tickers: List[str]):
    """
    Display basic information about selected tickers.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    """
    if not tickers:
        return
    
    st.subheader("📋 Ticker Information")
    
    info_data = []
    for ticker in tickers:
        info = get_ticker_info(ticker)
        info_data.append({
            'Ticker': ticker,
            'Name': info['name'],
            'Sector': info['sector'],
            'Industry': info['industry']
        })
    
    df_info = pd.DataFrame(info_data)
    st.dataframe(df_info, use_container_width=True)

def main():
    """
    Main Streamlit application.
    """
    # App title
    st.title("📈 Portfolio Analyzer")
    st.markdown("*Comprehensive portfolio analysis tool with real-time data*")
    
    # Sidebar inputs
    tickers = create_ticker_input()
    start_date, end_date = create_date_range_input()
    
    # Data type selection
    data_type = st.sidebar.selectbox(
        "Data Type:",
        ["Close", "Adj Close", "OHLC", "All"],
        help="Select what type of price data to analyze"
    )
    
    # Data frequency selection
    frequency_options = {
        "Daily": {"interval": "1d", "annualization_factor": 252},
        "Weekly": {"interval": "1wk", "annualization_factor": 52},
        "Monthly": {"interval": "1mo", "annualization_factor": 12}
    }
    
    selected_frequency = st.sidebar.selectbox(
        "Data Frequency:",
        options=list(frequency_options.keys()),
        index=0,  # Default to Daily
        help="Select the frequency of data points for analysis"
    )
    
    frequency_config = frequency_options[selected_frequency]
    interval = frequency_config["interval"]
    annualization_factor = frequency_config["annualization_factor"]
    
    # Display frequency information
    with st.sidebar.expander("ℹ️ Frequency Impact"):
        st.write(f"**Selected:** {selected_frequency}")
        st.write(f"**Periods/Year:** {annualization_factor}")
        st.write("**Impact on Analysis:**")
        st.write("• Daily: Most granular, 252 trading days/year")
        st.write("• Weekly: Reduced noise, 52 weeks/year")
        st.write("• Monthly: Long-term trends, 12 months/year")
    
    # Main content
    if not tickers:
        st.info("👈 Please select one or more tickers from the sidebar to begin analysis.")
        return
    
    # Display ticker information
    display_ticker_info(tickers)
    
    # Fetch and display data
    st.subheader("📊 Historical Data")
    
    with st.spinner("Fetching data..."):
        if len(tickers) == 1:
            data = fetch_stock_data(tickers[0], start_date, end_date, data_type, interval)
        else:
            if data_type in ["Close", "Adj Close"]:
                data = fetch_multiple_tickers(tickers, start_date, end_date, interval)
            else:
                st.warning("OHLC and All data types are only available for single ticker analysis.")
                data = fetch_multiple_tickers(tickers, start_date, end_date, interval)
    
    if data.empty:
        st.error("No data available for the selected tickers and date range.")
        return
    
    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(data))
    with col2:
        st.metric("Frequency", selected_frequency)
    with col3:
        st.metric("Date Range", f"{len(data)} periods")
    with col4:
        st.metric("Tickers", len(tickers))
    
    # Display raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(data.tail(100), use_container_width=True)
    
    # Basic price chart
    st.subheader("📈 Price Chart")
    
    if len(tickers) == 1 and data_type == "OHLC":
        # Candlestick chart for single ticker OHLC data
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=tickers[0]
        ))
        fig.update_layout(
            title=f"{tickers[0]} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600
        )
    else:
        # Line chart for closing prices
        fig = go.Figure()
        for column in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=column,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Price Movement",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            hovermode='x unified'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Store data in session state for use in other analysis modules
    st.session_state['data'] = data
    st.session_state['tickers'] = tickers
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['interval'] = interval
    st.session_state['annualization_factor'] = annualization_factor
    st.session_state['selected_frequency'] = selected_frequency
    
    # Auto-trigger analysis with default parameters when data changes
    current_data_key = f"{str(tickers)}_{start_date}_{end_date}_{interval}_{data_type}"
    if 'last_data_key' not in st.session_state or st.session_state['last_data_key'] != current_data_key:
        st.session_state['last_data_key'] = current_data_key
        # Trigger auto-calculation with default parameters
        st.session_state['auto_calculate'] = True
        # Clear previous calculations to force recalculation
        for key in ['metrics_df', 'risk_df', 'capm_results', 'correlation_analysis']:
            if key in st.session_state:
                del st.session_state[key]
    
    # Analysis Tabs
    st.subheader("📊 Portfolio Analysis")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", 
        "📊 Return Metrics", 
        "⚖️ Risk Analysis", 
        "📉 CAPM Analysis", 
        "🔗 Correlation", 
        "🎯 Optimization"
    ])
    
    with tab1:
        st.header("📈 Data Overview")
        st.write("**Dataset Summary:**")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Tickers", len(tickers))
        with col2:
            st.metric("Data Points", len(data))
        with col3:
            st.metric("Frequency", selected_frequency)
        with col4:
            st.metric("Date Range", f"{(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days} days")
        with col5:
            if len(tickers) == 1:
                latest_price = data.iloc[-1, 0]
                st.metric("Latest Price", f"${latest_price:.2f}")
            else:
                st.metric("Ann. Factor", annualization_factor)
                st.metric("Assets", len(tickers))
        
        # Data preview
        st.write("**Recent Data Preview:**")
        st.dataframe(data.tail(10), use_container_width=True)
        
        # Basic statistics
        st.write("**Basic Statistics:**")
        if not data.empty:
            stats_df = data.describe()
            st.dataframe(stats_df, use_container_width=True)
    
    with tab2:
        st.header("📊 Return Metrics Analysis")
        
        if not data.empty:
            # Risk-free rate selection with manual button
            col1, col2 = st.columns([3, 1])
            with col1:
                rf_rate_option = st.selectbox(
                    "Select Risk-Free Rate Proxy:",
                    options=[("^IRX", "3-Month Treasury"), ("^FVX", "5-Year Treasury"), 
                            ("^TNX", "10-Year Treasury"), ("^TYX", "30-Year Treasury")],
                    index=2,  # Default to 10-year
                    format_func=lambda x: x[1],
                    key="rf_rate_selection"
                )
                rf_ticker = rf_rate_option[0]
            
            with col2:
                manual_calculate_metrics = st.button("📊 Calculate Metrics", type="primary")
            
            # Trigger calculation either manually or automatically
            trigger_calculation = (
                manual_calculate_metrics or
                st.session_state.get('auto_calculate', False) or
                'metrics_df' not in st.session_state or
                st.session_state.get('rf_ticker') != rf_ticker
            )
            
            if trigger_calculation:
                with st.spinner("Calculating comprehensive performance metrics..."):
                    try:
                        # Calculate comprehensive metrics including excess returns
                        metrics_df = calculate_comprehensive_metrics(
                            data, start_date, end_date, rf_ticker, interval, annualization_factor
                        )
                        
                        if not metrics_df.empty:
                            st.session_state['metrics_df'] = metrics_df
                            st.session_state['rf_ticker'] = rf_ticker
                            st.success(f"✅ Metrics calculated using {rf_ticker} as risk-free rate")
                        else:
                            st.error("Unable to calculate metrics. Please check your data.")
                    
                    except Exception as e:
                        st.error(f"Error calculating metrics: {str(e)}")
            
            # Display metrics if available
            if 'metrics_df' in st.session_state and not st.session_state['metrics_df'].empty:
                metrics_df = st.session_state['metrics_df']
                
                # Key Performance Indicators
                st.subheader("🎯 Key Performance Indicators")
                
                n_assets = len(metrics_df)
                cols = st.columns(min(n_assets, 4))
                
                for i, (asset, row) in enumerate(metrics_df.iterrows()):
                    with cols[i % 4]:
                        ann_return = row.get('Annualized Return', 0)
                        excess_return = row.get('Excess_Return_Ann', ann_return)
                        sharpe = row.get('Annualized Sharpe Ratio', 0)
                        
                        st.metric(
                            label=f"{asset}",
                            value=f"{ann_return:.2%}",
                            delta=f"Excess: {excess_return:.2%}"
                        )
                        st.caption(f"Sharpe: {sharpe:.3f}")
                
                # Detailed Metrics Table
                st.subheader("📋 Comprehensive Metrics Table")
                
                # Format the metrics for display
                display_df = metrics_df.copy()
                
                # Format percentage columns
                percentage_cols = ['Annualized Return', 'Annualized Volatility', 'Excess_Return_Ann', 
                                 'Risk_Free_Rate_Ann', 'VaR (0.05)', 'CVaR (0.05)', 'Max Drawdown']
                
                for col in percentage_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                
                # Format ratio columns
                ratio_cols = ['Annualized Sharpe Ratio', 'Annualized Sortino Ratio', 'Excess_Sharpe_Ratio', 'Calmar Ratio']
                for col in ratio_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                
                # Format other numeric columns
                numeric_cols = ['Skewness', 'Excess Kurtosis']
                for col in numeric_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Return Distribution Analysis
                st.subheader("📈 Return Distribution Analysis")
                
                returns = data.pct_change().dropna()
                if not returns.empty:
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram of returns
                        fig_hist = px.histogram(
                            returns.melt(var_name='Asset', value_name='Daily_Return'),
                            x='Daily_Return', 
                            color='Asset',
                            title="Distribution of Daily Returns",
                            nbins=50
                        )
                        fig_hist.update_layout(height=400)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Box plot of returns
                        fig_box = px.box(
                            returns.melt(var_name='Asset', value_name='Daily_Return'),
                            x='Asset', 
                            y='Daily_Return',
                            title="Return Distribution by Asset"
                        )
                        fig_box.update_layout(height=400)
                        st.plotly_chart(fig_box, use_container_width=True)
                
                # Risk-Return Scatter Plot
                if len(metrics_df) > 1:
                    st.subheader("🎯 Risk-Return Profile")
                    
                    fig_scatter = px.scatter(
                        x=metrics_df['Annualized Volatility'],
                        y=metrics_df['Annualized Return'],
                        text=metrics_df.index,
                        title="Risk-Return Scatter Plot",
                        labels={
                            'x': 'Annualized Volatility (Risk)',
                            'y': 'Annualized Return'
                        }
                    )
                    fig_scatter.update_traces(textposition='top center')
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Downloadable Report
                st.subheader("📥 Download Report")
                
                csv_data = metrics_df.to_csv()
                st.download_button(
                    label="📊 Download Metrics CSV",
                    data=csv_data,
                    file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
        else:
            st.info("👆 Please select tickers and load data to view return metrics analysis.")
    
    with tab3:
        st.header("⚖️ Risk Analysis")
        
        if not data.empty:
            # Risk metrics calculation
            returns = data.pct_change().dropna()
            
            if not returns.empty:
                col1, col2 = st.columns([3, 1])
                with col1:
                    var_level = st.selectbox(
                        "Select VaR Confidence Level:",
                        options=[0.01, 0.05, 0.10],
                        index=1,  # Default to 5%
                        format_func=lambda x: f"{(1-x)*100:.0f}% confidence ({x*100:.0f}% VaR)",
                        key="var_level_selection"
                    )
                
                with col2:
                    manual_calculate_risk = st.button("📊 Calculate Risk Metrics", type="primary")
                
                # Trigger calculation either manually or automatically
                trigger_risk_calculation = (
                    manual_calculate_risk or
                    st.session_state.get('auto_calculate', False) or
                    'risk_df' not in st.session_state or
                    st.session_state.get('var_level') != var_level
                )
                
                if trigger_risk_calculation:
                    with st.spinner("Calculating comprehensive risk metrics..."):
                        try:
                            # Calculate risk metrics using utils functions
                            from utils import calc_risk_metrics
                            
                            risk_df = calc_risk_metrics(returns, as_df=True, var=var_level)
                            
                            if not risk_df.empty:
                                st.session_state['risk_df'] = risk_df
                                st.session_state['var_level'] = var_level
                                st.success(f"✅ Risk metrics calculated at {(1-var_level)*100:.0f}% confidence level")
                        
                        except Exception as e:
                            st.error(f"Error calculating risk metrics: {str(e)}")
                
                # Display risk metrics if available
                if 'risk_df' in st.session_state and not st.session_state['risk_df'].empty:
                    risk_df = st.session_state['risk_df']
                    current_var_level = st.session_state.get('var_level', 0.05)
                    
                    # Key Risk Indicators
                    st.subheader("🚨 Key Risk Indicators")
                    
                    n_assets = len(risk_df)
                    cols = st.columns(min(n_assets, 4))
                    
                    for i, (asset, row) in enumerate(risk_df.iterrows()):
                        with cols[i % 4]:
                            max_dd = row.get('Max Drawdown', 0)
                            var_value = row.get(f'VaR ({current_var_level})', 0)
                            
                            st.metric(
                                label=f"{asset}",
                                value=f"{max_dd:.2%}",
                                delta=f"VaR: {var_value:.2%}",
                                delta_color="inverse"
                            )
                            
                            # Additional risk info
                            skew = row.get('Skewness', 0)
                            kurt = row.get('Excess Kurtosis', 0)
                            st.caption(f"Skew: {skew:.2f} | Kurt: {kurt:.2f}")
                    
                    # Detailed Risk Metrics Table
                    st.subheader("📋 Comprehensive Risk Metrics")
                    
                    # Format the risk metrics for display
                    display_risk_df = risk_df.copy()
                    
                    # Format percentage columns
                    percentage_cols = [f'VaR ({current_var_level})', f'CVaR ({current_var_level})', 'Max Drawdown', 'Min', 'Max']
                    for col in percentage_cols:
                        if col in display_risk_df.columns:
                            display_risk_df[col] = display_risk_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                    
                    # Format other numeric columns
                    numeric_cols = ['Skewness', 'Excess Kurtosis']
                    for col in numeric_cols:
                        if col in display_risk_df.columns:
                            display_risk_df[col] = display_risk_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                    
                    # Format duration column
                    if 'Duration (days)' in display_risk_df.columns:
                        display_risk_df['Duration (days)'] = display_risk_df['Duration (days)'].apply(
                            lambda x: f"{x} days" if x != "-" and pd.notnull(x) else "N/A"
                        )
                    
                    st.dataframe(display_risk_df, use_container_width=True)
                    
                    # Drawdown Analysis
                    st.subheader("� Drawdown Analysis")
                    
                    # Calculate wealth index and drawdowns for visualization
                    wealth_index = 1000 * (1 + returns).cumprod()
                    previous_peaks = wealth_index.cummax()
                    drawdowns = (wealth_index - previous_peaks) / previous_peaks
                    
                    # Plot drawdown chart
                    fig_dd = go.Figure()
                    
                    for col in drawdowns.columns:
                        fig_dd.add_trace(go.Scatter(
                            x=drawdowns.index,
                            y=drawdowns[col] * 100,  # Convert to percentage
                            mode='lines',
                            name=col,
                            fill='tonexty' if col == drawdowns.columns[0] else None,
                            fillcolor='rgba(255, 0, 0, 0.3)' if col == drawdowns.columns[0] else None
                        ))
                    
                    fig_dd.update_layout(
                        title="Portfolio Drawdown Over Time",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        hovermode='x unified',
                        height=400
                    )
                    
                    # Add horizontal line at 0
                    fig_dd.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                    # Risk Distribution Analysis
                    st.subheader("📊 Risk Distribution Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # VaR visualization
                        fig_var = go.Figure()
                        
                        for col in returns.columns:
                            col_returns = returns[col].dropna()
                            var_cutoff = col_returns.quantile(current_var_level)
                            
                            # Create histogram
                            fig_var.add_trace(go.Histogram(
                                x=col_returns * 100,  # Convert to percentage
                                name=col,
                                nbinsx=50,
                                opacity=0.7
                            ))
                            
                            # Add VaR line
                            fig_var.add_vline(
                                x=var_cutoff * 100,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"VaR {col}"
                            )
                        
                        fig_var.update_layout(
                            title=f"Return Distribution with {(1-current_var_level)*100:.0f}% VaR",
                            xaxis_title="Daily Return (%)",
                            yaxis_title="Frequency",
                            height=400
                        )
                        
                        st.plotly_chart(fig_var, use_container_width=True)
                    
                    with col2:
                        # Rolling volatility
                        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)  # 30-day rolling annual vol
                        
                        fig_vol = go.Figure()
                        
                        for col in rolling_vol.columns:
                            fig_vol.add_trace(go.Scatter(
                                x=rolling_vol.index,
                                y=rolling_vol[col] * 100,  # Convert to percentage
                                mode='lines',
                                name=col
                            ))
                        
                        fig_vol.update_layout(
                            title="30-Day Rolling Volatility",
                            xaxis_title="Date",
                            yaxis_title="Annualized Volatility (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # Risk Summary Dashboard
                    if len(risk_df) > 1:
                        st.subheader("🎯 Risk Comparison Dashboard")
                        
                        # Create risk comparison radar chart
                        fig_radar = go.Figure()
                        
                        # Normalize metrics for radar chart (0-1 scale)
                        metrics_for_radar = ['Skewness', 'Excess Kurtosis', 'Max Drawdown']
                        available_metrics = [m for m in metrics_for_radar if m in risk_df.columns]
                        
                        if available_metrics:
                            for asset in risk_df.index:
                                values = []
                                labels = []
                                
                                for metric in available_metrics:
                                    val = risk_df.loc[asset, metric]
                                    if pd.notnull(val):
                                        # Normalize different metrics differently
                                        if metric == 'Max Drawdown':
                                            normalized_val = abs(val) * 100  # Convert to positive percentage
                                        elif metric in ['Skewness', 'Excess Kurtosis']:
                                            normalized_val = abs(val)  # Absolute value
                                        else:
                                            normalized_val = abs(val) if val is not None else 0
                                        
                                        values.append(normalized_val)
                                        labels.append(metric)
                                
                                if values:
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=labels,
                                        fill='toself',
                                        name=asset,
                                        opacity=0.6
                                    ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, max([max(values) for values in [[] if not hasattr(trace, 'r') else trace.r for trace in fig_radar.data] if values]) if fig_radar.data else 1]
                                    )
                                ),
                                title="Risk Profile Comparison",
                                height=400
                            )
                            
                            st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Downloadable Risk Report
                    st.subheader("📥 Download Risk Report")
                    
                    # Combine original data with risk metrics
                    combined_data = pd.concat([
                        risk_df,
                        drawdowns.describe().T.add_suffix('_Drawdown_Stats')
                    ], axis=1)
                    
                    csv_data = combined_data.to_csv()
                    st.download_button(
                        label="📊 Download Risk Analysis CSV",
                        data=csv_data,
                        file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            else:
                st.warning("📊 Unable to calculate returns. Please check your data.")
        
        else:
            st.info("👆 Please select tickers and load data to view risk analysis.")
    
    with tab4:
        st.header("📈 CAPM (Capital Asset Pricing Model) Analysis")
        
        if data is None or data.empty:
            st.warning("Please load data in the Overview tab first.")
        else:
            try:
                # CAPM Configuration
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    # Market benchmark selection
                    market_options = {
                        'S&P 500': 'SPY',
                        'NASDAQ 100': 'QQQ', 
                        'Russell 2000': 'IWM',
                        'Total Stock Market': 'VTI',
                        'International Developed': 'EFA',
                        'Emerging Markets': 'EEM',
                        'Custom': 'CUSTOM'
                    }
                    
                    selected_market_name = st.selectbox(
                        "Market Benchmark",
                        options=list(market_options.keys()),
                        index=0,
                        help="Select market benchmark for CAPM analysis"
                    )
                    
                    if selected_market_name == 'Custom':
                        market_ticker = st.text_input(
                            "Enter Custom Market Ticker",
                            value="SPY",
                            help="Enter a custom market benchmark ticker"
                        ).upper()
                    else:
                        market_ticker = market_options[selected_market_name]
                
                with col2:
                    # Risk-free rate selection (reuse from return metrics)
                    rf_options = {
                        '3-Month Treasury': '3m',
                        '5-Year Treasury': '5y', 
                        '10-Year Treasury': '10y',
                        '30-Year Treasury': '30y'
                    }
                    
                    selected_rf = st.selectbox(
                        "Risk-Free Rate",
                        options=list(rf_options.keys()),
                        index=0,
                        help="Select risk-free rate for CAPM calculations"
                    )
                    
                    rf_tenor = rf_options[selected_rf]
                
                with col3:
                    # Analysis period selection
                    max_lookback = len(data)
                    capm_period = st.selectbox(
                        "Analysis Period",
                        options=['1 Year', '2 Years', '3 Years', '5 Years', 'All Data'],
                        index=2,
                        help="Select time period for CAPM analysis",
                        key="capm_period_selection"
                    )
                
                # Convert period to days
                period_mapping = {
                    '1 Year': 252,
                    '2 Years': 504, 
                    '3 Years': 756,
                    '5 Years': 1260,
                    'All Data': max_lookback
                }
                analysis_days = min(period_mapping[capm_period], max_lookback)
                
                # Manual CAPM calculation button
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write("")  # Empty space
                with col2:
                    manual_calculate_capm = st.button("🔍 Run CAPM Analysis", type="primary")
                with col3:
                    st.write("")  # Empty space
                
                # Trigger calculation either manually or automatically
                trigger_capm_calculation = (
                    manual_calculate_capm or
                    st.session_state.get('auto_calculate', False) or
                    'capm_analysis' not in st.session_state or
                    st.session_state.get('capm_market_ticker') != market_ticker or
                    st.session_state.get('capm_rf_tenor') != rf_tenor or
                    st.session_state.get('caamp_period') != capm_period
                )
                
                if trigger_capm_calculation:
                    
                    with st.spinner("Fetching market data and calculating CAPM metrics..."):
                        
                        # Get analysis period data
                        analysis_data = data.tail(analysis_days).copy()
                        analysis_returns = analysis_data.pct_change().dropna()
                        
                        # Fetch market benchmark data
                        try:
                            market_data = fetch_multiple_tickers([market_ticker], 
                                                                analysis_data.index[0].strftime('%Y-%m-%d'), 
                                                                analysis_data.index[-1].strftime('%Y-%m-%d'))
                            
                            if market_data.empty:
                                st.error(f"Could not fetch market data for {market_ticker}")
                                st.stop()
                            
                            # Ensure we have the right column structure
                            if market_ticker in market_data.columns:
                                market_returns = market_data[[market_ticker]].pct_change().dropna()
                            else:
                                # Take the first (and likely only) column
                                market_returns = market_data.iloc[:, :1].pct_change().dropna()
                                market_returns.columns = [market_ticker]
                        
                            if market_returns.empty:
                                st.error(f"Could not calculate returns for market data {market_ticker}")
                                st.stop()
                                
                        except Exception as e:
                            st.error(f"Error fetching market data: {str(e)}")
                            st.stop()
                        
                        # Fetch risk-free rate
                        try:
                            # Map rf_tenor to appropriate ticker symbol
                            rf_ticker_mapping = {
                                '3m': '^IRX',
                                '5y': '^FVX', 
                                '10y': '^TNX',
                                '30y': '^TYX'
                            }
                            rf_ticker = rf_ticker_mapping.get(rf_tenor, '^TNX')
                            
                            rf_rate_df = fetch_risk_free_rate(
                                analysis_data.index[0].strftime('%Y-%m-%d'), 
                                analysis_data.index[-1].strftime('%Y-%m-%d'),
                                rf_ticker
                            )
                            
                            if rf_rate_df.empty:
                                st.error(f"Could not fetch {rf_tenor} Treasury rate data")
                                st.stop()
                                
                        except Exception as e:
                            st.error(f"Error fetching risk-free rate: {str(e)}")
                            st.stop()
                        
                        # Perform CAPM analysis
                        capm_analysis = calculate_capm_analysis(
                            analysis_returns, 
                            market_returns, 
                            rf_rate_df,
                            market_ticker
                        )
                        
                        if not capm_analysis:
                            st.error("Could not perform CAPM analysis. Please check your data and try again.")
                            st.stop()
                        
                        # Store in session state
                        st.session_state.capm_analysis = capm_analysis
                        st.session_state.capm_market_ticker = market_ticker
                        st.session_state.capm_rf_rate = rf_rate_df['Risk_Free_Rate'].mean() * 252
                
                # Display results if available
                if 'capm_analysis' in st.session_state and st.session_state.capm_analysis:
                    
                    capm_data = st.session_state.capm_analysis
                    market_ticker = st.session_state.capm_market_ticker
                    rf_annual = st.session_state.capm_rf_rate
                    
                    st.success(f"✅ CAPM Analysis completed for {len(capm_data['capm_results'])} assets")
                    
                    # Analysis Summary
                    st.subheader("📊 CAPM Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Market Benchmark", 
                            market_ticker,
                            help="Market index used for CAPM calculations"
                        )
                    
                    with col2:
                        st.metric(
                            "Analysis Period",
                            capm_data['analysis_period'].split(' to ')[1][:10],
                            help="End date of analysis period"
                        )
                    
                    with col3:
                        st.metric(
                            "Risk-Free Rate (Annual)",
                            f"{rf_annual:.2%}",
                            help="Annualized risk-free rate used"
                        )
                    
                    with col4:
                        market_return = capm_data.get('market_return_annual', 0)
                        st.metric(
                            "Market Return (Annual)",
                            f"{market_return:.2%}",
                            help="Annualized market benchmark return"
                        )
                    
                    # CAPM Results Table
                    st.subheader("🔍 CAPM Metrics by Asset")
                    
                    # Create results dataframe
                    capm_results_list = []
                    for asset, results in capm_data['capm_results'].items():
                        capm_results_list.append({
                            'Asset': asset,
                            'Beta (β)': results['beta'],
                            'Alpha (Annual)': results['alpha_annualized'],
                            'Expected Return (CAPM)': results['expected_return_capm'],
                            'Actual Return': results['actual_return'],
                            'Jensen Alpha': results['jensen_alpha'],
                            'R-Squared': results['r_squared'],
                            'Treynor Ratio': results['treynor_ratio'],
                            'Information Ratio': results['information_ratio'],
                            'Tracking Error': results['tracking_error'],
                            'Downside Beta': results['downside_beta'],
                            'Observations': results['observations']
                        })
                    
                    capm_df = pd.DataFrame(capm_results_list)
                    
                    # Format the dataframe for display
                    capm_df_display = capm_df.copy()
                    
                    # Format percentage columns
                    percentage_cols = ['Alpha (Annual)', 'Expected Return (CAPM)', 'Actual Return', 
                                     'Jensen Alpha', 'R-Squared', 'Treynor Ratio', 
                                     'Information Ratio', 'Tracking Error']
                    
                    for col in percentage_cols:
                        if col in capm_df_display.columns:
                            capm_df_display[col] = capm_df_display[col].apply(lambda x: f"{x:.2%}")
                    
                    # Format numerical columns  
                    numerical_cols = ['Beta (β)', 'Downside Beta']
                    for col in numerical_cols:
                        if col in capm_df_display.columns:
                            capm_df_display[col] = capm_df_display[col].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(
                        capm_df_display,
                        use_container_width=True,
                        height=300
                    )
                    
                    # Security Market Line
                    st.subheader("📈 Security Market Line (SML)")
                    
                    sml_fig = create_security_market_line(capm_data, rf_annual)
                    st.plotly_chart(sml_fig, use_container_width=True)
                    
                    st.info("""
                    **Security Market Line Interpretation:**
                    - Assets **above** the SML have positive alpha (outperforming)
                    - Assets **below** the SML have negative alpha (underperforming)  
                    - The SML shows the expected return for any given level of systematic risk (beta)
                    """)
                    
                    # Beta Analysis
                    st.subheader("⚖️ Beta Analysis")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        beta_fig = create_beta_analysis_chart(capm_data)
                        st.plotly_chart(beta_fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Beta Interpretation:**")
                        st.markdown("• **β > 1.2**: High risk/return (Aggressive)")
                        st.markdown("• **0.8 < β < 1.2**: Market-like risk")  
                        st.markdown("• **0.5 < β < 0.8**: Defensive stocks")
                        st.markdown("• **β < 0.5**: Very conservative")
                        st.markdown("• **β < 0**: Negative correlation with market")
                        
                        # Beta statistics
                        betas = [results['beta'] for results in capm_data['capm_results'].values()]
                        st.markdown("**Portfolio Beta Stats:**")
                        st.markdown(f"• Average: {np.mean(betas):.3f}")
                        st.markdown(f"• Median: {np.median(betas):.3f}")
                        st.markdown(f"• Min: {np.min(betas):.3f}")
                        st.markdown(f"• Max: {np.max(betas):.3f}")
                    
                    # Alpha Analysis
                    st.subheader("🎯 Alpha Analysis")
                    
                    # Create alpha comparison chart
                    alpha_fig = go.Figure()
                    
                    assets = list(capm_data['capm_results'].keys())
                    alphas = [results['jensen_alpha'] for results in capm_data['capm_results'].values()]
                    colors = ['green' if alpha > 0 else 'red' for alpha in alphas]
                    
                    alpha_fig.add_trace(go.Bar(
                        x=assets,
                        y=alphas,
                        marker_color=colors,
                        name='Jensen Alpha',
                        hovertemplate='%{x}<br>Alpha: %{y:.2%}<extra></extra>'
                    ))
                    
                    alpha_fig.add_hline(y=0, line_dash="dash", line_color="black")
                    
                    alpha_fig.update_layout(
                        title='Jensen Alpha - Risk-Adjusted Outperformance',
                        xaxis_title='Assets',
                        yaxis_title='Alpha (Annualized)',
                        yaxis=dict(tickformat='.1%'),
                        height=400
                    )
                    
                    st.plotly_chart(alpha_fig, use_container_width=True)
                    
                    # Download CAPM results
                    st.subheader("💾 Download CAPM Analysis")
                    
                    # Prepare download data
                    download_data = capm_df.copy()
                    
                    csv_data = download_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CAPM Analysis (CSV)",
                        data=csv_data,
                        file_name=f"capm_analysis_{market_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        help="Download detailed CAPM analysis results"
                    )
                    
                    # Key insights
                    st.subheader("💡 Key Insights")
                    
                    # Find best and worst performers
                    best_alpha = max(capm_data['capm_results'].items(), 
                                   key=lambda x: x[1]['jensen_alpha'])
                    worst_alpha = min(capm_data['capm_results'].items(),
                                    key=lambda x: x[1]['jensen_alpha'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"""
                        **🏆 Best Alpha Generator:**
                        - **{best_alpha[0]}**: {best_alpha[1]['jensen_alpha']:.2%} alpha
                        - Beta: {best_alpha[1]['beta']:.3f}
                        - Information Ratio: {best_alpha[1]['information_ratio']:.3f}
                        """)
                    
                    with col2:
                        st.error(f"""
                        **📉 Worst Alpha Generator:**
                        - **{worst_alpha[0]}**: {worst_alpha[1]['jensen_alpha']:.2%} alpha  
                        - Beta: {worst_alpha[1]['beta']:.3f}
                        - Information Ratio: {worst_alpha[1]['information_ratio']:.3f}
                        """)
                    
                    # Overall portfolio insights
                    avg_beta = np.mean([results['beta'] for results in capm_data['capm_results'].values()])
                    avg_alpha = np.mean([results['jensen_alpha'] for results in capm_data['capm_results'].values()])
                    
                    st.info(f"""
                    **📊 Portfolio CAPM Summary:**
                    - **Average Beta**: {avg_beta:.3f} ({'More aggressive' if avg_beta > 1 else 'More defensive'} than market)
                    - **Average Alpha**: {avg_alpha:.2%} ({'Outperforming' if avg_alpha > 0 else 'Underperforming'} vs. CAPM expectation)
                    - **Market Benchmark**: {market_ticker} ({selected_market_name})
                    - **Analysis Period**: {capm_data['total_observations']} trading days
                    """)
                    
            except Exception as e:
                st.error(f"Error in CAPM analysis: {str(e)}")
                st.exception(e)
        
        # Correlation Analysis Tab
        with tab5:
            st.header("🔗 Correlation Analysis")
            
            if not data.empty and len(tickers) > 1:
                # Calculate returns for correlation analysis
                returns = data.pct_change().dropna()
                
                if not returns.empty and len(returns.columns) > 1:
                    # Correlation analysis options
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        correlation_method = st.selectbox(
                            "Correlation Method:",
                            options=["pearson", "spearman", "kendall"],
                            index=0,
                            help="Pearson: Linear relationships, Spearman: Monotonic relationships, Kendall: Non-parametric",
                            key="correlation_method_selection"
                        )
                    
                    with col2:
                        rolling_window = st.number_input(
                            "Rolling Window (days):",
                            min_value=10,
                            max_value=min(252, len(returns)),
                            value=30,
                            help="Window size for rolling correlation calculation",
                            key="rolling_window_selection"
                        )
                    
                    with col3:
                        manual_calculate_corr = st.button("📊 Calculate Correlations", type="primary")
                    
                    # Trigger calculation either manually or automatically
                    trigger_corr_calculation = (
                        manual_calculate_corr or
                        st.session_state.get('auto_calculate', False) or
                        'corr_matrix' not in st.session_state or
                        st.session_state.get('corr_method') != correlation_method or
                        st.session_state.get('rolling_window') != rolling_window
                    )
                    
                    if trigger_corr_calculation:
                        with st.spinner("Calculating correlation analysis..."):
                            try:
                                # Calculate correlation matrix
                                corr_matrix = returns.corr(method=correlation_method)
                                
                                # Calculate rolling correlations
                                rolling_data = calculate_rolling_correlations(returns, rolling_window)
                                
                                # Calculate diversification metrics
                                div_metrics = calculate_diversification_metrics(corr_matrix)
                                
                                # Store in session state
                                st.session_state['corr_matrix'] = corr_matrix
                                st.session_state['rolling_data'] = rolling_data
                                st.session_state['div_metrics'] = div_metrics
                                st.session_state['corr_method'] = correlation_method
                                st.session_state['rolling_window'] = rolling_window
                                st.success(f"✅ Correlation analysis completed using {correlation_method} method")
                                
                            except Exception as e:
                                st.error(f"Error calculating correlations: {str(e)}")
                    
                    # Display correlation analysis if available
                    if 'corr_matrix' in st.session_state and not st.session_state['corr_matrix'].empty:
                        corr_matrix = st.session_state['corr_matrix']
                        rolling_data = st.session_state['rolling_data']
                        div_metrics = st.session_state['div_metrics']
                        corr_method = st.session_state.get('corr_method', 'pearson')
                        
                        st.success(f"✅ Correlation analysis completed using {corr_method.title()} method")
                        
                        # Key Correlation Insights
                        st.subheader("🎯 Key Correlation Insights")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_corr = div_metrics.get('avg_correlation', 0)
                            st.metric(
                                label="Average Correlation",
                                value=f"{avg_corr:.3f}",
                                help="Average correlation across all asset pairs"
                            )
                        
                        with col2:
                            max_corr = div_metrics.get('max_correlation', 0)
                            st.metric(
                                label="Highest Correlation",
                                value=f"{max_corr:.3f}",
                                help="Strongest positive correlation"
                            )
                        
                        with col3:
                            min_corr = div_metrics.get('min_correlation', 0)
                            st.metric(
                                label="Lowest Correlation",
                                value=f"{min_corr:.3f}",
                                help="Strongest negative correlation"
                            )
                        
                        with col4:
                            div_ratio = div_metrics.get('diversification_ratio', 0)
                            st.metric(
                                label="Diversification Score",
                                value=f"{div_ratio:.3f}",
                                help="Higher values indicate better diversification"
                            )
                        
                        # Correlation Matrix Heatmap
                        st.subheader("🌡️ Correlation Matrix Heatmap")
                        
                        # Create correlation heatmap using plotly
                        fig_heatmap = px.imshow(
                            corr_matrix,
                            color_continuous_scale='RdBu_r',
                            aspect='auto',
                            title=f"Asset Correlation Matrix ({corr_method.title()})",
                            labels=dict(color="Correlation"),
                            zmin=-1,
                            zmax=1
                        )
                        
                        # Add correlation values as text
                        fig_heatmap.update_traces(
                            text=corr_matrix.round(3).values,
                            texttemplate="%{text}",
                            textfont={"size": 10}
                        )
                        
                        fig_heatmap.update_layout(height=500)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # Detailed Correlation Matrix Table
                        st.subheader("📋 Detailed Correlation Matrix")
                        
                        # Format correlation matrix for display
                        display_corr = corr_matrix.round(4)
                        st.dataframe(display_corr, use_container_width=True)
                        
                        # Correlation Insights
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🔗 Highly Correlated Pairs")
                            highly_corr = div_metrics.get('highly_correlated_pairs', pd.Series())
                            
                            if not highly_corr.empty:
                                highly_corr_df = pd.DataFrame({
                                    'Asset Pair': [f"{idx[0]} - {idx[1]}" for idx in highly_corr.index],
                                    'Correlation': highly_corr.values
                                })
                                st.dataframe(highly_corr_df, use_container_width=True)
                                st.caption("⚠️ High correlations (>0.7) may reduce diversification benefits")
                            else:
                                st.info("No highly correlated pairs found (correlation > 0.7)")
                        
                        with col2:
                            st.subheader("↕️ Negatively Correlated Pairs")
                            neg_corr = div_metrics.get('negatively_correlated_pairs', pd.Series())
                            
                            if not neg_corr.empty:
                                neg_corr_df = pd.DataFrame({
                                    'Asset Pair': [f"{idx[0]} - {idx[1]}" for idx in neg_corr.index],
                                    'Correlation': neg_corr.values
                                })
                                st.dataframe(neg_corr_df, use_container_width=True)
                                st.caption("✅ Negative correlations can provide diversification benefits")
                            else:
                                st.info("No significantly negatively correlated pairs found (correlation < -0.3)")
                        
                        # Rolling Correlations Analysis
                        if rolling_data and rolling_data.get('rolling_correlations'):
                            st.subheader("📈 Rolling Correlations Over Time")
                            
                            rolling_corrs = rolling_data['rolling_correlations']
                            
                            # Let user select which pairs to display
                            available_pairs = list(rolling_corrs.keys())
                            
                            if len(available_pairs) > 6:
                                # For many pairs, let user select
                                selected_pairs = st.multiselect(
                                    "Select asset pairs to display:",
                                    options=available_pairs,
                                    default=available_pairs[:6],  # Show first 6 by default
                                    help="Select up to 6 pairs for optimal visualization"
                                )
                            else:
                                selected_pairs = available_pairs
                            
                            if selected_pairs:
                                # Create rolling correlation chart
                                fig_rolling = go.Figure()
                                
                                colors = px.colors.qualitative.Set1
                                
                                for i, pair in enumerate(selected_pairs[:6]):  # Limit to 6 for readability
                                    if pair in rolling_corrs:
                                        rolling_corr_data = rolling_corrs[pair]
                                        fig_rolling.add_trace(go.Scatter(
                                            x=rolling_corr_data.index,
                                            y=rolling_corr_data.values,
                                            mode='lines',
                                            name=pair,
                                            line=dict(color=colors[i % len(colors)], width=2)
                                        ))
                                
                                # Add reference lines
                                fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                                fig_rolling.add_hline(y=0.7, line_dash="dot", line_color="red", opacity=0.7, 
                                                    annotation_text="High Correlation")
                                fig_rolling.add_hline(y=-0.3, line_dash="dot", line_color="green", opacity=0.7,
                                                    annotation_text="Negative Correlation")
                                
                                fig_rolling.update_layout(
                                    title=f"{rolling_window}-Day Rolling Correlations",
                                    xaxis_title="Date",
                                    yaxis_title="Correlation",
                                    height=500,
                                    hovermode='x unified',
                                    yaxis=dict(range=[-1, 1])
                                )
                                
                                st.plotly_chart(fig_rolling, use_container_width=True)
                            
                            # Rolling correlation statistics
                            st.subheader("📊 Rolling Correlation Statistics")
                            
                            rolling_stats = []
                            for pair, corr_data in rolling_corrs.items():
                                if not corr_data.empty:
                                    rolling_stats.append({
                                        'Asset Pair': pair,
                                        'Mean Correlation': corr_data.mean(),
                                        'Std Deviation': corr_data.std(),
                                        'Min Correlation': corr_data.min(),
                                        'Max Correlation': corr_data.max(),
                                        'Latest Correlation': corr_data.iloc[-1] if len(corr_data) > 0 else np.nan
                                    })
                            
                            if rolling_stats:
                                rolling_stats_df = pd.DataFrame(rolling_stats)
                                
                               
                                
                                # Format for display
                                for col in ['Mean Correlation', 'Std Deviation', 'Min Correlation', 'Max Correlation', 'Latest Correlation']:
                                    rolling_stats_df[col] = rolling_stats_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                                
                                st.dataframe(rolling_stats_df, use_container_width=True)
                        
                        # Correlation Distribution Analysis
                        st.subheader("📊 Correlation Distribution")
                        
                        correlation_dist = div_metrics.get('correlation_distribution', pd.Series())
                        
                        if not correlation_dist.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Histogram of correlations
                                fig_hist = px.histogram(
                                    x=correlation_dist.values,
                                    nbins=20,
                                    title="Distribution of Pairwise Correlations",
                                    labels={'x': 'Correlation', 'y': 'Frequency'}
                                )
                                fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
                                fig_hist.update_layout(height=400)
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            with col2:
                                # Box plot of correlations
                                fig_box = px.box(
                                    y=correlation_dist.values,
                                    title="Correlation Distribution Box Plot",
                                    labels={'y': 'Correlation'}
                                )
                                fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
                                fig_box.add_hline(y=0.7, line_dash="dot", line_color="red", opacity=0.7)
                                fig_box.add_hline(y=-0.3, line_dash="dot", line_color="green", opacity=0.7)
                                fig_box.update_layout(height=400)
                                st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Network Graph (for smaller portfolios)
                        if len(tickers) <= 10:
                            st.subheader("🕸️ Correlation Network Graph")
                            
                            # Create network-style visualization
                            fig_network = go.Figure()
                            
                            # Add nodes (assets)
                            n_assets = len(corr_matrix)
                            angles = np.linspace(0, 2*np.pi, n_assets, endpoint=False)
                            x_pos = np.cos(angles)
                            y_pos = np.sin(angles)
                            
                            # Add asset nodes
                            fig_network.add_trace(go.Scatter(
                                x=x_pos,
                                y=y_pos,
                                mode='markers+text',
                                marker=dict(size=30, color='lightblue', line=dict(width=2, color='darkblue')),
                                text=corr_matrix.columns,
                                textposition='middle center',
                                textfont=dict(size=10, color='black'),
                                name='Assets',
                                showlegend=False
                            ))
                            
                            # Add correlation lines
                            for i in range(n_assets):
                                for j in range(i+1, n_assets):
                                    corr_val = corr_matrix.iloc[i, j]
                                    
                                    # Only show significant correlations
                                    if abs(corr_val) > 0.3:
                                        line_color = 'red' if corr_val > 0 else 'blue'
                                        line_width = abs(corr_val) * 5;
                                        
                                        fig_network.add_trace(go.Scatter(
                                            x=[x_pos[i], x_pos[j]],
                                            y=[y_pos[i], y_pos[j]],
                                            mode='lines',
                                            line=dict(color=line_color, width=line_width),
                                            opacity=abs(corr_val),
                                            showlegend=False,
                                            hovertemplate=f"Correlation: {corr_val:.3f}<extra></extra>"
                                        ))
                            
                            fig_network.update_layout(
                                title="Correlation Network (|correlation| > 0.3)",
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=500,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_network, use_container_width=True)
                            st.caption("Red lines: Positive correlations, Blue lines: Negative correlations. Line thickness represents correlation strength.")
                        
                        # Downloadable Correlation Report
                        st.subheader("📥 Download Correlation Report")
                        
                        # Combine all correlation data
                        correlation_report = {
                            'correlation_matrix': corr_matrix,
                            'diversification_metrics': pd.DataFrame([div_metrics]),
                        }
                        
                        # Add rolling correlation stats if available
                        if rolling_stats:
                            correlation_report['rolling_correlation_stats'] = pd.DataFrame(rolling_stats)
                        
                        # Create downloadable CSV
                        csv_buffer = []
                        csv_buffer.append("CORRELATION MATRIX")
                        csv_buffer.append(corr_matrix.to_csv())
                        csv_buffer.append("\nDIVERSIFICATION METRICS")
                        csv_buffer.append(pd.DataFrame([div_metrics]).to_csv())
                        
                        if rolling_stats:
                            csv_buffer.append("\nROLLING CORRELATION STATISTICS")
                            csv_buffer.append(pd.DataFrame(rolling_stats).to_csv())
                        
                        full_csv = "\n".join(csv_buffer)
                        
                        st.download_button(
                            label="📊 Download Correlation Analysis CSV",
                            data=full_csv,
                            file_name=f"correlation_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
            
                else:
                    st.warning("📊 Unable to calculate returns for correlation analysis. Please check your data.")
        
            elif len(tickers) <= 1:
                st.warning("📝 **Note:** Correlation analysis requires multiple tickers. Please select at least 2 assets.")
                
                # Show what correlation analysis offers
                st.info("**🔗 What you'll get with correlation analysis:**")
                st.write("- **Correlation Matrix**: See how assets move together")
                st.write("- **Diversification Insights**: Identify which assets provide the best diversification")
                st.write("- **Rolling Correlations**: Track how relationships change over time")
                st.write("- **Network Visualization**: Visual representation of asset relationships")
                st.write("- **Risk Insights**: Understand concentration risk in your portfolio")
            
            else:
                st.info("👆 Please select tickers and load data to view correlation analysis.")
    
    with tab6:
        st.header("🎯 Portfolio Optimization")
        
        if data is None or data.empty:
            st.warning("Please load data in the Overview tab first.")
        elif len(tickers) < 2:
            st.warning("📝 **Note:** Portfolio optimization requires multiple assets. Please select at least 2 tickers.")
            
            # Show what features will be available
            st.subheader("🔮 Available Features (Multi-Asset)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Optimization Methods:**")
                st.write("• Tangency Portfolio (Maximum Sharpe)")
                st.write("• Global Minimum Variance Portfolio")
                st.write("• Mean-Variance Efficient Portfolios")
                st.write("• Equal Weight Portfolio")
                
            with col2:
                st.write("**📊 Visualizations:**")
                st.write("• Efficient Frontier")
                st.write("• Portfolio Weights Comparison")
                st.write("• Risk-Return Analysis")
                st.write("• Portfolio Performance Over Time")
                
        else:
            # Portfolio optimization for multiple assets
            try:
                # Calculate returns
                returns = data.pct_change().dropna()
                
                if returns.empty:
                    st.error("Unable to calculate returns from the data.")
                else:
                    # Optimization controls
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        rolling_window = st.selectbox(
                            "Rolling Window for Time-Series Analysis:",
                            options=[30, 60, 90, 120],
                            index=1,  # Default to 60 days
                            help="Window size for rolling portfolio metrics analysis"
                        )
                    
                    with col2:
                        manual_calculate_optimization = st.button("🎯 Optimize Portfolios", type="primary")
                    
                    with col3:
                        show_advanced = st.checkbox("Show Advanced Options", value=False)
                    
                    # Advanced options
                    if show_advanced:
                        with st.expander("⚙️ Advanced Optimization Settings"):
                            st.write("**Risk-Free Rate:**")
                            use_custom_rf = st.checkbox("Use custom risk-free rate", value=False)
                            if use_custom_rf:
                                custom_rf_rate = st.number_input(
                                    "Annual risk-free rate (%)", 
                                    min_value=0.0, 
                                    max_value=10.0, 
                                    value=2.0, 
                                    step=0.1
                                ) / 100
                            
                            st.write("**Frontier Resolution:**")
                            frontier_points = st.slider(
                                "Number of efficient frontier points", 
                                min_value=50, 
                                max_value=200, 
                                value=100, 
                                step=10
                            )
                    
                    # Trigger calculation
                    trigger_optimization = (
                        manual_calculate_optimization or
                        st.session_state.get('auto_calculate', False) or
                        'optimal_portfolios' not in st.session_state
                    )
                    
                    if trigger_optimization:
                        with st.spinner("🎯 Calculating optimal portfolios..."):
                            # Calculate optimal portfolios
                            optimal_data = calculate_optimal_portfolios(returns, annualization_factor)
                            
                            if optimal_data:
                                st.session_state['optimal_portfolios'] = optimal_data
                                st.session_state['rolling_window'] = rolling_window
                                st.success("✅ Portfolio optimization completed!")
                            else:
                                st.error("Unable to calculate optimal portfolios.")
                    
                    # Display results if available
                    if 'optimal_portfolios' in st.session_state and st.session_state['optimal_portfolios']:
                        optimal_data = st.session_state['optimal_portfolios']
                        
                        # Key Performance Summary
                        st.subheader("📊 Optimal Portfolio Summary")
                        
                        portfolios = optimal_data['portfolios']
                        
                        # Create metrics display
                        col1, col2, col3 = st.columns(3)
                        
                        portfolio_names = list(portfolios.keys())
                        
                        for i, (portfolio_name, portfolio_data) in enumerate(portfolios.items()):
                            with [col1, col2, col3][i]:
                                expected_return = portfolio_data['expected_return']
                                volatility = portfolio_data['volatility']
                                sharpe_ratio = portfolio_data['sharpe_ratio']
                                
                                st.metric(
                                    label=f"🎯 {portfolio_name}",
                                    value=f"{expected_return:.2%}",
                                    delta=f"Risk: {volatility:.2%}"
                                )
                                st.caption(f"Sharpe: {sharpe_ratio:.3f}")
                        
                        # Efficient Frontier Visualization
                        st.subheader("📈 Efficient Frontier")
                        
                        frontier_fig = create_efficient_frontier_plot(optimal_data, returns)
                        st.plotly_chart(frontier_fig, use_container_width=True)
                        
                        # Portfolio Weights Analysis
                        st.subheader("📊 Portfolio Weights Comparison")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Weights chart
                            weights_fig = create_portfolio_weights_chart(optimal_data)
                            st.plotly_chart(weights_fig, use_container_width=True)
                        
                        with col2:
                            # Weights table
                            st.write("**Detailed Portfolio Weights:**")
                            weights_df = pd.DataFrame({
                                name: data['weights'] 
                                for name, data in portfolios.items()
                            })
                            
                            # Format as percentages
                            weights_display = weights_df.copy()
                            for col in weights_display.columns:
                                weights_display[col] = weights_display[col].apply(lambda x: f"{x:.2%}")
                            
                            st.dataframe(weights_display, use_container_width=True)
                        
                        # Risk-Return Comparison
                        st.subheader("⚖️ Risk-Return Analysis")
                        
                        comparison_df = create_risk_return_comparison(optimal_data)
                        
                        if not comparison_df.empty:
                            # Format the display
                            display_comparison = comparison_df.copy()
                            display_comparison['Expected Return'] = display_comparison['Expected Return'].apply(lambda x: f"{x:.2%}")
                            display_comparison['Volatility'] = display_comparison['Volatility'].apply(lambda x: f"{x:.2%}")
                            display_comparison['Sharpe Ratio'] = display_comparison['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(display_comparison, use_container_width=True)
                        
                        # Portfolio Performance Over Time
                        st.subheader("📈 Portfolio Performance Analysis")
                        
                        # Performance comparison for each optimal portfolio
                        portfolio_tabs = st.tabs([f"📊 {name}" for name in portfolios.keys()])
                        
                        for i, (portfolio_name, portfolio_data) in enumerate(portfolios.items()):
                            with portfolio_tabs[i]:
                                weights = portfolio_data['weights']
                                
                                # Calculate portfolio metrics over time
                                window = st.session_state.get('rolling_window', 60)
                                performance_metrics = calculate_portfolio_metrics_over_time(
                                    returns, weights, window
                                )
                                
                                if not performance_metrics.empty:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Cumulative returns
                                        fig_cumret = go.Figure()
                                        fig_cumret.add_trace(go.Scatter(
                                            x=performance_metrics.index,
                                            y=performance_metrics['Cumulative_Return'],
                                            mode='lines',
                                            name=f'{portfolio_name} Portfolio',
                                            line=dict(width=2)
                                        ))
                                        
                                        # Add individual asset cumulative returns for comparison
                                        for asset in returns.columns:
                                            asset_cumret = (1 + returns[asset]).cumprod()
                                            fig_cumret.add_trace(go.Scatter(
                                                x=asset_cumret.index,
                                                y=asset_cumret,
                                                mode='lines',
                                                name=asset,
                                                line=dict(width=1, dash='dot'),
                                                opacity=0.6
                                            ))
                                        
                                        fig_cumret.update_layout(
                                            title=f'{portfolio_name} - Cumulative Returns',
                                            xaxis_title='Date',
                                            yaxis_title='Cumulative Return',
                                            height=400
                                        )
                                        st.plotly_chart(fig_cumret, use_container_width=True)
                                    
                                    with col2:
                                        # Rolling Sharpe ratio
                                        fig_sharpe = go.Figure()
                                        fig_sharpe.add_trace(go.Scatter(
                                            x=performance_metrics.index,
                                            y=performance_metrics['Rolling_Sharpe'],
                                            mode='lines',
                                            name=f'Rolling Sharpe ({window} days)',
                                            line=dict(width=2)
                                        ))
                                        
                                        fig_sharpe.update_layout(
                                            title=f'{portfolio_name} - Rolling Sharpe Ratio',
                                            xaxis_title='Date',
                                            yaxis_title='Sharpe Ratio',
                                            height=400
                                        )
                                        st.plotly_chart(fig_sharpe, use_container_width=True)
                                    
                                    # Portfolio statistics
                                    st.write(f"**{portfolio_name} Portfolio Statistics:**")
                                    
                                    portfolio_returns = performance_metrics['Portfolio_Return']
                                    total_return = performance_metrics['Cumulative_Return'].iloc[-1] - 1
                                    annualized_return = portfolio_data['expected_return']
                                    annualized_vol = portfolio_data['volatility']
                                    max_drawdown = ((performance_metrics['Cumulative_Return'] / 
                                                   performance_metrics['Cumulative_Return'].cummax()) - 1).min()
                                    
                                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                                    
                                    with stats_col1:
                                        st.metric("Total Return", f"{total_return:.2%}")
                                    with stats_col2:
                                        st.metric("Annualized Return", f"{annualized_return:.2%}")
                                    with stats_col3:
                                        st.metric("Annualized Volatility", f"{annualized_vol:.2%}")
                                    with stats_col4:
                                        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                        
                        # Download Portfolio Data
                        st.subheader("📥 Download Portfolio Data")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Portfolio weights CSV
                            weights_csv = pd.DataFrame({
                                name: data['weights'] 
                                for name, data in portfolios.items()
                            }).to_csv()
                            
                            st.download_button(
                                label="📊 Download Portfolio Weights",
                                data=weights_csv,
                                file_name=f"optimal_portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Portfolio summary CSV
                            summary_csv = comparison_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📈 Download Portfolio Summary",
                                data=summary_csv,
                                file_name=f"portfolio_optimization_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        # Investment Recommendations
                        st.subheader("💡 Investment Insights")
                        
                        # Find best portfolio by Sharpe ratio
                        best_portfolio = max(portfolios.items(), key=lambda x: x[1]['sharpe_ratio'])
                        best_name, best_data = best_portfolio
                        
                        st.success(f"🎯 **Recommended Portfolio:** {best_name}")
                        st.write(f"**Rationale:** This portfolio offers the highest risk-adjusted return (Sharpe ratio: {best_data['sharpe_ratio']:.3f})")
                        
                        # Portfolio insights
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info("📊 **Portfolio Insights:**")
                            
                            # Concentration analysis
                            best_weights = best_data['weights']
                            max_weight = best_weights.max()
                            max_weight_asset = best_weights.idxmax()
                            
                            if max_weight > 0.4:
                                st.write(f"⚠️ High concentration in {max_weight_asset} ({max_weight:.1%})")
                            else:
                                st.write(f"✅ Well diversified portfolio (max weight: {max_weight:.1%})")
                            
                            # Risk assessment
                            if best_data['volatility'] < 0.15:
                                st.write("🔒 Conservative risk profile")
                            elif best_data['volatility'] < 0.25:
                                st.write("⚖️ Moderate risk profile")
                            else:
                                st.write("⚡ Aggressive risk profile")
                        
                        with col2:
                            st.info("🎯 **Optimization Notes:**")
                            st.write("• Tangency portfolio maximizes Sharpe ratio")
                            st.write("• GMV portfolio minimizes volatility")
                            st.write("• Equal weight provides naive diversification")
                            st.write("• Results based on historical data")
                            st.write("• Consider transaction costs in practice")
            
            except Exception as e:
                st.error(f"Error in portfolio optimization: {str(e)}")
                st.write("Please check your data and try again.")
    
    # Reset auto-calculate flag after all tabs have been processed
    if 'auto_calculate' in st.session_state:
        st.session_state['auto_calculate'] = False

if __name__ == "__main__":
    main()
