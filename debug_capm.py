#!/usr/bin/env python3
"""
Simple debug script for CAPM analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import fetch_multiple_tickers, fetch_risk_free_rate, calculate_excess_returns
from utils import calc_univariate_regression

def debug_capm():
    """Debug CAPM analysis step by step"""
    print("🔍 Debugging CAPM Analysis")
    
    # Get test data
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch data
    print("1. Fetching asset data...")
    asset_data = fetch_multiple_tickers(['AAPL'], start_date, end_date)
    print(f"   Raw asset data columns: {asset_data.columns}")
    print(f"   Raw asset data shape: {asset_data.shape}")
    print(f"   Raw asset data head:")
    print(asset_data.head())
    
    asset_returns = asset_data.pct_change().dropna()
    print(f"   Asset returns shape: {asset_returns.shape}")
    print(f"   Asset returns columns: {asset_returns.columns}")
    print(f"   Asset returns index type: {type(asset_returns.index)}")
    print(f"   Asset returns index sample: {asset_returns.index[:5]}")
    
    print("2. Fetching market data...")
    market_data = fetch_multiple_tickers(['SPY'], start_date, end_date)
    market_returns = market_data.pct_change().dropna()
    print(f"   Market returns shape: {market_returns.shape}")
    print(f"   Market returns columns: {market_returns.columns}")
    
    print("3. Fetching risk-free rate...")
    rf_data = fetch_risk_free_rate(start_date, end_date, '^TNX')
    print(f"   RF data shape: {rf_data.shape}")
    print(f"   RF data columns: {rf_data.columns}")
    print(f"   RF data index type: {type(rf_data.index)}")
    print(f"   RF data index sample: {rf_data.index[:5]}")
    print(f"   RF data head:")
    print(rf_data.head())
    
    print("4. Aligning dates...")
    common_dates = asset_returns.index.intersection(market_returns.index).intersection(rf_data.index)
    print(f"   Common dates: {len(common_dates)}")
    print(f"   Date range: {common_dates[0]} to {common_dates[-1]}")
    
    if len(common_dates) < 30:
        print("   ❌ Insufficient data!")
        return
    
    # Filter to common dates
    asset_returns_aligned = asset_returns.loc[common_dates]
    market_returns_aligned = market_returns.loc[common_dates]
    rf_rate_aligned = rf_data.loc[common_dates]
    
    print("5. Calculating excess returns...")
    try:
        asset_excess_returns = calculate_excess_returns(asset_returns_aligned, rf_rate_aligned)
        market_excess_returns = calculate_excess_returns(market_returns_aligned, rf_rate_aligned)
        print(f"   Asset excess returns shape: {asset_excess_returns.shape}")
        print(f"   Market excess returns shape: {market_excess_returns.shape}")
        print(f"   Asset excess returns columns: {asset_excess_returns.columns}")
        print(f"   Market excess returns columns: {market_excess_returns.columns}")
    except Exception as e:
        print(f"   ❌ Error calculating excess returns: {e}")
        return
    
    print("6. Testing regression for one asset...")
    asset = asset_excess_returns.columns[0]
    print(f"   Testing asset: {asset}")
    
    try:
        # Prepare data for regression
        asset_excess = asset_excess_returns[[asset]].dropna()
        market_excess_common = market_excess_returns.reindex(asset_excess.index, method='ffill')
        
        print(f"   Asset excess data shape: {asset_excess.shape}")
        print(f"   Market excess data shape: {market_excess_common.shape}")
        print(f"   Asset excess first 5 values:")
        print(asset_excess.head())
        print(f"   Market excess first 5 values:")
        print(market_excess_common.head())
        
        # Try the regression
        print("   Running regression...")
        regression_results = calc_univariate_regression(
            y=asset_excess, 
            X=market_excess_common, 
            intercept=True, 
            adj=252
        )
        
        print(f"   Regression results type: {type(regression_results)}")
        print(f"   Regression results shape: {regression_results.shape if hasattr(regression_results, 'shape') else 'No shape'}")
        print(f"   Regression results:")
        print(regression_results)
        
        if not regression_results.empty:
            print("   ✅ Regression successful!")
            for col in regression_results.columns:
                print(f"      {col}: {regression_results.loc[asset, col]}")
        else:
            print("   ❌ Regression returned empty results")
            
    except Exception as e:
        print(f"   ❌ Error in regression: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_capm()
