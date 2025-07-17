#!/usr/bin/env python3
"""
Test script for CAPM analysis functionality
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path to import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from app.py
from app import (
    fetch_multiple_tickers, 
    fetch_risk_free_rate, 
    calculate_capm_analysis,
    create_security_market_line,
    create_beta_analysis_chart
)

def test_fetch_multiple_tickers():
    """Test the fetch_multiple_tickers function"""
    print("🧪 Testing fetch_multiple_tickers...")
    
    try:
        # Test with single ticker
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = fetch_multiple_tickers(['SPY'], start_date, end_date)
        
        if not data.empty:
            print(f"✅ Successfully fetched SPY data: {len(data)} rows")
            print(f"   Columns: {data.columns.tolist()}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            return data
        else:
            print("❌ Failed to fetch SPY data")
            return None
            
    except Exception as e:
        print(f"❌ Error in fetch_multiple_tickers: {str(e)}")
        return None

def test_fetch_risk_free_rate():
    """Test the fetch_risk_free_rate function"""
    print("\n🧪 Testing fetch_risk_free_rate...")
    
    try:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        rf_data = fetch_risk_free_rate(start_date, end_date, '^TNX')
        
        if not rf_data.empty:
            print(f"✅ Successfully fetched risk-free rate data: {len(rf_data)} rows")
            print(f"   Columns: {rf_data.columns.tolist()}")
            print(f"   Average daily rate: {rf_data['Risk_Free_Rate'].mean():.6f}")
            print(f"   Annualized rate: {rf_data['Risk_Free_Rate'].mean() * 252:.4f}")
            return rf_data
        else:
            print("❌ Failed to fetch risk-free rate data")
            return None
            
    except Exception as e:
        print(f"❌ Error in fetch_risk_free_rate: {str(e)}")
        return None

def test_capm_analysis():
    """Test the complete CAPM analysis"""
    print("\n🧪 Testing CAPM analysis...")
    
    try:
        # Set up test data
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch asset data (portfolio)
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        print(f"   Fetching data for: {tickers}")
        
        asset_data = fetch_multiple_tickers(tickers, start_date, end_date)
        if asset_data.empty:
            print("❌ Failed to fetch asset data")
            return False
        
        asset_returns = asset_data.pct_change().dropna()
        print(f"   Asset returns: {len(asset_returns)} observations")
        
        # Fetch market data
        market_data = fetch_multiple_tickers(['SPY'], start_date, end_date)
        if market_data.empty:
            print("❌ Failed to fetch market data")
            return False
        
        market_returns = market_data.pct_change().dropna()
        print(f"   Market returns: {len(market_returns)} observations")
        
        # Fetch risk-free rate
        rf_data = fetch_risk_free_rate(start_date, end_date, '^TNX')
        if rf_data.empty:
            print("❌ Failed to fetch risk-free rate")
            return False
        
        print(f"   Risk-free rate: {len(rf_data)} observations")
        
        # Perform CAPM analysis
        print(f"   Performing CAPM analysis...")
        print(f"   Asset returns shape: {asset_returns.shape}")
        print(f"   Market returns shape: {market_returns.shape}")
        print(f"   Risk-free rate shape: {rf_data.shape}")
        
        capm_results = calculate_capm_analysis(
            asset_returns, 
            market_returns, 
            rf_data,
            'SPY'
        )
        
        print(f"   CAPM results type: {type(capm_results)}")
        print(f"   CAPM results: {capm_results}")
        
        if capm_results and 'capm_results' in capm_results:
            print(f"✅ CAPM analysis successful!")
            print(f"   Assets analyzed: {len(capm_results['capm_results'])}")
            print(f"   Market ticker: {capm_results['market_ticker']}")
            print(f"   Analysis period: {capm_results['analysis_period']}")
            print(f"   Total observations: {capm_results['total_observations']}")
            
            # Display individual asset results
            print("\n   📊 Individual Asset Results:")
            for asset, results in capm_results['capm_results'].items():
                print(f"   {asset}:")
                print(f"      Beta: {results['beta']:.3f}")
                print(f"      Alpha (annual): {results['alpha_annualized']:.2%}")
                print(f"      Jensen Alpha: {results['jensen_alpha']:.2%}")
                print(f"      R-squared: {results['r_squared']:.3f}")
                print(f"      Treynor Ratio: {results['treynor_ratio']:.3f}")
            
            return capm_results
        else:
            print("❌ CAPM analysis failed")
            return None
            
    except Exception as e:
        print(f"❌ Error in CAPM analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_visualizations():
    """Test CAPM visualization functions"""
    print("\n🧪 Testing CAPM visualizations...")
    
    # Run CAPM analysis first
    capm_results = test_capm_analysis()
    
    if not capm_results:
        print("❌ Cannot test visualizations without CAPM results")
        return False
    
    try:
        # Test Security Market Line
        rf_rate = 0.05  # 5% annual risk-free rate
        sml_fig = create_security_market_line(capm_results, rf_rate)
        
        if sml_fig and hasattr(sml_fig, 'data') and len(sml_fig.data) > 0:
            print("✅ Security Market Line visualization created successfully")
            print(f"   Number of traces: {len(sml_fig.data)}")
        else:
            print("❌ Failed to create Security Market Line")
        
        # Test Beta Analysis Chart
        beta_fig = create_beta_analysis_chart(capm_results)
        
        if beta_fig and hasattr(beta_fig, 'data') and len(beta_fig.data) > 0:
            print("✅ Beta analysis chart created successfully")
            print(f"   Number of traces: {len(beta_fig.data)}")
        else:
            print("❌ Failed to create Beta analysis chart")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization tests: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all CAPM tests"""
    print("🚀 Starting CAPM Analysis Tests")
    print("=" * 50)
    
    # Test individual components
    market_data = test_fetch_multiple_tickers()
    rf_data = test_fetch_risk_free_rate()
    
    if market_data is not None and rf_data is not None:
        print("\n✅ Basic data fetching tests passed")
        
        # Test complete CAPM analysis
        capm_results = test_capm_analysis()
        
        if capm_results:
            print("\n✅ CAPM analysis test passed")
            
            # Test visualizations
            viz_success = test_visualizations()
            
            if viz_success:
                print("\n🎉 All CAPM tests passed successfully!")
                print("\n📋 Test Summary:")
                print("   ✅ Data fetching")
                print("   ✅ Risk-free rate fetching")
                print("   ✅ CAPM calculations")
                print("   ✅ Visualizations")
                return True
            else:
                print("\n⚠️  CAPM calculations work but visualizations failed")
                return False
        else:
            print("\n❌ CAPM analysis failed")
            return False
    else:
        print("\n❌ Basic data fetching failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
