#!/usr/bin/env python3
"""
Test script for the Return Metrics functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    fetch_risk_free_rate, 
    calculate_excess_returns, 
    calculate_comprehensive_metrics,
    fetch_stock_data
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_risk_free_rate():
    """Test risk-free rate fetching"""
    print("Testing risk-free rate fetching...")
    
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    # Test with different Treasury rates
    for rate_type, name in [("^IRX", "3-Month"), ("^TNX", "10-Year")]:
        print(f"\nTesting {name} Treasury ({rate_type}):")
        
        rf_data = fetch_risk_free_rate(start_date, end_date, rate_type)
        
        if not rf_data.empty:
            print(f"✅ Successfully fetched {len(rf_data)} data points")
            print(f"   Average daily rate: {rf_data['Risk_Free_Rate'].mean():.6f}")
            print(f"   Annualized rate: {rf_data['Risk_Free_Rate'].mean() * 252:.4%}")
        else:
            print("❌ No data returned")

def test_excess_returns():
    """Test excess returns calculation"""
    print("\n" + "="*50)
    print("Testing excess returns calculation...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    
    # Sample returns data
    returns_data = pd.DataFrame({
        'AAPL': [0.01, -0.02, 0.015, 0.008, -0.01, 0.02, -0.005, 0.012, 0.003, 0.007],
        'MSFT': [0.008, -0.015, 0.02, 0.005, -0.012, 0.018, 0.001, 0.014, -0.002, 0.009]
    }, index=dates)
    
    # Sample risk-free rate (daily)
    rf_data = pd.DataFrame({
        'Risk_Free_Rate': [0.0001] * len(dates)  # ~2.5% annual rate
    }, index=dates)
    
    excess_returns = calculate_excess_returns(returns_data, rf_data)
    
    print(f"✅ Sample excess returns calculated")
    print(f"   Original AAPL mean return: {returns_data['AAPL'].mean():.4f}")
    print(f"   Excess AAPL return: {excess_returns['AAPL'].mean():.4f}")
    print(f"   Difference (should equal RF rate): {returns_data['AAPL'].mean() - excess_returns['AAPL'].mean():.6f}")

def test_comprehensive_metrics():
    """Test comprehensive metrics calculation with real data"""
    print("\n" + "="*50)
    print("Testing comprehensive metrics calculation...")
    
    # Fetch real data for testing
    start_date = "2024-01-01"
    end_date = "2024-02-01"
    
    print("Fetching AAPL data...")
    aapl_data = fetch_stock_data("AAPL", start_date, end_date, "Close")
    
    if not aapl_data.empty:
        print(f"✅ Fetched {len(aapl_data)} data points")
        
        # Calculate comprehensive metrics
        print("Calculating comprehensive metrics...")
        metrics = calculate_comprehensive_metrics(aapl_data, start_date, end_date, "^TNX")
        
        if not metrics.empty:
            print("✅ Metrics calculated successfully!")
            print("\nKey Metrics:")
            
            asset = metrics.index[0]
            print(f"Asset: {asset}")
            
            # Show key metrics
            for metric in ['Annualized Return', 'Annualized Volatility', 'Annualized Sharpe Ratio', 
                          'Risk_Free_Rate_Ann', 'Excess_Return_Ann']:
                if metric in metrics.columns:
                    value = metrics.loc[asset, metric]
                    if pd.notnull(value):
                        if 'Return' in metric or 'Volatility' in metric or 'Rate' in metric:
                            print(f"   {metric}: {value:.2%}")
                        else:
                            print(f"   {metric}: {value:.3f}")
        else:
            print("❌ No metrics calculated")
    else:
        print("❌ No price data available")

def main():
    """Run all tests"""
    print("🧪 Testing Return Metrics Functionality")
    print("="*60)
    
    try:
        test_risk_free_rate()
        test_excess_returns()
        test_comprehensive_metrics()
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("\n💡 To see the full functionality:")
        print("   1. Open http://localhost:8501 in your browser")
        print("   2. Select tickers (e.g., AAPL, MSFT)")
        print("   3. Click on the 'Return Metrics' tab")
        print("   4. Click 'Calculate Metrics' button")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
