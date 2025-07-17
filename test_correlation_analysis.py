#!/usr/bin/env python3
"""
Test script for the Correlation Analysis functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_correlation_functions():
    """Test correlation analysis functions"""
    print("🔗 Testing Correlation Analysis Functions")
    print("="*60)
    
    # Import functions
    try:
        from app import (
            calculate_rolling_correlations,
            calculate_diversification_metrics,
            fetch_multiple_tickers
        )
        print("✅ Successfully imported correlation functions")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Create sample data for testing
    print("\n1️⃣ Creating sample correlation data...")
    
    dates = pd.date_range('2024-01-01', '2024-03-01', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Create correlated returns
    n_days = len(dates)
    base_returns = np.random.normal(0, 0.02, n_days)
    
    # AAPL and MSFT - moderately correlated (tech stocks)
    aapl_returns = base_returns + np.random.normal(0, 0.01, n_days)
    msft_returns = base_returns * 0.7 + np.random.normal(0, 0.015, n_days)
    
    # SPY - broader market, some correlation
    spy_returns = base_returns * 0.5 + np.random.normal(0, 0.012, n_days)
    
    # BTC - lower correlation with traditional assets
    btc_returns = base_returns * 0.2 + np.random.normal(0, 0.04, n_days)
    
    returns_data = pd.DataFrame({
        'AAPL': aapl_returns,
        'MSFT': msft_returns,
        'SPY': spy_returns,
        'BTC-USD': btc_returns
    }, index=dates)
    
    print(f"✅ Created sample data with {len(returns_data)} days")
    print(f"   Assets: {', '.join(returns_data.columns)}")
    
    # Test correlation matrix calculation
    print("\n2️⃣ Testing correlation matrix calculation...")
    
    corr_matrix = returns_data.corr()
    print("✅ Correlation matrix calculated:")
    print(corr_matrix.round(3))
    
    # Test diversification metrics
    print("\n3️⃣ Testing diversification metrics...")
    
    div_metrics = calculate_diversification_metrics(corr_matrix)
    
    if div_metrics:
        print("✅ Diversification metrics calculated:")
        print(f"   Average Correlation: {div_metrics['avg_correlation']:.3f}")
        print(f"   Max Correlation: {div_metrics['max_correlation']:.3f}")
        print(f"   Min Correlation: {div_metrics['min_correlation']:.3f}")
        print(f"   Diversification Score: {div_metrics['diversification_ratio']:.3f}")
        
        highly_corr = div_metrics.get('highly_correlated_pairs', pd.Series())
        if not highly_corr.empty:
            print(f"   Highly correlated pairs: {len(highly_corr)}")
            for pair, corr in highly_corr.head(3).items():
                print(f"     {pair[0]} - {pair[1]}: {corr:.3f}")
    else:
        print("❌ Failed to calculate diversification metrics")
    
    # Test rolling correlations
    print("\n4️⃣ Testing rolling correlations...")
    
    rolling_data = calculate_rolling_correlations(returns_data, window=30)
    
    if rolling_data and rolling_data.get('rolling_correlations'):
        rolling_corrs = rolling_data['rolling_correlations']
        print(f"✅ Rolling correlations calculated for {len(rolling_corrs)} pairs:")
        
        for pair, corr_series in list(rolling_corrs.items())[:3]:  # Show first 3
            if not corr_series.empty:
                print(f"   {pair}:")
                print(f"     Mean: {corr_series.mean():.3f}")
                print(f"     Std: {corr_series.std():.3f}")
                print(f"     Latest: {corr_series.iloc[-1]:.3f}")
    else:
        print("❌ Failed to calculate rolling correlations")

def test_with_real_data():
    """Test with real market data"""
    print("\n" + "="*60)
    print("📊 Testing with Real Market Data")
    print("="*60)
    
    try:
        from app import fetch_multiple_tickers, calculate_diversification_metrics
        
        # Fetch real data
        tickers = ["AAPL", "MSFT", "SPY", "BTC-USD"]
        start_date = "2024-01-01"
        end_date = "2024-02-01"
        
        print(f"Fetching data for: {', '.join(tickers)}")
        print(f"Period: {start_date} to {end_date}")
        
        price_data = fetch_multiple_tickers(tickers, start_date, end_date)
        
        if not price_data.empty:
            print(f"✅ Fetched {len(price_data)} data points")
            
            # Calculate returns and correlations
            returns = price_data.pct_change().dropna()
            corr_matrix = returns.corr()
            
            print("\n📊 Real Market Correlations:")
            print(corr_matrix.round(3))
            
            # Calculate diversification metrics
            div_metrics = calculate_diversification_metrics(corr_matrix)
            
            print(f"\n🎯 Diversification Analysis:")
            print(f"   Average Correlation: {div_metrics['avg_correlation']:.3f}")
            print(f"   Diversification Score: {div_metrics['diversification_ratio']:.3f}")
            
            # Interpretation
            avg_corr = div_metrics['avg_correlation']
            if avg_corr > 0.7:
                print("   📈 High correlation - Limited diversification benefits")
            elif avg_corr > 0.3:
                print("   ⚖️ Moderate correlation - Some diversification benefits")
            else:
                print("   ✅ Low correlation - Good diversification potential")
        
        else:
            print("❌ No real data available")
    
    except Exception as e:
        print(f"❌ Error with real data test: {e}")

def main():
    """Run all correlation tests"""
    print("🧪 Testing Correlation Analysis Functionality")
    print("📅 Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    try:
        test_correlation_functions()
        test_with_real_data()
        
        print("\n" + "="*70)
        print("✅ All correlation analysis tests completed!")
        print("\n💡 To see the full functionality:")
        print("   1. Open http://localhost:8501 in your browser")
        print("   2. Select multiple tickers (e.g., AAPL, MSFT, SPY, BTC-USD)")
        print("   3. Click on the 'Correlation' tab")
        print("   4. Click 'Calculate Correlations' button")
        print("   5. Explore heatmaps, rolling correlations, and network graphs!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
