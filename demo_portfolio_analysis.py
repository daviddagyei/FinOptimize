#!/usr/bin/env python3
"""
Demonstration script for comprehensive Return and Risk Metrics
Shows how to use the new functionality with multiple assets including crypto
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    fetch_multiple_tickers,
    calculate_comprehensive_metrics,
    fetch_risk_free_rate
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def demo_comprehensive_analysis():
    """Demonstrate comprehensive portfolio analysis with multiple assets"""
    print("🚀 Portfolio Analyzer - Return & Risk Metrics Demo")
    print("="*60)
    
    # Demo portfolio
    tickers = ["AAPL", "MSFT", "BTC-USD", "SPY"]
    start_date = "2024-01-01"
    end_date = "2024-03-01"
    
    print(f"📊 Analyzing portfolio: {', '.join(tickers)}")
    print(f"📅 Period: {start_date} to {end_date}")
    print("-" * 60)
    
    try:
        # Fetch portfolio data
        print("1️⃣ Fetching portfolio data...")
        portfolio_data = fetch_multiple_tickers(tickers, start_date, end_date)
        
        if portfolio_data.empty:
            print("❌ No data available for the selected tickers")
            return
        
        print(f"✅ Fetched data for {len(portfolio_data.columns)} assets")
        print(f"   Data points per asset: {len(portfolio_data)}")
        
        # Show latest prices
        print("\n📈 Latest Prices:")
        latest_prices = portfolio_data.iloc[-1]
        for ticker, price in latest_prices.items():
            print(f"   {ticker}: ${price:.2f}")
        
        # Calculate comprehensive metrics
        print("\n2️⃣ Calculating comprehensive metrics...")
        metrics_df = calculate_comprehensive_metrics(
            portfolio_data, start_date, end_date, "^TNX"
        )
        
        if metrics_df.empty:
            print("❌ Unable to calculate metrics")
            return
        
        print("✅ Metrics calculated successfully!")
        
        # Display key results
        print("\n🎯 KEY PERFORMANCE SUMMARY")
        print("-" * 60)
        
        for asset in metrics_df.index:
            print(f"\n📊 {asset}:")
            
            # Returns
            ann_return = metrics_df.loc[asset, 'Annualized Return']
            excess_return = metrics_df.loc[asset, 'Excess_Return_Ann']
            volatility = metrics_df.loc[asset, 'Annualized Volatility']
            
            print(f"   📈 Annualized Return: {ann_return:.2%}")
            print(f"   ⚡ Excess Return: {excess_return:.2%}")
            print(f"   📊 Volatility: {volatility:.2%}")
            
            # Risk-adjusted metrics
            sharpe = metrics_df.loc[asset, 'Annualized Sharpe Ratio']
            excess_sharpe = metrics_df.loc[asset, 'Excess_Sharpe_Ratio']
            
            print(f"   🏆 Sharpe Ratio: {sharpe:.3f}")
            print(f"   ⭐ Excess Sharpe: {excess_sharpe:.3f}")
            
            # Risk metrics
            max_dd = metrics_df.loc[asset, 'Max Drawdown']
            var_5 = metrics_df.loc[asset, 'VaR (0.05)']
            
            print(f"   📉 Max Drawdown: {max_dd:.2%}")
            print(f"   ⚠️  VaR (95%): {var_5:.2%}")
        
        # Portfolio comparison
        print("\n🏆 PORTFOLIO RANKING")
        print("-" * 60)
        
        # Rank by Sharpe ratio
        sharpe_ranking = metrics_df.sort_values('Annualized Sharpe Ratio', ascending=False)
        print("📊 By Sharpe Ratio (Best to Worst):")
        for i, (asset, row) in enumerate(sharpe_ranking.iterrows(), 1):
            sharpe = row['Annualized Sharpe Ratio']
            print(f"   {i}. {asset}: {sharpe:.3f}")
        
        # Rank by return
        return_ranking = metrics_df.sort_values('Annualized Return', ascending=False)
        print("\n📈 By Annualized Return (Highest to Lowest):")
        for i, (asset, row) in enumerate(return_ranking.iterrows(), 1):
            ret = row['Annualized Return']
            print(f"   {i}. {asset}: {ret:.2%}")
        
        # Risk assessment
        risk_ranking = metrics_df.sort_values('Max Drawdown', ascending=True)  # Lower drawdown is better
        print("\n🛡️ By Risk (Lowest to Highest Max Drawdown):")
        for i, (asset, row) in enumerate(risk_ranking.iterrows(), 1):
            dd = row['Max Drawdown']
            print(f"   {i}. {asset}: {dd:.2%}")
        
        # Risk-free rate info
        rf_rate = metrics_df.iloc[0]['Risk_Free_Rate_Ann']
        print(f"\n📊 Risk-Free Rate (10Y Treasury): {rf_rate:.2%}")
        
        print("\n✅ Analysis complete! Open the Streamlit app for interactive visualizations.")
        print("🌐 URL: http://localhost:8501")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_risk_free_rates():
    """Demo different risk-free rate options"""
    print("\n" + "="*60)
    print("📊 RISK-FREE RATE COMPARISON")
    print("="*60)
    
    start_date = "2024-01-01"
    end_date = "2024-02-01"
    
    rates = [
        ("^IRX", "3-Month Treasury"),
        ("^FVX", "5-Year Treasury"),
        ("^TNX", "10-Year Treasury"),
        ("^TYX", "30-Year Treasury")
    ]
    
    print(f"Period: {start_date} to {end_date}\n")
    
    for ticker, name in rates:
        try:
            rf_data = fetch_risk_free_rate(start_date, end_date, ticker)
            if not rf_data.empty:
                avg_daily = rf_data['Risk_Free_Rate'].mean()
                avg_annual = avg_daily * 252
                print(f"📊 {name} ({ticker}):")
                print(f"   Daily Rate: {avg_daily:.6f}")
                print(f"   Annualized: {avg_annual:.2%}\n")
            else:
                print(f"❌ {name}: No data available\n")
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}\n")

if __name__ == "__main__":
    demo_comprehensive_analysis()
    demo_risk_free_rates()
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED!")
    print("="*60)
    print("\n💡 Next Steps:")
    print("1. Open http://localhost:8501 in your browser")
    print("2. Try the 'Return Metrics' tab with your own tickers")
    print("3. Explore the 'Risk Analysis' tab for detailed risk metrics")
    print("4. Download CSV reports for further analysis")
    print("\n🚀 Happy analyzing!")
