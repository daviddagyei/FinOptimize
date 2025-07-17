#!/usr/bin/env python3
"""
Real data verification script for portfolio analyzer
Tests actual yfinance data fetching without mocking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_direct_yfinance():
    """Test direct yfinance calls"""
    print("=== Testing Direct yfinance Data Fetching ===\n")
    
    # Test parameters
    ticker = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Testing ticker: {ticker}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # Direct yfinance call
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'))
        
        print(f"✅ Successfully fetched data")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Index type: {type(data.index)}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        
        if not data.empty:
            print(f"   Latest close price: ${data['Close'].iloc[-1]:.2f}")
            print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        print("\nFirst 3 rows:")
        print(data.head(3))
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        return pd.DataFrame()

def test_app_functions():
    """Test the actual app functions with real data"""
    print("\n=== Testing App Functions with Real Data ===\n")
    
    try:
        # Import app functions
        from app import fetch_stock_data, fetch_multiple_tickers, get_ticker_info
        
        # Test single ticker
        print("Testing fetch_stock_data...")
        result = fetch_stock_data('AAPL', '2025-07-08', '2025-07-15', 'Close')
        
        if not result.empty:
            print(f"✅ fetch_stock_data successful")
            print(f"   Shape: {result.shape}")
            print(f"   Columns: {list(result.columns)}")
            print(f"   Sample data:\n{result.head()}")
        else:
            print("❌ fetch_stock_data returned empty DataFrame")
        
        # Test multiple tickers
        print("\nTesting fetch_multiple_tickers...")
        multi_result = fetch_multiple_tickers(['AAPL', 'MSFT'], '2025-07-08', '2025-07-15')
        
        if not multi_result.empty:
            print(f"✅ fetch_multiple_tickers successful")
            print(f"   Shape: {multi_result.shape}")
            print(f"   Columns: {list(multi_result.columns)}")
            print(f"   Sample data:\n{multi_result.head()}")
        else:
            print("❌ fetch_multiple_tickers returned empty DataFrame")
        
        # Test ticker info
        print("\nTesting get_ticker_info...")
        info = get_ticker_info('AAPL')
        print(f"✅ get_ticker_info result: {info}")
        
    except ImportError as e:
        print(f"❌ Cannot import app functions: {str(e)}")
    except Exception as e:
        print(f"❌ Error testing app functions: {str(e)}")

def test_streamlit_session_state():
    """Test what might be happening with Streamlit session state"""
    print("\n=== Testing Streamlit Components ===\n")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        # Test cache decorator
        print("Testing @st.cache_data decorator...")
        
        @st.cache_data(ttl=300)
        def test_cached_function(ticker):
            stock = yf.Ticker(ticker)
            return stock.history(period="5d")
        
        # This should work even outside Streamlit context for testing
        print("Cache decorator test: Attempting to call cached function...")
        
    except Exception as e:
        print(f"❌ Streamlit components error: {str(e)}")

def test_popular_tickers():
    """Test a few popular tickers to ensure they work"""
    print("\n=== Testing Popular Tickers ===\n")
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="5d")
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                print(f"✅ {ticker}: ${latest_price:.2f} ({len(data)} days)")
            else:
                print(f"❌ {ticker}: No data returned")
                
        except Exception as e:
            print(f"❌ {ticker}: Error - {str(e)}")

def debug_streamlit_app():
    """Debug potential issues in the Streamlit app"""
    print("\n=== Debugging Streamlit App Issues ===\n")
    
    # Check if there are any obvious issues in the app logic
    try:
        with open('app.py', 'r') as f:
            app_content = f.read()
        
        # Check for common issues
        issues = []
        
        if 'st.error' in app_content and 'st.success' not in app_content:
            issues.append("App may only show errors, not success messages")
        
        if 'if data.empty:' in app_content:
            issues.append("App checks for empty data - this might be triggering")
        
        if 'return pd.DataFrame()' in app_content:
            issues.append("Functions return empty DataFrame on errors")
        
        if issues:
            print("Potential issues found:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("No obvious issues found in app.py")
            
        # Check if the main function has proper error handling
        if 'if not tickers:' in app_content:
            print("✅ App has ticker validation")
        
        if 'with st.spinner' in app_content:
            print("✅ App shows loading spinner")
            
    except Exception as e:
        print(f"❌ Error reading app.py: {str(e)}")

if __name__ == "__main__":
    print("Portfolio Analyzer - Real Data Verification")
    print("=" * 50)
    
    # Run all tests
    test_direct_yfinance()
    test_app_functions() 
    test_streamlit_session_state()
    test_popular_tickers()
    debug_streamlit_app()
    
    print("\n" + "=" * 50)
    print("Verification complete!")
    print("\nIf yfinance works but Streamlit doesn't show data:")
    print("1. Check browser console for errors")
    print("2. Verify ticker input is being processed")
    print("3. Check if data is empty due to date ranges")
    print("4. Look for error messages in Streamlit interface")
