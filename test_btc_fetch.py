#!/usr/bin/env python3
"""
Test script to fetch Bitcoin (BTC) data using our portfolio analyzer implementation
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add the current directory to path to import app functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_btc_data_fetch():
    """Test fetching Bitcoin data using our app functions"""
    print("=" * 60)
    print("🪙 BITCOIN (BTC) DATA FETCH TEST")
    print("=" * 60)
    
    try:
        # Import our app functions
        from app import fetch_stock_data, get_ticker_info
        
        # Bitcoin ticker symbol
        btc_ticker = "BTC-USD"  # Yahoo Finance format for Bitcoin
        
        # Date range - last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"📅 Date Range: {start_str} to {end_str}")
        print(f"🎯 Ticker: {btc_ticker}")
        print("\n🔄 Fetching Bitcoin data...")
        
        # Test ticker info first
        print("\n1️⃣ Getting Bitcoin info...")
        btc_info = get_ticker_info(btc_ticker)
        print(f"   Name: {btc_info['name']}")
        print(f"   Sector: {btc_info['sector']}")
        print(f"   Currency: {btc_info['currency']}")
        
        # Test Close price data
        print("\n2️⃣ Fetching Close price data...")
        close_data = fetch_stock_data(btc_ticker, start_str, end_str, "Close")
        
        if not close_data.empty:
            print(f"   ✅ Success! Got {len(close_data)} data points")
            print(f"   📊 Latest price: ${close_data.iloc[-1, 0]:,.2f}")
            print(f"   📈 Highest price: ${close_data.max().iloc[0]:,.2f}")
            print(f"   📉 Lowest price: ${close_data.min().iloc[0]:,.2f}")
            
            # Calculate some basic stats
            price_change = close_data.iloc[-1, 0] - close_data.iloc[0, 0]
            price_change_pct = (price_change / close_data.iloc[0, 0]) * 100
            
            print(f"   📊 30-day change: ${price_change:,.2f} ({price_change_pct:+.2f}%)")
            
            # Show recent data
            print("\n📋 Recent BTC prices:")
            recent_data = close_data.tail(5).copy()
            recent_data.columns = ['BTC_Price_USD']
            recent_data['BTC_Price_USD'] = recent_data['BTC_Price_USD'].apply(lambda x: f"${x:,.2f}")
            print(recent_data.to_string())
            
        else:
            print("   ❌ No data received")
        
        # Test OHLC data
        print("\n3️⃣ Fetching OHLC data...")
        ohlc_data = fetch_stock_data(btc_ticker, start_str, end_str, "OHLC")
        
        if not ohlc_data.empty:
            print(f"   ✅ Success! Got OHLC data with shape: {ohlc_data.shape}")
            print("   📊 Recent OHLC data:")
            recent_ohlc = ohlc_data.tail(3).copy()
            
            # Format for better display
            for col in ['Open', 'High', 'Low', 'Close']:
                recent_ohlc[col] = recent_ohlc[col].apply(lambda x: f"${x:,.2f}")
            
            print(recent_ohlc[['Open', 'High', 'Low', 'Close']].to_string())
        else:
            print("   ❌ No OHLC data received")
        
        print("\n" + "=" * 60)
        print("✅ Bitcoin data fetch test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during Bitcoin data fetch: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def test_other_crypto():
    """Test other cryptocurrencies"""
    print("\n" + "=" * 60)
    print("🚀 OTHER CRYPTOCURRENCY TEST")
    print("=" * 60)
    
    crypto_tickers = [
        ("ETH-USD", "Ethereum"),
        ("ADA-USD", "Cardano"),
        ("DOT-USD", "Polkadot")
    ]
    
    try:
        from app import fetch_stock_data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        for ticker, name in crypto_tickers:
            print(f"\n🪙 Testing {name} ({ticker})...")
            
            data = fetch_stock_data(ticker, start_str, end_str, "Close")
            
            if not data.empty:
                latest_price = data.iloc[-1, 0]
                print(f"   ✅ {name}: ${latest_price:,.2f}")
            else:
                print(f"   ❌ {name}: No data")
                
    except Exception as e:
        print(f"❌ Error testing other cryptocurrencies: {str(e)}")

if __name__ == "__main__":
    # Test Bitcoin data fetching
    test_btc_data_fetch()
    
    # Test other cryptocurrencies
    test_other_crypto()
    
    print("\n🎉 All tests completed!")
