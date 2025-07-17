#!/usr/bin/env python3
"""
Simple test to verify Bitcoin data fetching works in the Streamlit app
"""

import yfinance as yf
from datetime import datetime, timedelta

def test_bitcoin_ticker():
    """Test Bitcoin ticker formats"""
    print("🪙 Testing Bitcoin ticker formats...")
    
    # Test different Bitcoin ticker formats
    tickers_to_test = ['BTC', 'BTC-USD', 'BTCUSD']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    for ticker in tickers_to_test:
        print(f"\n🔍 Testing ticker: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                print(f"   ✅ SUCCESS: Latest price = ${latest_price:,.2f}")
                print(f"   📊 Data points: {len(data)}")
            else:
                print(f"   ❌ FAILED: No data returned")
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    print("- Use 'BTC-USD' for Bitcoin in Yahoo Finance")
    print("- 'BTC' alone typically doesn't work")
    print("- Always use the '-USD' suffix for crypto")
    print("="*50)

if __name__ == "__main__":
    test_bitcoin_ticker()
