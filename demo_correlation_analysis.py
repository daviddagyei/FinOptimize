#!/usr/bin/env python3
"""
Comprehensive demo of the new Correlation Analysis functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import fetch_multiple_tickers, calculate_diversification_metrics
import pandas as pd
import numpy as np
from datetime import datetime

def demo_correlation_analysis():
    """Demonstrate the comprehensive correlation analysis features"""
    print("🔗 Portfolio Analyzer - Correlation Analysis Demo")
    print("="*70)
    
    # Demo portfolios with different correlation characteristics
    portfolios = {
        "Tech Portfolio": ["AAPL", "MSFT", "GOOGL", "META"],
        "Diversified Portfolio": ["AAPL", "SPY", "BTC-USD", "GLD"],
        "Mixed Assets": ["MSFT", "JPM", "BTC-USD", "QQQ", "GLD"]
    }
    
    start_date = "2024-01-01"
    end_date = "2024-03-01"
    
    print(f"📅 Analysis Period: {start_date} to {end_date}")
    print("-" * 70)
    
    for portfolio_name, tickers in portfolios.items():
        print(f"\n📊 {portfolio_name}")
        print(f"Assets: {', '.join(tickers)}")
        print("-" * 50)
        
        try:
            # Fetch data
            price_data = fetch_multiple_tickers(tickers, start_date, end_date)
            
            if price_data.empty:
                print("❌ No data available")
                continue
            
            # Calculate returns and correlations
            returns = price_data.pct_change().dropna()
            corr_matrix = returns.corr()
            
            # Calculate diversification metrics
            div_metrics = calculate_diversification_metrics(corr_matrix)
            
            # Display results
            print(f"✅ Data points: {len(returns)}")
            print(f"📈 Average Correlation: {div_metrics['avg_correlation']:.3f}")
            print(f"🎯 Diversification Score: {div_metrics['diversification_ratio']:.3f}")
            print(f"📊 Correlation Range: {div_metrics['min_correlation']:.3f} to {div_metrics['max_correlation']:.3f}")
            
            # Interpretation
            avg_corr = div_metrics['avg_correlation']
            div_score = div_metrics['diversification_ratio']
            
            if avg_corr > 0.7:
                correlation_assessment = "🔴 High correlation - Limited diversification"
            elif avg_corr > 0.3:
                correlation_assessment = "🟡 Moderate correlation - Some diversification"
            else:
                correlation_assessment = "🟢 Low correlation - Good diversification"
            
            if div_score > 0.7:
                diversification_assessment = "🟢 Excellent diversification"
            elif div_score > 0.4:
                diversification_assessment = "🟡 Moderate diversification"
            else:
                diversification_assessment = "🔴 Poor diversification"
            
            print(f"   {correlation_assessment}")
            print(f"   {diversification_assessment}")
            
            # Show strongest correlations
            highly_corr = div_metrics.get('highly_correlated_pairs', pd.Series())
            if not highly_corr.empty:
                print(f"⚠️  High correlations (>0.7):")
                for pair, corr in highly_corr.head(3).items():
                    print(f"   • {pair[0]} ↔ {pair[1]}: {corr:.3f}")
            
            # Show negative correlations (good for diversification)
            neg_corr = div_metrics.get('negatively_correlated_pairs', pd.Series())
            if not neg_corr.empty:
                print(f"✅ Negative correlations (<-0.3):")
                for pair, corr in neg_corr.head(3).items():
                    print(f"   • {pair[0]} ↔ {pair[1]}: {corr:.3f}")
            
        except Exception as e:
            print(f"❌ Error analyzing {portfolio_name}: {str(e)}")
    
    # Correlation insights and recommendations
    print("\n" + "="*70)
    print("📋 CORRELATION ANALYSIS INSIGHTS")
    print("="*70)
    
    print("""
🔍 How to Interpret Correlation Results:

📊 Correlation Ranges:
   • 0.7 to 1.0: High positive correlation (move together)
   • 0.3 to 0.7: Moderate positive correlation  
   • -0.3 to 0.3: Low correlation (more independent)
   • -0.7 to -0.3: Moderate negative correlation
   • -1.0 to -0.7: High negative correlation (move opposite)

🎯 Diversification Benefits:
   • High correlations reduce diversification benefits
   • Negative correlations provide excellent diversification
   • Mix of asset classes typically improves diversification

🚀 Portfolio Construction Tips:
   • Tech stocks often highly correlated (AAPL, MSFT, GOOGL)
   • Traditional assets vs. crypto often lower correlation
   • International exposure can reduce correlation
   • Different sectors provide diversification benefits
""")

def demo_rolling_correlations():
    """Demonstrate rolling correlation analysis"""
    print("\n" + "="*70)
    print("📈 ROLLING CORRELATION INSIGHTS")
    print("="*70)
    
    print("""
🔄 What Rolling Correlations Show:

⏰ Time-Varying Relationships:
   • Correlations change over time due to market conditions
   • Crisis periods often see correlations increase
   • Normal periods may show lower correlations

📊 Key Patterns to Watch:
   • Increasing correlations = Reduced diversification benefits
   • Decreasing correlations = Improved diversification
   • Volatile correlations = Unstable relationships

🎯 Investment Implications:
   • Monitor correlation stability for risk management
   • Rebalance when correlations become too high
   • Consider alternative assets during high correlation periods
""")

def main():
    """Run the complete correlation analysis demo"""
    print("🧪 Portfolio Analyzer - Comprehensive Correlation Demo")
    print("📅 Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    try:
        demo_correlation_analysis()
        demo_rolling_correlations()
        
        print("\n" + "="*70)
        print("🎉 CORRELATION ANALYSIS DEMO COMPLETED!")
        print("="*70)
        
        print("\n🚀 Ready to explore the interactive features:")
        print("   1. Open http://localhost:8501 in your browser")
        print("   2. Select multiple tickers (mix different asset classes)")
        print("   3. Navigate to the 'Correlation' tab")
        print("   4. Click 'Calculate Correlations'")
        print("   5. Explore these powerful features:")
        print("      • 🌡️ Interactive correlation heatmaps")
        print("      • 📈 Rolling correlation charts over time")
        print("      • 🕸️ Network visualization of relationships")
        print("      • 📊 Diversification metrics and insights")
        print("      • 📥 Downloadable correlation reports")
        
        print("\n💡 Pro Tips:")
        print("   • Try different correlation methods (Pearson, Spearman, Kendall)")
        print("   • Adjust rolling window size to see different time scales")
        print("   • Mix asset classes for better diversification insights")
        print("   • Use correlation data for portfolio rebalancing decisions")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
