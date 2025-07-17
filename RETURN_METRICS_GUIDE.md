# Portfolio Analyzer - Return and Risk Metrics Implementation

## 🎯 Overview

This implementation provides comprehensive return and risk metrics analysis for portfolio management, including:

- **Excess Returns Calculation**: ri - rf (asset return minus risk-free rate)
- **Risk-Free Rate Integration**: Using Treasury bill data from yfinance
- **Comprehensive Performance Metrics**: All 5 principal risk measures according to Investopedia
- **Interactive Visualizations**: Risk-return profiles, drawdown analysis, and distribution charts

## 📊 Features Implemented

### Return Metrics Tab

#### Core Functionality:
1. **Risk-Free Rate Selection**:
   - 3-Month Treasury (^IRX)
   - 5-Year Treasury (^FVX)
   - 10-Year Treasury (^TNX) - Default
   - 30-Year Treasury (^TYX)

2. **Excess Returns Calculation**:
   - Formula: `ri - rf` where:
     - `ri` = return rate of asset i
     - `rf` = risk-free return rate
     - Result = excess return rate of asset i

3. **Performance Metrics**:
   - Annualized Return
   - Annualized Volatility
   - Sharpe Ratio (both standard and excess return based)
   - Sortino Ratio
   - Calmar Ratio

#### Visualizations:
- Return distribution histograms
- Box plots for return analysis
- Risk-return scatter plots
- Key performance indicators dashboard

### Risk Analysis Tab

#### Core Functionality:
1. **Value at Risk (VaR)**:
   - Configurable confidence levels (99%, 95%, 90%)
   - Historical simulation method
   - Visual representation with distribution charts

2. **Conditional Value at Risk (CVaR)**:
   - Expected loss beyond VaR threshold
   - Tail risk assessment

3. **Drawdown Analysis**:
   - Maximum drawdown calculation
   - Recovery period analysis
   - Drawdown duration tracking
   - Visual drawdown charts

4. **Distribution Statistics**:
   - Skewness (asymmetry measure)
   - Excess Kurtosis (tail thickness)
   - Min/Max returns

#### Advanced Features:
- Rolling 30-day volatility charts
- Risk profile comparison radar charts
- Comprehensive drawdown visualization
- Risk distribution analysis

## 🔧 Technical Implementation

### Key Functions:

1. **`fetch_risk_free_rate(start_date, end_date, rate_type)`**:
   - Fetches Treasury yield data from yfinance
   - Converts annual yields to daily rates
   - Provides fallback to 2% constant rate

2. **`calculate_excess_returns(returns_df, rf_rate_df)`**:
   - Aligns dates between returns and risk-free rate
   - Subtracts risk-free rate from asset returns
   - Returns excess returns DataFrame

3. **`calculate_comprehensive_metrics(data, start_date, end_date, rf_rate_type)`**:
   - Orchestrates complete metrics calculation
   - Combines return and risk metrics
   - Integrates excess returns analysis

### Utils Integration:

The implementation leverages existing utility functions from `utils.py`:
- `calc_performance_metrics()`: Comprehensive performance analysis
- `calc_return_metrics()`: Return-based calculations
- `calc_risk_metrics()`: Risk-based calculations

## 📈 Usage Instructions

### Step 1: Load Data
1. Select tickers (stocks, ETFs, crypto with -USD suffix)
2. Choose date range
3. Load historical price data

### Step 2: Return Metrics Analysis
1. Navigate to "Return Metrics" tab
2. Select preferred risk-free rate proxy
3. Click "Calculate Metrics" button
4. Review comprehensive metrics table
5. Analyze visualizations (distribution, risk-return plots)
6. Download CSV report

### Step 3: Risk Analysis
1. Navigate to "Risk Analysis" tab
2. Select VaR confidence level
3. Click "Calculate Risk Metrics" button
4. Review risk indicators and detailed metrics
5. Analyze drawdown charts and risk distributions
6. Download risk analysis report

## 📊 Output Metrics

### Return Metrics:
- **Annualized Return**: `mean(returns) * 252`
- **Excess Return**: `annualized_return - risk_free_rate`
- **Annualized Volatility**: `std(returns) * sqrt(252)`
- **Sharpe Ratio**: `annualized_return / annualized_volatility`
- **Excess Sharpe Ratio**: `excess_return / annualized_volatility`
- **Sortino Ratio**: `annualized_return / downside_deviation`

### Risk Metrics:
- **VaR (α%)**: `quantile(returns, α)`
- **CVaR (α%)**: `mean(returns[returns <= VaR])`
- **Maximum Drawdown**: `min((wealth_index - peak) / peak)`
- **Skewness**: Distribution asymmetry measure
- **Excess Kurtosis**: Tail thickness measure

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_return_metrics.py
```

Tests cover:
- Risk-free rate data fetching
- Excess returns calculation accuracy
- Comprehensive metrics with real market data
- Error handling and edge cases

## 💡 Best Practices

1. **Risk-Free Rate Selection**:
   - Use 3-month Treasury for short-term analysis
   - Use 10-year Treasury for long-term portfolio analysis
   - Consider 5-year for medium-term strategies

2. **Data Quality**:
   - Ensure sufficient historical data (minimum 1 month)
   - Check for missing data or trading halts
   - Verify ticker formats (especially for crypto: BTC-USD, not BTC)

3. **Interpretation Guidelines**:
   - Sharpe Ratio > 1.0 indicates good risk-adjusted returns
   - Negative skewness indicates left-tail risk
   - High excess kurtosis suggests extreme events
   - Maximum drawdown shows worst-case scenario

## 📋 Error Handling

The implementation includes robust error handling:
- Fallback to constant risk-free rate if Treasury data unavailable
- Warning messages for data quality issues
- Graceful handling of missing or invalid tickers
- Input validation for date ranges and parameters

## 🚀 Future Enhancements

Potential improvements:
1. **CAPM Analysis**: Beta calculation against market benchmarks
2. **Correlation Analysis**: Multi-asset correlation matrices
3. **Portfolio Optimization**: Markowitz efficient frontier
4. **Regime Analysis**: Bull/bear market performance
5. **Factor Models**: Fama-French factor exposures

## 📚 References

- **Investopedia**: "The five principal risk measures include alpha, beta, R-squared, standard deviation, and the Sharpe ratio"
- **Modern Portfolio Theory**: Risk-return optimization principles
- **Yahoo Finance API**: Real-time market data source
- **Federal Reserve**: Treasury yield curve data
