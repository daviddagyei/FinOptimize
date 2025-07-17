# Portfolio Analyzer

A comprehensive Streamlit-based portfolio analysis tool that provides real-time financial data fetching, advanced analytics, and interactive visualizations for investment portfolio management.

## Features

### 🚀 Core Functionality
- **Dynamic Ticker Input**: Multiple input methods including single ticker, multiple tickers, and CSV upload
- **Real-time Data Fetching**: Cached data retrieval using Yahoo Finance API
- **Interactive Visualizations**: Price charts, candlestick charts, and performance metrics
- **Comprehensive Analytics**: Risk metrics, return analysis, and portfolio optimization

### 📊 Analysis Capabilities
- **Return Metrics**: Annualized returns, volatility, Sharpe ratio, Sortino ratio
- **Risk Analysis**: VaR, CVaR, skewness, kurtosis, maximum drawdown
- **Regression Analysis**: CAPM, univariate/multivariate regressions
- **Portfolio Optimization**: Mean-variance frontier, tangency portfolio, GMV portfolio
- **Correlation Analysis**: Correlation matrices and relationship mapping

### 🎯 Key Features
- **Cached Data Loading**: `@st.cache_data` decorator for efficient data management
- **Multiple Input Methods**: Text input, batch upload, and quick-select options
- **Date Range Flexibility**: Preset ranges or custom date selection
- **Interactive Charts**: Plotly-powered visualizations with zoom and hover features
- **Responsive Design**: Wide layout optimized for analysis workflows

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd portfolio-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

## Usage

### Getting Started
1. **Open the application** in your browser (typically `http://localhost:8501`)
2. **Select ticker input method** from the sidebar:
   - Single Ticker: Enter one stock symbol
   - Multiple Tickers: Enter multiple symbols (one per line)
   - Upload CSV: Upload a file with ticker symbols
3. **Choose date range** using preset options or custom dates
4. **Select data type** (Close, Adj Close, OHLC, All)

### Input Methods

#### Single Ticker
```python
# Example: Enter "AAPL" for Apple Inc.
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value="AAPL")
```

#### Multiple Tickers
```
AAPL
MSFT
GOOGL
TSLA
```

#### CSV Upload
CSV file should contain a column named 'ticker' or 'symbol':
```csv
ticker
AAPL
MSFT
GOOGL
```

### Data Fetching Functions

#### Single Ticker Data
```python
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, start_date: str, end_date: str, data_type: str = "Close"):
    """
    Fetch historical stock data with caching
    - ticker: Stock symbol (e.g., 'AAPL')
    - start_date: Start date in 'YYYY-MM-DD' format
    - end_date: End date in 'YYYY-MM-DD' format
    - data_type: 'Close', 'Adj Close', 'OHLC', or 'All'
    """
```

#### Multiple Tickers Data
```python
@st.cache_data(ttl=300)
def fetch_multiple_tickers(tickers: List[str], start_date: str, end_date: str):
    """
    Fetch data for multiple tickers efficiently
    - tickers: List of ticker symbols
    - Returns: DataFrame with closing prices for all tickers
    """
```

## Architecture

### File Structure
```
portfolio-analyzer/
├── app.py              # Main Streamlit application
├── utils.py            # Financial analysis utility functions
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Components

#### Data Management
- **Caching Strategy**: 5-minute TTL for price data, 1-hour for ticker info
- **Error Handling**: Comprehensive error handling for invalid tickers/dates
- **Data Validation**: Automatic data cleaning and validation

#### User Interface
- **Sidebar Controls**: Organized input controls for easy navigation
- **Responsive Layout**: Wide layout with expandable sections
- **Interactive Charts**: Plotly integration for rich visualizations

#### Analysis Engine
- **Utility Functions**: Comprehensive set of financial analysis functions
- **Performance Metrics**: Risk-adjusted returns and portfolio analytics
- **Optimization Tools**: Mean-variance optimization and portfolio construction

## Advanced Features

### Caching Implementation
```python
@st.cache_data(ttl=300)  # 5-minute cache
def fetch_stock_data(ticker, start_date, end_date, data_type):
    # Efficient data fetching with automatic cache invalidation
```

### Dynamic Reloading
The application automatically reloads data when:
- Ticker selection changes
- Date range is modified
- Data type selection is updated

### Session State Management
```python
# Store data for cross-component access
st.session_state['data'] = data
st.session_state['tickers'] = tickers
st.session_state['start_date'] = start_date
st.session_state['end_date'] = end_date
```

## Extending the Application

### Adding New Analysis Modules
1. Import utility functions: `from utils import calc_performance_metrics`
2. Add new analysis sections to the main app
3. Create new visualization functions using Plotly
4. Implement caching for computationally expensive operations

### Custom Indicators
```python
# Example: Adding custom technical indicators
def calculate_moving_averages(data, windows=[20, 50, 200]):
    for window in windows:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    return data
```

## Dependencies

- **streamlit**: Web application framework
- **yfinance**: Yahoo Finance API for market data
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **statsmodels**: Statistical modeling
- **seaborn & matplotlib**: Additional plotting capabilities

## Performance Considerations

- **Caching**: Intelligent caching reduces API calls and improves response times
- **Batch Processing**: Multiple ticker data fetched efficiently in single API call
- **Memory Management**: Session state used judiciously to avoid memory bloat
- **Error Recovery**: Graceful degradation when data is unavailable

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This application is for educational and research purposes. Always verify financial calculations and consult with financial professionals for investment decisions.
