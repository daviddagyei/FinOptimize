import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to path to import app functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit components before importing app
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit components for all tests"""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.cache_data') as mock_cache, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.subheader'), \
         patch('streamlit.dataframe'):
        
        # Configure cache_data to return the original function
        mock_cache.side_effect = lambda ttl=None: lambda func: func
        yield {
            'cache_data': mock_cache,
            'error': mock_error,
            'sidebar': mock_sidebar
        }

# Import app functions after mocking streamlit
from app import (
    fetch_stock_data,
    fetch_multiple_tickers,
    get_ticker_info,
    create_ticker_input,
    create_date_range_input,
    display_ticker_info
)


class TestFetchStockData:
    """Test suite for fetch_stock_data function"""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        return pd.DataFrame({
            'Open': [150, 152, 148, 155, 160, 158, 162, 165, 163, 168],
            'High': [155, 157, 153, 160, 165, 163, 167, 170, 168, 173],
            'Low': [148, 150, 145, 152, 157, 155, 159, 162, 160, 165],
            'Close': [153, 155, 151, 158, 163, 161, 165, 168, 166, 171],
            'Volume': [1000000, 1200000, 900000, 1500000, 1800000, 1100000, 1600000, 1400000, 1300000, 1700000]
        }, index=dates)
    
    @patch('app.yf.Ticker')
    @patch('app.st.error')
    def test_fetch_stock_data_close_success(self, mock_error, mock_ticker, sample_stock_data):
        """Test successful fetching of close price data"""
        mock_stock = MagicMock()
        mock_stock.history.return_value = sample_stock_data
        mock_ticker.return_value = mock_stock
        
        result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-10", "Close")
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert 'AAPL' in result.columns
        assert len(result) == 10
        assert not result.empty
        mock_ticker.assert_called_once_with("AAPL")
        mock_error.assert_not_called()
    
    @patch('app.yf.Ticker')
    @patch('app.st.error')
    def test_fetch_stock_data_ohlc_success(self, mock_error, mock_ticker, sample_stock_data):
        """Test successful fetching of OHLC data"""
        mock_stock = MagicMock()
        mock_stock.history.return_value = sample_stock_data
        mock_ticker.return_value = mock_stock
        
        result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-10", "OHLC")
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['Open', 'High', 'Low', 'Close']
        assert all(col in result.columns for col in expected_columns)
        assert len(result) == 10
        mock_error.assert_not_called()
    
    @patch('app.yf.Ticker')
    @patch('app.st.error')
    def test_fetch_stock_data_empty_data(self, mock_error, mock_ticker):
        """Test handling of empty data response"""
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame()  # Empty dataframe
        mock_ticker.return_value = mock_stock
        
        result = fetch_stock_data("INVALID", "2024-01-01", "2024-01-10", "Close")
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        mock_error.assert_called_once()
    
    @patch('app.yf.Ticker')
    @patch('app.st.error')
    def test_fetch_stock_data_exception(self, mock_error, mock_ticker):
        """Test handling of API exceptions"""
        mock_ticker.side_effect = Exception("API Error")
        
        result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-10", "Close")
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        mock_error.assert_called_once()


class TestFetchMultipleTickers:
    """Test suite for fetch_multiple_tickers function"""
    
    @pytest.fixture
    def sample_multi_ticker_data(self):
        """Create sample multi-ticker data"""
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        tickers = ['AAPL', 'MSFT']
        columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume']])
        
        data = pd.DataFrame(
            np.random.rand(5, 10) * 100 + 100,
            index=dates,
            columns=columns
        )
        return data
    
    @patch('app.yf.download')
    @patch('app.st.error')
    def test_fetch_multiple_tickers_success(self, mock_error, mock_download, sample_multi_ticker_data):
        """Test successful fetching of multiple tickers"""
        mock_download.return_value = sample_multi_ticker_data
        
        result = fetch_multiple_tickers(['AAPL', 'MSFT'], '2024-01-01', '2024-01-05')
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_download.assert_called_once_with(['AAPL', 'MSFT'], start='2024-01-01', end='2024-01-05', group_by='ticker')
        mock_error.assert_not_called()
    
    @patch('app.yf.download')
    @patch('app.st.error')
    def test_fetch_multiple_tickers_exception(self, mock_error, mock_download):
        """Test exception handling in multiple ticker fetch"""
        mock_download.side_effect = Exception("Download failed")
        
        result = fetch_multiple_tickers(['AAPL', 'MSFT'], '2024-01-01', '2024-01-05')
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        mock_error.assert_called_once()


class TestGetTickerInfo:
    """Test suite for get_ticker_info function"""
    
    @patch('app.yf.Ticker')
    def test_get_ticker_info_success(self, mock_ticker):
        """Test successful ticker info retrieval"""
        mock_stock = MagicMock()
        mock_stock.info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'currency': 'USD'
        }
        mock_ticker.return_value = mock_stock
        
        result = get_ticker_info('AAPL')
        
        # Assertions
        assert result['name'] == 'Apple Inc.'
        assert result['sector'] == 'Technology'
        assert result['industry'] == 'Consumer Electronics'
        assert result['market_cap'] == 3000000000000
        assert result['currency'] == 'USD'
    
    @patch('app.yf.Ticker')
    def test_get_ticker_info_exception(self, mock_ticker):
        """Test ticker info exception handling"""
        mock_ticker.side_effect = Exception("API Error")
        
        result = get_ticker_info('INVALID')
        
        # Assertions - should return default values
        assert result['name'] == 'INVALID'
        assert result['sector'] == 'N/A'
        assert result['industry'] == 'N/A'
        assert result['market_cap'] == 'N/A'
        assert result['currency'] == 'USD'


class TestCreateTickerInput:
    """Test suite for create_ticker_input function"""
    
    @patch('app.st.sidebar')
    @patch('app.st.session_state', {})
    def test_create_ticker_input_single_ticker(self, mock_sidebar):
        """Test single ticker input"""
        mock_sidebar.header = MagicMock()
        mock_sidebar.radio.return_value = "Single Ticker"
        mock_sidebar.text_input.return_value = "AAPL"
        mock_sidebar.write = MagicMock()
        mock_sidebar.columns.return_value = [MagicMock() for _ in range(3)]
        
        result = create_ticker_input()
        
        assert result == ['AAPL']
    
    @patch('app.st.sidebar')
    def test_create_ticker_input_multiple_tickers(self, mock_sidebar):
        """Test multiple ticker input"""
        mock_sidebar.header = MagicMock()
        mock_sidebar.radio.return_value = "Multiple Tickers"
        mock_sidebar.text_area.return_value = "AAPL\nMSFT\nGOOGL"
        
        result = create_ticker_input()
        
        assert result == ['AAPL', 'MSFT', 'GOOGL']


class TestCreateDateRangeInput:
    """Test suite for create_date_range_input function"""
    
    @patch('app.st.sidebar')
    def test_create_date_range_1_year_preset(self, mock_sidebar):
        """Test 1 year preset date range"""
        mock_sidebar.header = MagicMock()
        mock_sidebar.selectbox.return_value = "1 Year"
        
        start_date, end_date = create_date_range_input()
        
        # Parse dates and verify they're approximately 1 year apart
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        diff = end_dt - start_dt
        
        assert 360 <= diff.days <= 370  # Allow some variation for leap years


class TestDisplayTickerInfo:
    """Test suite for display_ticker_info function"""
    
    def test_display_ticker_info_empty_list(self):
        """Test display with empty ticker list - should return early"""
        result = display_ticker_info([])
        assert result is None  # Function returns None for empty list


class TestDataValidation:
    """Test suite for data validation and edge cases"""
    
    def test_date_format_validation(self):
        """Test date format consistency"""
        start_date = '2024-01-01'
        end_date = '2024-01-10'
        
        # Should not raise exceptions
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        assert start_dt < end_dt
        assert isinstance(start_dt, datetime)
        assert isinstance(end_dt, datetime)
    
    def test_ticker_symbol_validation(self):
        """Test ticker symbol format validation"""
        valid_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        for ticker in valid_tickers:
            assert isinstance(ticker, str)
            assert len(ticker) > 0
            assert ticker.isupper()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
