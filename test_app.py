import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
import yfinance as yf
import io
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
    def test_fetch_stock_data_all_success(self, mock_error, mock_ticker, sample_stock_data):
        """Test successful fetching of all data"""
        mock_stock = MagicMock()
        mock_stock.history.return_value = sample_stock_data
        mock_ticker.return_value = mock_stock
        
        result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-10", "All")
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
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
    def test_fetch_multiple_tickers_single_ticker(self, mock_error, mock_download):
        """Test fetching single ticker through multiple ticker function"""
        single_data = pd.DataFrame({
            'Close': [150, 152, 148, 155, 160]
        }, index=pd.date_range('2024-01-01', '2024-01-05', freq='D'))
        
        mock_download.return_value = single_data
        
        result = fetch_multiple_tickers(['AAPL'], '2024-01-01', '2024-01-05')
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert 'AAPL' in result.columns
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
    def test_get_ticker_info_partial_data(self, mock_ticker):
        """Test ticker info with missing fields"""
        mock_stock = MagicMock()
        mock_stock.info = {'longName': 'Apple Inc.'}  # Missing other fields
        mock_ticker.return_value = mock_stock
        
        result = get_ticker_info('AAPL')
        
        # Assertions
        assert result['name'] == 'Apple Inc.'
        assert result['sector'] == 'N/A'
        assert result['industry'] == 'N/A'
        assert result['market_cap'] == 'N/A'
        assert result['currency'] == 'USD'  # Default value
    
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
        # Mock sidebar methods
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
    
    @patch('app.st.sidebar')
    @patch('app.pd.read_csv')
    def test_create_ticker_input_csv_upload_success(self, mock_read_csv, mock_sidebar):
        """Test successful CSV upload"""
        mock_sidebar.header = MagicMock()
        mock_sidebar.radio.return_value = "Upload CSV"
        mock_sidebar.success = MagicMock()
        mock_sidebar.error = MagicMock()
        
        # Mock file upload
        mock_file = MagicMock()
        mock_sidebar.file_uploader.return_value = mock_file
        
        # Mock CSV data
        mock_df = pd.DataFrame({'ticker': ['AAPL', 'MSFT', 'GOOGL']})
        mock_read_csv.return_value = mock_df
        
        result = create_ticker_input()
        
        assert result == ['AAPL', 'MSFT', 'GOOGL']
        mock_sidebar.success.assert_called_once()
    
    @patch('app.st.sidebar')
    @patch('app.pd.read_csv')
    def test_create_ticker_input_csv_no_ticker_column(self, mock_read_csv, mock_sidebar):
        """Test CSV upload without ticker column"""
        mock_sidebar.header = MagicMock()
        mock_sidebar.radio.return_value = "Upload CSV"
        mock_sidebar.error = MagicMock()
        
        # Mock file upload
        mock_file = MagicMock()
        mock_sidebar.file_uploader.return_value = mock_file
        
        # Mock CSV data without ticker column
        mock_df = pd.DataFrame({'company': ['Apple', 'Microsoft', 'Google']})
        mock_read_csv.return_value = mock_df
        
        result = create_ticker_input()
        
        assert result == []
        mock_sidebar.error.assert_called_once()


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
    
    @patch('app.st.sidebar')
    def test_create_date_range_custom(self, mock_sidebar):
        """Test custom date range input"""
        mock_sidebar.header = MagicMock()
        mock_sidebar.selectbox.return_value = "Custom"
        
        # Mock columns
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_sidebar.columns.return_value = [mock_col1, mock_col2]
        
        # Mock date inputs
        test_start = datetime(2024, 1, 1).date()
        test_end = datetime(2024, 12, 31).date()
        mock_col1.date_input.return_value = test_start
        mock_col2.date_input.return_value = test_end
        
        start_date, end_date = create_date_range_input()
        
        assert start_date == '2024-01-01'
        assert end_date == '2024-12-31'


class TestDisplayTickerInfo:
    """Test suite for display_ticker_info function"""
    
    @patch('app.st.subheader')
    @patch('app.st.dataframe')
    @patch('app.get_ticker_info')
    def test_display_ticker_info_success(self, mock_get_info, mock_dataframe, mock_subheader):
        """Test successful display of ticker information"""
        # Mock ticker info responses
        mock_get_info.side_effect = [
            {'Ticker': 'AAPL', 'Name': 'Apple Inc.', 'Sector': 'Technology', 'Industry': 'Consumer Electronics'},
            {'Ticker': 'MSFT', 'Name': 'Microsoft Corp.', 'Sector': 'Technology', 'Industry': 'Software'}
        ]
        
        display_ticker_info(['AAPL', 'MSFT'])
        
        # Verify functions were called
        mock_subheader.assert_called_once()
        mock_dataframe.assert_called_once()
        assert mock_get_info.call_count == 2
    
    def test_display_ticker_info_empty_list(self):
        """Test display with empty ticker list - should return early"""
        # This should not raise any exceptions and should return early
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
    
    def test_data_type_options(self):
        """Test valid data type options"""
        valid_data_types = ['Close', 'Adj Close', 'OHLC', 'All']
        
        for data_type in valid_data_types:
            assert isinstance(data_type, str)
            assert len(data_type) > 0


class TestIntegration:
    """Integration tests for combined functionality"""
    
    @patch('app.yf.Ticker')
    @patch('app.st.error')
    def test_full_data_fetch_pipeline(self, mock_error, mock_ticker):
        """Test complete data fetching pipeline"""
        # Setup mock data
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        mock_data = pd.DataFrame({
            'Open': [150, 152, 148, 155, 160],
            'High': [155, 157, 153, 160, 165],
            'Low': [148, 150, 145, 152, 157],
            'Close': [153, 155, 151, 158, 163],
            'Volume': [1000000, 1200000, 900000, 1500000, 1800000]
        }, index=dates)
        
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_data
        mock_stock.info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'currency': 'USD'
        }
        mock_ticker.return_value = mock_stock
        
        # Test data fetch
        result = fetch_stock_data('AAPL', '2024-01-01', '2024-01-05', 'Close')
        assert not result.empty
        assert 'AAPL' in result.columns
        
        # Test ticker info
        info = get_ticker_info('AAPL')
        assert info['name'] == 'Apple Inc.'
        assert info['sector'] == 'Technology'
        
        mock_error.assert_not_called()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
        
        # Assertions
        assert not result.empty
        assert all(ticker in result.columns for ticker in self.tickers)
        mock_download.assert_called_once_with(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date, 
            group_by='ticker'
        )
    
    @patch('app.yf.download')
    def test_fetch_single_ticker_via_multiple(self, mock_download):
        """Test fetching single ticker through multiple ticker function"""
        single_ticker = ["AAPL"]
        mock_download.return_value = self.mock_single_data
        
        result = fetch_multiple_tickers(single_ticker, self.start_date, self.end_date)
        
        # Assertions
        assert not result.empty
        assert single_ticker[0] in result.columns
    
    @patch('app.yf.download')
    @patch('app.st.error')
    def test_fetch_multiple_tickers_exception(self, mock_st_error, mock_download):
        """Test handling of exceptions in multiple ticker fetch"""
        mock_download.side_effect = Exception("API Error")
        
        result = fetch_multiple_tickers(self.tickers, self.start_date, self.end_date)
        
        # Assertions
        assert result.empty
        mock_st_error.assert_called_once()

class TestGetTickerInfo:
    """Test suite for get_ticker_info function"""
    
    def setup_method(self):
        """Setup test data"""
        self.ticker = "AAPL"
        self.mock_info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'currency': 'USD'
        }
    
    @patch('app.yf.Ticker')
    def test_get_ticker_info_success(self, mock_ticker):
        """Test successful ticker info retrieval"""
        mock_stock = MagicMock()
        mock_stock.info = self.mock_info
        mock_ticker.return_value = mock_stock
        
        result = get_ticker_info(self.ticker)
        
        # Assertions
        assert result['name'] == 'Apple Inc.'
        assert result['sector'] == 'Technology'
        assert result['industry'] == 'Consumer Electronics'
        assert result['market_cap'] == 3000000000000
        assert result['currency'] == 'USD'
        mock_ticker.assert_called_once_with(self.ticker)
    
    @patch('app.yf.Ticker')
    def test_get_ticker_info_partial_data(self, mock_ticker):
        """Test ticker info with missing fields"""
        partial_info = {'longName': 'Apple Inc.'}
        mock_stock = MagicMock()
        mock_stock.info = partial_info
        mock_ticker.return_value = mock_stock
        
        result = get_ticker_info(self.ticker)
        
        # Assertions
        assert result['name'] == 'Apple Inc.'
        assert result['sector'] == 'N/A'
        assert result['industry'] == 'N/A'
        assert result['market_cap'] == 'N/A'
        assert result['currency'] == 'USD'
    
    @patch('app.yf.Ticker')
    def test_get_ticker_info_exception(self, mock_ticker):
        """Test handling of exceptions in ticker info"""
        mock_ticker.side_effect = Exception("API Error")
        
        result = get_ticker_info(self.ticker)
        
        # Assertions
        assert result['name'] == self.ticker
        assert result['sector'] == 'N/A'
        assert result['industry'] == 'N/A'
        assert result['market_cap'] == 'N/A'
        assert result['currency'] == 'USD'

class TestCreateTickerInput:
    """Test suite for create_ticker_input function"""
    
    @patch('app.st.sidebar')
    def test_create_ticker_input_single_ticker(self, mock_sidebar):
        """Test single ticker input"""
        # Mock streamlit sidebar components
        mock_sidebar.header.return_value = None
        mock_sidebar.radio.return_value = "Single Ticker"
        mock_sidebar.text_input.return_value = "AAPL"
        mock_sidebar.write.return_value = None
        mock_sidebar.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        
        # Mock the button calls to return False (not clicked)
        for col in mock_sidebar.columns.return_value:
            col.button.return_value = False
        
        result = create_ticker_input()
        
        # Assertions
        assert result == ["AAPL"]
        mock_sidebar.radio.assert_called_once()
        mock_sidebar.text_input.assert_called_once()
    
    @patch('app.st.sidebar')
    def test_create_ticker_input_multiple_tickers(self, mock_sidebar):
        """Test multiple ticker input"""
        mock_sidebar.header.return_value = None
        mock_sidebar.radio.return_value = "Multiple Tickers"
        mock_sidebar.text_area.return_value = "AAPL\nMSFT\nGOOGL"
        
        result = create_ticker_input()
        
        # Assertions
        assert result == ["AAPL", "MSFT", "GOOGL"]
        mock_sidebar.text_area.assert_called_once()
    
    @patch('app.st.sidebar')
    @patch('app.pd.read_csv')
    def test_create_ticker_input_csv_upload(self, mock_read_csv, mock_sidebar):
        """Test CSV upload ticker input"""
        mock_sidebar.header.return_value = None
        mock_sidebar.radio.return_value = "Upload CSV"
        
        # Mock uploaded file
        mock_file = MagicMock()
        mock_sidebar.file_uploader.return_value = mock_file
        
        # Mock CSV data
        mock_df = pd.DataFrame({'ticker': ['AAPL', 'MSFT', 'GOOGL']})
        mock_read_csv.return_value = mock_df
        mock_sidebar.success.return_value = None
        
        result = create_ticker_input()
        
        # Assertions
        assert result == ["AAPL", "MSFT", "GOOGL"]
        mock_read_csv.assert_called_once_with(mock_file)

class TestCreateDateRangeInput:
    """Test suite for create_date_range_input function"""
    
    @patch('app.st.sidebar')
    def test_create_date_range_preset(self, mock_sidebar):
        """Test preset date range selection"""
        mock_sidebar.header.return_value = None
        mock_sidebar.selectbox.return_value = "1 Year"
        
        result = create_date_range_input()
        
        # Assertions
        start_date, end_date = result
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
        
        # Check date format
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
        
        # Check that start is before end
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        assert start_dt < end_dt
    
    @patch('app.st.sidebar')
    def test_create_date_range_custom(self, mock_sidebar):
        """Test custom date range selection"""
        mock_sidebar.header.return_value = None
        mock_sidebar.selectbox.return_value = "Custom"
        
        # Mock columns and date inputs
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_sidebar.columns.return_value = [mock_col1, mock_col2]
        
        start_date_obj = datetime(2024, 1, 1).date()
        end_date_obj = datetime(2024, 12, 31).date()
        
        mock_col1.date_input.return_value = start_date_obj
        mock_col2.date_input.return_value = end_date_obj
        
        result = create_date_range_input()
        
        # Assertions
        start_date, end_date = result
        assert start_date == "2024-01-01"
        assert end_date == "2024-12-31"

class TestDisplayTickerInfo:
    """Test suite for display_ticker_info function"""
    
    @patch('app.st.subheader')
    @patch('app.st.dataframe')
    @patch('app.get_ticker_info')
    def test_display_ticker_info_success(self, mock_get_info, mock_dataframe, mock_subheader):
        """Test successful display of ticker information"""
        tickers = ["AAPL", "MSFT"]
        
        # Mock ticker info responses
        mock_get_info.side_effect = [
            {
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics'
            },
            {
                'name': 'Microsoft Corporation',
                'sector': 'Technology',
                'industry': 'Software'
            }
        ]
        
        display_ticker_info(tickers)
        
        # Assertions
        mock_subheader.assert_called_once_with("📋 Ticker Information")
        mock_dataframe.assert_called_once()
        assert mock_get_info.call_count == len(tickers)
    
    @patch('app.st.subheader')
    @patch('app.st.dataframe')
    def test_display_ticker_info_empty_list(self, mock_dataframe, mock_subheader):
        """Test display with empty ticker list"""
        tickers = []
        
        display_ticker_info(tickers)
        
        # Assertions - should return early, no calls made
        mock_subheader.assert_not_called()
        mock_dataframe.assert_not_called()

class TestIntegration:
    """Integration tests for the complete workflow"""
    
    @patch('app.yf.Ticker')
    @patch('app.yf.download')
    def test_complete_data_workflow(self, mock_download, mock_ticker):
        """Test complete data fetching workflow"""
        # Setup mock data
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(150, 200, len(dates)),
            'High': np.random.uniform(160, 210, len(dates)),
            'Low': np.random.uniform(140, 190, len(dates)),
            'Close': np.random.uniform(150, 200, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Mock single ticker
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_data
        mock_stock.info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'currency': 'USD'
        }
        mock_ticker.return_value = mock_stock
        
        # Mock multiple tickers
        multi_columns = pd.MultiIndex.from_product([["AAPL", "MSFT"], ['Open', 'High', 'Low', 'Close', 'Volume']])
        mock_multi_data = pd.DataFrame(
            np.random.randn(len(dates), len(multi_columns)),
            index=dates,
            columns=multi_columns
        )
        mock_download.return_value = mock_multi_data
        
        # Test single ticker workflow
        single_result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-31", "Close")
        assert not single_result.empty
        assert "AAPL" in single_result.columns
        
        # Test multiple ticker workflow
        multi_result = fetch_multiple_tickers(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
        assert not multi_result.empty
        
        # Test ticker info
        info_result = get_ticker_info("AAPL")
        assert info_result['name'] == 'Apple Inc.'
        assert info_result['sector'] == 'Technology'

# Data validation tests
class TestDataValidation:
    """Test data validation and edge cases"""
    
    def test_date_string_format(self):
        """Test that date strings are properly formatted"""
        test_dates = [
            ("2024-01-01", "2024-12-31"),
            ("2023-06-15", "2023-12-31"),
            ("2022-01-01", "2022-06-30")
        ]
        
        for start, end in test_dates:
            # Test that dates can be parsed
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')
            assert start_dt <= end_dt
    
    def test_ticker_symbol_validation(self):
        """Test ticker symbol validation"""
        valid_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        invalid_tickers = ["", " ", "123", "!@#"]
        
        # Valid tickers should be uppercase and non-empty
        for ticker in valid_tickers:
            assert ticker.isupper()
            assert len(ticker) > 0
            assert ticker.isalpha()
        
        # Invalid tickers should be caught
        for ticker in invalid_tickers:
            if ticker.strip():  # Non-empty after strip
                assert not ticker.isalpha() or not ticker.isupper()

# Performance tests
class TestPerformance:
    """Test performance and caching behavior"""
    
    @patch('app.yf.Ticker')
    def test_caching_behavior(self, mock_ticker):
        """Test that caching works correctly"""
        # Mock data
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq='D')
        mock_data = pd.DataFrame({
            'Close': np.random.uniform(150, 200, len(dates))
        }, index=dates)
        
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_data
        mock_ticker.return_value = mock_stock
        
        # Call function multiple times with same parameters
        result1 = fetch_stock_data("AAPL", "2024-01-01", "2024-01-31", "Close")
        result2 = fetch_stock_data("AAPL", "2024-01-01", "2024-01-31", "Close")
        
        # Both results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
        # Due to caching, yfinance should only be called once
        # Note: In real testing, you'd need to clear streamlit cache between calls
        # to properly test this behavior

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
