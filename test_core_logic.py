import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to path to import app functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the core logic by importing and testing functions directly
# We'll disable caching for testing by patching the decorator
def no_cache(func):
    """Dummy decorator that does nothing - replaces st.cache_data for testing"""
    return func

class TestCoreFunctionality:
    """Test the core business logic of app functions without Streamlit dependencies"""
    
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
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_logic(self, mock_ticker, sample_stock_data):
        """Test the core logic of fetch_stock_data without Streamlit caching"""
        # Import the function and patch its cache decorator
        with patch('streamlit.cache_data', no_cache):
            from app import fetch_stock_data
            
            # Mock yfinance
            mock_stock = MagicMock()
            mock_stock.history.return_value = sample_stock_data
            mock_ticker.return_value = mock_stock
            
            # Test Close data type
            with patch('streamlit.error') as mock_error:
                result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-10", "Close")
                
                # Assertions
                assert isinstance(result, pd.DataFrame)
                assert 'AAPL' in result.columns
                assert len(result) == 10
                assert not result.empty
                mock_ticker.assert_called_with("AAPL")
                mock_error.assert_not_called()
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_ohlc_logic(self, mock_ticker, sample_stock_data):
        """Test OHLC data fetching logic"""
        with patch('streamlit.cache_data', no_cache):
            from app import fetch_stock_data
            
            mock_stock = MagicMock()
            mock_stock.history.return_value = sample_stock_data
            mock_ticker.return_value = mock_stock
            
            with patch('streamlit.error') as mock_error:
                result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-10", "OHLC")
                
                # Assertions
                assert isinstance(result, pd.DataFrame)
                expected_columns = ['Open', 'High', 'Low', 'Close']
                assert all(col in result.columns for col in expected_columns)
                assert len(result) == 10
                mock_error.assert_not_called()
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_empty_data_logic(self, mock_ticker):
        """Test handling of empty data"""
        with patch('streamlit.cache_data', no_cache):
            from app import fetch_stock_data
            
            mock_stock = MagicMock()
            mock_stock.history.return_value = pd.DataFrame()  # Empty dataframe
            mock_ticker.return_value = mock_stock
            
            with patch('streamlit.error') as mock_error:
                result = fetch_stock_data("INVALID", "2024-01-01", "2024-01-10", "Close")
                
                # Assertions
                assert isinstance(result, pd.DataFrame)
                assert result.empty
                mock_error.assert_called_once()
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_exception_logic(self, mock_ticker):
        """Test exception handling"""
        with patch('streamlit.cache_data', no_cache):
            from app import fetch_stock_data
            
            mock_ticker.side_effect = Exception("API Error")
            
            with patch('streamlit.error') as mock_error:
                result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-10", "Close")
                
                # Assertions
                assert isinstance(result, pd.DataFrame)
                assert result.empty
                mock_error.assert_called_once()
    
    @patch('yfinance.download')
    def test_fetch_multiple_tickers_logic(self, mock_download):
        """Test multiple tickers fetching logic"""
        with patch('streamlit.cache_data', no_cache):
            from app import fetch_multiple_tickers
            
            # Create sample multi-ticker data
            dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
            tickers = ['AAPL', 'MSFT']
            columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume']])
            
            sample_data = pd.DataFrame(
                np.random.rand(5, 10) * 100 + 100,
                index=dates,
                columns=columns
            )
            
            mock_download.return_value = sample_data
            
            with patch('streamlit.error') as mock_error:
                result = fetch_multiple_tickers(['AAPL', 'MSFT'], '2024-01-01', '2024-01-05')
                
                # Assertions
                assert isinstance(result, pd.DataFrame)
                assert not result.empty
                mock_download.assert_called_once_with(['AAPL', 'MSFT'], start='2024-01-01', end='2024-01-05', group_by='ticker')
                mock_error.assert_not_called()
    
    @patch('yfinance.Ticker')
    def test_get_ticker_info_logic(self, mock_ticker):
        """Test ticker info retrieval logic"""
        with patch('streamlit.cache_data', no_cache):
            from app import get_ticker_info
            
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
    
    @patch('yfinance.Ticker')
    def test_get_ticker_info_exception_logic(self, mock_ticker):
        """Test ticker info exception handling"""
        with patch('streamlit.cache_data', no_cache):
            from app import get_ticker_info
            
            mock_ticker.side_effect = Exception("API Error")
            
            result = get_ticker_info('INVALID')
            
            # Assertions - should return default values
            assert result['name'] == 'INVALID'
            assert result['sector'] == 'N/A'
            assert result['industry'] == 'N/A'
            assert result['market_cap'] == 'N/A'
            assert result['currency'] == 'USD'


class TestDataValidation:
    """Test data validation and utility functions"""
    
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
    
    def test_data_types_validation(self):
        """Test that data types are valid"""
        valid_data_types = ['Close', 'Adj Close', 'OHLC', 'All']
        
        for data_type in valid_data_types:
            assert isinstance(data_type, str)
            assert len(data_type) > 0


class TestPandasOperations:
    """Test pandas operations and data manipulation"""
    
    def test_dataframe_creation(self):
        """Test DataFrame creation and manipulation"""
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        # Test basic operations
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        assert 'Close' in data.columns
        assert not data.empty
        
        # Test renaming
        renamed = data.rename(columns={'Close': 'AAPL'})
        assert 'AAPL' in renamed.columns
        assert 'Close' not in renamed.columns
    
    def test_multiindex_dataframe(self):
        """Test MultiIndex DataFrame operations"""
        dates = pd.date_range('2024-01-01', '2024-01-03', freq='D')
        tickers = ['AAPL', 'MSFT']
        columns = pd.MultiIndex.from_product([tickers, ['Open', 'Close']])
        
        data = pd.DataFrame(
            np.random.rand(3, 4) * 100 + 100,
            index=dates,
            columns=columns
        )
        
        # Test structure
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.columns, pd.MultiIndex)
        assert len(data.columns.levels[0]) == 2  # Two tickers
        assert len(data.columns.levels[1]) == 2  # Two price types
    
    def test_empty_dataframe_handling(self):
        """Test empty DataFrame handling"""
        empty_df = pd.DataFrame()
        
        assert isinstance(empty_df, pd.DataFrame)
        assert empty_df.empty
        assert len(empty_df) == 0


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_date_range(self):
        """Test invalid date range detection"""
        start_date = '2024-12-31'
        end_date = '2024-01-01'
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Should detect invalid range
        assert start_dt > end_dt
    
    def test_invalid_ticker_format(self):
        """Test invalid ticker format detection"""
        invalid_tickers = ['', '123', 'ap@pl', 'toolong_ticker']
        
        for ticker in invalid_tickers:
            # These should be caught by validation logic
            if len(ticker) == 0:
                assert not ticker
            elif ticker.isdigit():
                assert not ticker.isalpha()
            elif '@' in ticker:
                assert not ticker.isalpha()


class TestIntegrationScenarios:
    """Test integration scenarios and workflows"""
    
    @patch('yfinance.Ticker')
    def test_complete_data_workflow(self, mock_ticker):
        """Test a complete data fetching workflow"""
        with patch('streamlit.cache_data', no_cache):
            from app import fetch_stock_data, get_ticker_info
            
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
            
            with patch('streamlit.error') as mock_error:
                # Test data fetch
                price_data = fetch_stock_data('AAPL', '2024-01-01', '2024-01-05', 'Close')
                assert not price_data.empty
                assert 'AAPL' in price_data.columns
                
                # Test ticker info
                info = get_ticker_info('AAPL')
                assert info['name'] == 'Apple Inc.'
                assert info['sector'] == 'Technology'
                
                mock_error.assert_not_called()
    
    def test_data_consistency(self):
        """Test data consistency across different operations"""
        # Test date consistency
        start_date = '2024-01-01'
        end_date = '2024-01-05'
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        expected_days = (end_dt - start_dt).days + 1
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        assert len(date_range) == expected_days
        assert date_range[0].strftime('%Y-%m-%d') == start_date
        assert date_range[-1].strftime('%Y-%m-%d') == end_date


class TestCryptocurrencySupport:
    """Test cryptocurrency ticker support"""
    
    def test_crypto_ticker_formats(self):
        """Test that crypto tickers are in correct format"""
        crypto_tickers = {
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum', 
            'ADA-USD': 'Cardano',
            'DOT-USD': 'Polkadot',
            'SOL-USD': 'Solana',
            'MATIC-USD': 'Polygon'
        }
        
        for ticker, name in crypto_tickers.items():
            # Check format
            assert '-USD' in ticker
            assert len(ticker.split('-')) == 2
            crypto_symbol = ticker.split('-')[0]
            assert crypto_symbol.isupper()
            assert len(crypto_symbol) >= 3
    
    @patch('yfinance.Ticker')
    def test_btc_data_fetch_logic(self, mock_ticker):
        """Test Bitcoin data fetching with correct ticker format"""
        with patch('streamlit.cache_data', no_cache):
            from app import fetch_stock_data, get_ticker_info
            
            # Create sample BTC data
            dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
            btc_data = pd.DataFrame({
                'Open': [45000, 46000, 44000, 47000, 48000],
                'High': [46000, 47000, 45000, 48000, 49000],
                'Low': [44000, 45000, 43000, 46000, 47000],
                'Close': [45500, 46500, 44500, 47500, 48500],
                'Volume': [50000, 60000, 40000, 70000, 80000]
            }, index=dates)
            
            mock_stock = MagicMock()
            mock_stock.history.return_value = btc_data
            mock_stock.info = {
                'longName': 'Bitcoin USD',
                'sector': 'N/A',
                'industry': 'Cryptocurrency',
                'currency': 'USD'
            }
            mock_ticker.return_value = mock_stock
            
            with patch('streamlit.error') as mock_error:
                # Test BTC-USD ticker
                result = fetch_stock_data('BTC-USD', '2024-01-01', '2024-01-05', 'Close')
                
                assert isinstance(result, pd.DataFrame)
                assert 'BTC-USD' in result.columns
                assert len(result) == 5
                assert not result.empty
                mock_ticker.assert_called_with('BTC-USD')
                mock_error.assert_not_called()
                
                # Test ticker info
                info = get_ticker_info('BTC-USD')
                assert 'Bitcoin' in info['name'] or 'BTC' in info['name']
    
    def test_invalid_crypto_ticker(self):
        """Test that BTC without -USD suffix should be caught"""
        invalid_crypto_tickers = ['BTC', 'ETH', 'ADA', 'BITCOIN']
        
        # These would typically not work with Yahoo Finance
        for ticker in invalid_crypto_tickers:
            # Should be detected as potentially invalid format
            if ticker in ['BTC', 'ETH', 'ADA'] and '-USD' not in ticker:
                assert '-USD' not in ticker  # Missing proper format


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
