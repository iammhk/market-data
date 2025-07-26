"""
NSE Data Fetcher
Module for fetching data from National Stock Exchange (NSE) of India.
"""

import requests
import pandas as pd
import json
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class NSEFetcher:
    """Class to fetch data from NSE."""
    
    def __init__(self):
        """Initialize NSE Fetcher with required headers and session."""
        self.base_url = "https://www.nseindia.com"
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session.headers.update(self.headers)
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize session by visiting NSE homepage to get cookies."""
        try:
            response = self.session.get(self.base_url)
            logger.info(f"Session initialized. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to initialize session: {str(e)}")
    
    def get_index_data(self, index_name: str = "NIFTY") -> Optional[Dict[str, Any]]:
        """
        Get index data for specified index.
        
        Args:
            index_name: Name of the index (e.g., 'NIFTY', 'BANKNIFTY', 'NIFTYIT')
            
        Returns:
            Dictionary containing index data or None if failed
        """
        try:
            url = f"{self.base_url}/api/allIndices"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for index in data.get('data', []):
                    if index.get('index', '').upper() == index_name.upper():
                        logger.info(f"Successfully fetched data for {index_name}")
                        return index
                
                logger.warning(f"Index {index_name} not found in response")
                return None
            else:
                logger.error(f"Failed to fetch index data. Status code: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching index data for {index_name}: {str(e)}")
            return None
    
    def get_stock_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get stock quote for a specific symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            
        Returns:
            Dictionary containing stock quote data or None if failed
        """
        try:
            url = f"{self.base_url}/api/quote-equity?symbol={symbol.upper()}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched quote for {symbol}")
                return data
            else:
                logger.error(f"Failed to fetch quote for {symbol}. Status code: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return None
    
    def get_stock_codes(self) -> List[str]:
        """
        Get list of all stock symbols available on NSE.
        
        Returns:
            List of stock symbols
        """
        try:
            # This is a simplified approach - you might want to use a more comprehensive endpoint
            url = f"{self.base_url}/api/equity-stockIndices?index=NIFTY%2050"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                symbols = [stock.get('symbol', '') for stock in data.get('data', [])]
                logger.info(f"Successfully fetched {len(symbols)} stock symbols")
                return symbols
            else:
                logger.error(f"Failed to fetch stock codes. Status code: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching stock codes: {str(e)}")
            return []
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical data for a stock (Note: NSE API has limitations for historical data).
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            # This is a placeholder - NSE doesn't provide easy historical data API
            # You might want to use alternative sources like Yahoo Finance
            logger.warning("Historical data from NSE API is limited. Consider using yfinance or other sources.")
            
            # For demonstration, return current quote as single row DataFrame
            quote_data = self.get_stock_quote(symbol)
            if quote_data and 'data' in quote_data:
                stock_data = quote_data['data'][0]
                df = pd.DataFrame([{
                    'symbol': stock_data.get('symbol'),
                    'open': stock_data.get('open'),
                    'high': stock_data.get('dayHigh'),
                    'low': stock_data.get('dayLow'),
                    'close': stock_data.get('lastPrice'),
                    'volume': stock_data.get('totalTradedVolume'),
                    'date': datetime.now().strftime('%Y-%m-%d')
                }])
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def get_top_gainers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top gaining stocks.
        
        Args:
            limit: Number of top gainers to return
            
        Returns:
            List of dictionaries containing top gainer data
        """
        try:
            url = f"{self.base_url}/api/equity-stockIndices?index=NIFTY%2050"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                stocks = data.get('data', [])
                
                # Sort by percentage change
                gainers = sorted(stocks, key=lambda x: float(x.get('pChange', 0)), reverse=True)
                top_gainers = gainers[:limit]
                
                logger.info(f"Successfully fetched top {len(top_gainers)} gainers")
                return top_gainers
            else:
                logger.error(f"Failed to fetch top gainers. Status code: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching top gainers: {str(e)}")
            return []
    
    def get_top_losers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top losing stocks.
        
        Args:
            limit: Number of top losers to return
            
        Returns:
            List of dictionaries containing top loser data
        """
        try:
            url = f"{self.base_url}/api/equity-stockIndices?index=NIFTY%2050"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                stocks = data.get('data', [])
                
                # Sort by percentage change (ascending for losers)
                losers = sorted(stocks, key=lambda x: float(x.get('pChange', 0)))
                top_losers = losers[:limit]
                
                logger.info(f"Successfully fetched top {len(top_losers)} losers")
                return top_losers
            else:
                logger.error(f"Failed to fetch top losers. Status code: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching top losers: {str(e)}")
            return []
