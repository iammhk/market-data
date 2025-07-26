"""
Web Scraper for Market Data
Module for scraping stock market data from various websites.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any
import time
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WebScraper:
    """Class for scraping market data from various web sources."""
    
    def __init__(self):
        """Initialize web scraper with headers."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_yahoo_finance_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """
        Get stock data from Yahoo Finance using yfinance.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS' for NSE, 'AAPL' for NASDAQ)
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                logger.info(f"Successfully fetched {len(data)} records for {symbol} from Yahoo Finance")
                return data
            else:
                logger.warning(f"No data found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {str(e)}")
            return None
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed stock information from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info:
                logger.info(f"Successfully fetched info for {symbol}")
                return info
            else:
                logger.warning(f"No info found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return None
    
    def get_multiple_stocks_data(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            period: Time period
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        results = {}
        for symbol in symbols:
            data = self.get_yahoo_finance_data(symbol, period)
            if data is not None:
                results[symbol] = data
            time.sleep(0.1)  # Small delay to avoid overwhelming the API
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def scrape_economic_times_indices(self) -> Optional[List[Dict[str, Any]]]:
        """
        Scrape major indices data from Economic Times.
        
        Returns:
            List of dictionaries with index data or None if failed
        """
        try:
            url = "https://economictimes.indiatimes.com/markets"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # This is a simplified example - you would need to adapt based on the actual HTML structure
                indices_data = []
                
                # Find index data elements (this would need to be customized based on actual HTML)
                index_elements = soup.find_all('div', class_='marketstats')
                
                for element in index_elements[:5]:  # Limit to first 5 for demo
                    try:
                        name = element.find('span', class_='name').text.strip()
                        value = element.find('span', class_='value').text.strip()
                        change = element.find('span', class_='change').text.strip()
                        
                        indices_data.append({
                            'name': name,
                            'value': value,
                            'change': change,
                            'timestamp': datetime.now().isoformat()
                        })
                    except AttributeError:
                        continue
                
                logger.info(f"Successfully scraped {len(indices_data)} indices from Economic Times")
                return indices_data
            else:
                logger.error(f"Failed to fetch Economic Times data. Status code: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping Economic Times indices: {str(e)}")
            return None
    
    def get_nse_symbols_list(self) -> List[str]:
        """
        Get list of NSE stock symbols using various methods.
        
        Returns:
            List of NSE stock symbols
        """
        symbols = []
        
        # Method 1: Try to get from a known source
        try:
            # This is a placeholder - you would implement based on available data sources
            nifty_50_symbols = [
                'RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFC.NS', 'HDFCBANK.NS',
                'ICICIBANK.NS', 'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS',
                'ITC.NS', 'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS',
                'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'WIPRO.NS',
                'M&M.NS', 'NTPC.NS', 'TECHM.NS', 'HCLTECH.NS', 'POWERGRID.NS',
                'TATAMOTORS.NS', 'BAJFINANCE.NS', 'TATASTEEL.NS', 'ADANIGREEN.NS', 'ONGC.NS'
            ]
            symbols.extend(nifty_50_symbols)
            
            logger.info(f"Loaded {len(symbols)} NSE symbols")
            
        except Exception as e:
            logger.error(f"Error loading NSE symbols: {str(e)}")
        
        return symbols
    
    def get_market_news(self, source: str = "yahoo") -> List[Dict[str, Any]]:
        """
        Get market news from various sources.
        
        Args:
            source: News source ('yahoo', 'economic_times', etc.)
            
        Returns:
            List of news articles
        """
        news_articles = []
        
        try:
            if source.lower() == "yahoo":
                # Use yfinance to get news for major indices
                ticker = yf.Ticker("^NSEI")  # NIFTY 50
                news = ticker.news
                
                for article in news[:10]:  # Limit to 10 articles
                    news_articles.append({
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('link', ''),
                        'published': article.get('providerPublishTime', ''),
                        'source': article.get('publisher', '')
                    })
                
                logger.info(f"Successfully fetched {len(news_articles)} news articles from Yahoo")
            
        except Exception as e:
            logger.error(f"Error fetching market news from {source}: {str(e)}")
        
        return news_articles
    
    def get_currency_rates(self) -> Dict[str, float]:
        """
        Get currency exchange rates.
        
        Returns:
            Dictionary with currency pairs and their rates
        """
        rates = {}
        
        try:
            # Common currency pairs
            currency_pairs = ['USDINR=X', 'EURINR=X', 'GBPINR=X', 'JPYINR=X']
            
            for pair in currency_pairs:
                ticker = yf.Ticker(pair)
                data = ticker.history(period="1d")
                
                if not data.empty:
                    latest_rate = data['Close'].iloc[-1]
                    rates[pair.replace('=X', '')] = round(latest_rate, 4)
            
            logger.info(f"Successfully fetched {len(rates)} currency rates")
            
        except Exception as e:
            logger.error(f"Error fetching currency rates: {str(e)}")
        
        return rates
    
    def get_commodities_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get commodities data.
        
        Returns:
            Dictionary with commodity data
        """
        commodities = {}
        
        try:
            # Common commodities
            commodity_symbols = {
                'Gold': 'GC=F',
                'Silver': 'SI=F',
                'Crude Oil': 'CL=F',
                'Natural Gas': 'NG=F'
            }
            
            for name, symbol in commodity_symbols.items():
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                
                if not data.empty:
                    latest_data = data.iloc[-1]
                    commodities[name] = {
                        'price': round(latest_data['Close'], 2),
                        'change': round(latest_data['Close'] - latest_data['Open'], 2),
                        'volume': int(latest_data['Volume']),
                        'timestamp': datetime.now().isoformat()
                    }
            
            logger.info(f"Successfully fetched data for {len(commodities)} commodities")
            
        except Exception as e:
            logger.error(f"Error fetching commodities data: {str(e)}")
        
        return commodities
