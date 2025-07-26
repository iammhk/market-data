"""
Market Data Fetcher Package
A comprehensive tool for downloading stock market data from NSE, Zerodha, and other internet sources.
"""

__version__ = "1.0.0"
__author__ = "Market Data Team"
__description__ = "Stock market data fetcher for NSE, Zerodha, and web sources"

from .data_sources.nse_fetcher import NSEFetcher
from .data_sources.zerodha_fetcher import ZerodhaFetcher
from .data_sources.web_scraper import WebScraper
from .utils.data_manager import DataManager
from .utils.config_manager import ConfigManager

__all__ = [
    'NSEFetcher',
    'ZerodhaFetcher', 
    'WebScraper',
    'DataManager',
    'ConfigManager'
]
