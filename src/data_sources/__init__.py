"""
Data Sources Package
Contains modules for fetching data from various sources like NSE, Zerodha, and web scraping.
"""

from .nse_fetcher import NSEFetcher
from .zerodha_fetcher import ZerodhaFetcher
from .web_scraper import WebScraper

__all__ = ['NSEFetcher', 'ZerodhaFetcher', 'WebScraper']
