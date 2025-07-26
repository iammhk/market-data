"""
Market Data Fetcher - Main Module
A comprehensive tool for downloading stock market data from NSE, Zerodha, and other internet sources.
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.data_sources.nse_fetcher import NSEFetcher
from src.data_sources.zerodha_fetcher import ZerodhaFetcher
from src.data_sources.web_scraper import WebScraper
from src.utils.data_manager import DataManager
from src.utils.config_manager import ConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_data.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate the market data fetcher."""
    try:
        logger.info("Starting Market Data Fetcher...")
        
        # Initialize configuration
        config = ConfigManager()
        
        # Initialize data sources
        nse_fetcher = NSEFetcher()
        web_scraper = WebScraper()
        
        # Example: Fetch NSE data
        logger.info("Fetching NSE data...")
        nifty_data = nse_fetcher.get_index_data('NIFTY')
        if nifty_data:
            logger.info(f"NIFTY current value: {nifty_data.get('lastPrice', 'N/A')}")
        
        # Example: Get stock list
        stock_list = nse_fetcher.get_stock_codes()
        logger.info(f"Found {len(stock_list)} stocks in NSE")
        
        logger.info("Market Data Fetcher completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
