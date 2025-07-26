# Market Data Fetcher

A comprehensive Python application for downloading and managing stock market data from multiple sources including NSE (National Stock Exchange of India), Zerodha KiteConnect API, and various web sources.

## Features

- **NSE Data Fetching**: Direct API integration with NSE for real-time stock quotes, indices data, and market statistics
- **Zerodha Integration**: Complete KiteConnect API integration for trading data, positions, and orders
- **Web Scraping**: Flexible web scraping capabilities for additional market data sources
- **Yahoo Finance Integration**: Historical data and financial information using yfinance
- **Data Management**: SQLite database for data persistence and retrieval
- **Configuration Management**: Flexible configuration system with environment variable support
- **Export Capabilities**: Export data to CSV and JSON formats
- **Watchlist Management**: Personal watchlist functionality
- **Logging**: Comprehensive logging system for monitoring and debugging

## Project Structure

```
market-data/
├── src/
│   ├── data_sources/
│   │   ├── nse_fetcher.py      # NSE API integration
│   │   ├── zerodha_fetcher.py  # Zerodha KiteConnect integration
│   │   └── web_scraper.py      # Web scraping utilities
│   └── utils/
│       ├── config_manager.py   # Configuration management
│       └── data_manager.py     # Database and data operations
├── data/                       # Data storage directory
├── config/                     # Configuration files
├── notebooks/                  # Jupyter notebooks for analysis
├── main.py                     # Main application entry point
└── requirements.txt            # Python dependencies
```

## Installation

1. **Clone the repository** (if applicable) or ensure you're in the project directory
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Copy the environment template**:
   ```bash
   copy .env.example .env  # On Windows
   # cp .env.example .env  # On macOS/Linux
   ```

2. **Edit `.env` file** with your credentials:
   ```env
   ZERODHA_API_KEY=your_zerodha_api_key_here
   ZERODHA_ACCESS_TOKEN=your_zerodha_access_token_here
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
   ```

## Usage

### Basic Usage

Run the main application:
```bash
python main.py
```

### Using Individual Components

#### NSE Data Fetching
```python
from src.data_sources.nse_fetcher import NSEFetcher

nse = NSEFetcher()

# Get index data
nifty_data = nse.get_index_data('NIFTY')
print(f"NIFTY: {nifty_data['lastPrice']}")

# Get stock quote
reliance_data = nse.get_stock_quote('RELIANCE')
print(f"RELIANCE: {reliance_data['data'][0]['lastPrice']}")

# Get top gainers
top_gainers = nse.get_top_gainers(10)
```

#### Zerodha Integration
```python
from src.data_sources.zerodha_fetcher import ZerodhaFetcher

zerodha = ZerodhaFetcher()
zerodha.set_credentials(api_key="your_key", access_token="your_token")

# Get instruments
instruments = zerodha.get_instruments("NSE")

# Get quotes
quotes = zerodha.get_quote(["NSE:RELIANCE"])

# Get historical data
from datetime import datetime, timedelta
historical = zerodha.get_historical_data(
    instrument_token="738561",
    from_date=datetime.now() - timedelta(days=30),
    to_date=datetime.now()
)
```

#### Web Scraping & Yahoo Finance
```python
from src.data_sources.web_scraper import WebScraper

scraper = WebScraper()

# Get Yahoo Finance data
data = scraper.get_yahoo_finance_data('RELIANCE.NS', period='1mo')

# Get stock information
info = scraper.get_stock_info('RELIANCE.NS')

# Get market news
news = scraper.get_market_news()

# Get currency rates
rates = scraper.get_currency_rates()
```

#### Data Management
```python
from src.utils.data_manager import DataManager

dm = DataManager()

# Store data
dm.store_stock_data(data, source="yahoo")

# Retrieve data
stored_data = dm.get_stock_data('RELIANCE', days=30)

# Export to CSV
dm.export_to_csv('RELIANCE')

# Add to watchlist
dm.add_to_watchlist('RELIANCE', name='Reliance Industries')

# Get data summary
summary = dm.get_data_summary()
```

## API Credentials Setup

### Zerodha KiteConnect
1. Visit [Kite Connect](https://kite.trade/) and create an account
2. Generate API key and secret
3. Complete the authentication process to get access token
4. Add credentials to your `.env` file

### Alpha Vantage (Optional)
1. Visit [Alpha Vantage](https://www.alphavantage.co/) and get a free API key
2. Add the key to your `.env` file

## Data Sources

- **NSE India**: Real-time and historical data from National Stock Exchange
- **Yahoo Finance**: Historical data, financial ratios, and company information
- **Zerodha KiteConnect**: Trading data, positions, and order management
- **Web Scraping**: Custom scraping for additional data sources

## Rate Limits

Be mindful of API rate limits:
- NSE: ~30 requests per minute
- Yahoo Finance: ~60 requests per minute  
- Zerodha: Up to 10 requests per second (600 per minute)

## Database Schema

The application uses SQLite with the following main tables:
- `stock_data`: Historical stock prices and volumes
- `indices_data`: Index values and changes
- `news_data`: Market news and updates
- `watchlist`: User's stock watchlist

## Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include error handling and logging
4. Respect API rate limits
5. Test your changes thoroughly

## Disclaimer

This tool is for educational and research purposes only. Always verify data accuracy and comply with the terms of service of data providers. Do not use for unauthorized trading or commercial purposes without proper licenses.

## License

This project is for educational purposes. Please ensure compliance with all data provider terms of service.

## Support

For issues and questions, please check the logs in `market_data.log` for detailed error information.
