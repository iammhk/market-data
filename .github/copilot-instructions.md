# Copilot Instructions for Market Data Project

This is a Python project for downloading and managing stock market data from multiple sources including NSE (National Stock Exchange of India), Zerodha KiteConnect API, and various web sources.

## Project Structure
- `src/data_sources/`: Contains modules for different data sources (NSE, Zerodha, web scraping)
- `src/utils/`: Utility modules for configuration and data management
- `data/`: Directory for storing downloaded data and database files
- `config/`: Configuration files
- `notebooks/`: Jupyter notebooks for data analysis

## Key Components
1. **NSE Fetcher**: Direct API calls to NSE for real-time and historical data
2. **Zerodha Fetcher**: Integration with Zerodha KiteConnect API for trading data
3. **Web Scraper**: Generic web scraping capabilities using BeautifulSoup and yfinance
4. **Data Manager**: SQLite database management for data storage and retrieval
5. **Config Manager**: Configuration management with support for environment variables

## Dependencies
- pandas, numpy: Data manipulation and analysis
- requests, beautifulsoup4: Web scraping and HTTP requests
- yfinance: Yahoo Finance data integration
- kiteconnect: Zerodha API integration
- sqlite3: Database operations
- python-dotenv: Environment variable management

## Development Guidelines
- Always handle exceptions gracefully with proper logging
- Use type hints for better code clarity
- Follow PEP 8 style guidelines
- Include comprehensive docstrings for all functions
- Implement rate limiting to respect API limitations
- Store sensitive credentials in environment variables, never in code

## Common Tasks
- Fetching real-time stock quotes
- Downloading historical data
- Managing watchlists
- Exporting data to CSV/JSON formats
- Database operations for data persistence

When working with this codebase, prioritize data reliability, error handling, and respect for API rate limits.
