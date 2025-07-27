"""
Spot Data Loader Module

This module provides functionality to download and manage spot price data
from Yahoo Finance for any symbol with caching capabilities.
"""

import yfinance as yf
import pandas as pd
import os
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Predefined symbol mappings for common Indian indices and stocks
SYMBOL_MAPPINGS = {
    # Indian Indices
    'BANKNIFTY': '^NSEBANK',
    'NIFTY': '^NSEI',
    'NIFTY50': '^NSEI',
    'SENSEX': '^BSESN',
    'NIFTYIT': '^CNXIT',
    'NIFTYPHARMA': '^CNXPHARMA',
    'NIFTYAUTO': '^CNXAUTO',
    'NIFTYMETAL': '^CNXMETAL',
    'NIFTYBANK': '^NSEBANK',
    
    # US Indices
    'SPY': 'SPY',
    'QQQ': 'QQQ',
    'DIA': 'DIA',
    'SP500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW': '^DJI',
    
    # Crypto (if supported by yfinance)
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'BITCOIN': 'BTC-USD',
    'ETHEREUM': 'ETH-USD',
}


def load_spot_data(
    symbol: str,
    data_path: str,
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    force_download: bool = False,
    plot_data: bool = True,
    custom_filename: Optional[str] = None
) -> pd.DataFrame:
    """
    Load spot price data from Yahoo Finance for any symbol with caching support.
    
    Args:
        symbol (str): Symbol to download (e.g., 'BANKNIFTY', '^NSEBANK', 'AAPL', 'SPY')
        data_path (str): Directory path where the CSV file will be saved/loaded
        start_date (str): Start date for data download (YYYY-MM-DD format)
        end_date (Optional[str]): End date for data download (YYYY-MM-DD format)
        force_download (bool): If True, forces fresh download even if file exists
        plot_data (bool): If True, creates a basic plot of the data
        custom_filename (Optional[str]): Custom filename for the CSV file
        
    Returns:
        pd.DataFrame: Spot data with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
    Raises:
        Exception: If data download fails or file operations fail
    """
    
    # Resolve symbol using predefined mappings
    yf_symbol = resolve_symbol(symbol)
    
    # Generate filename
    if custom_filename:
        filename = custom_filename if custom_filename.endswith('.csv') else f"{custom_filename}.csv"
    else:
        safe_symbol = symbol.replace('^', '').replace('-', '_').replace('.', '_')
        filename = f"{safe_symbol}_yfinance.csv"
    
    spot_data_file = os.path.join(data_path, filename)
    spot_data = pd.DataFrame()
    
    print(f"🔍 Loading spot data for symbol: {symbol} (Yahoo Finance: {yf_symbol})")
    
    # Check if file already exists and load it (unless force download is True)
    if os.path.exists(spot_data_file) and not force_download:
        print("📂 Loading existing spot data from file...")
        try:
            spot_data = pd.read_csv(spot_data_file)
            spot_data['Date'] = pd.to_datetime(spot_data['Date'])
            print(f"✅ Loaded spot data from: {os.path.basename(spot_data_file)}")
            print(f"📋 Data Shape: {spot_data.shape}")
            print(f"📅 Date Range: {spot_data['Date'].min().strftime('%d-%b-%Y')} to {spot_data['Date'].max().strftime('%d-%b-%Y')}")
            print(f"📊 Columns: {list(spot_data.columns)}")
        except Exception as e:
            print(f"❌ Error loading existing file: {e}")
            print("🔄 Will download fresh data...")
            spot_data = pd.DataFrame()
    else:
        if force_download:
            print("🔄 Force download requested. Downloading fresh data...")
        else:
            print("📥 No existing spot data file found. Downloading fresh data...")
    
    # Download fresh data if file doesn't exist, couldn't be loaded, or force download
    if spot_data.empty:
        try:
            print(f"🌐 Downloading spot data for {symbol} from Yahoo Finance...")
            # Download historical data
            spot_data = yf.download(yf_symbol, start=start_date, end=end_date)
            
            if spot_data is not None and not spot_data.empty:
                spot_data.reset_index(inplace=True)
                
                # Debug: Print column structure to understand the data format
                print("📋 Downloaded Spot Data Structure:")
                print(f"Shape: {spot_data.shape}")
                
                # Handle multi-level columns if present (yfinance sometimes returns multi-level columns)
                if isinstance(spot_data.columns, pd.MultiIndex):
                    # Flatten multi-level columns
                    spot_data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in spot_data.columns]
                
                # Standardize column names
                column_mapping = {}
                for col in spot_data.columns:
                    col_lower = str(col).lower()
                    if 'open' in col_lower:
                        column_mapping[col] = 'Open'
                    elif 'high' in col_lower:
                        column_mapping[col] = 'High'
                    elif 'low' in col_lower:
                        column_mapping[col] = 'Low'
                    elif 'close' in col_lower or 'adj close' in col_lower:
                        column_mapping[col] = 'Close'
                    elif 'volume' in col_lower:
                        column_mapping[col] = 'Volume'
                
                # Rename columns
                spot_data.rename(columns=column_mapping, inplace=True)
                print(f"Final columns: {list(spot_data.columns)}")
                
                # Save to CSV file
                try:
                    # Ensure data directory exists
                    os.makedirs(data_path, exist_ok=True)
                    
                    # Save to CSV
                    spot_data.to_csv(spot_data_file, index=False)
                    print(f"💾 Spot data saved to: {spot_data_file}")
                    print(f"📁 File size: {os.path.getsize(spot_data_file):,} bytes")
                    
                    # Display summary of saved data
                    print(f"\n📊 SAVED DATA SUMMARY:")
                    print("-" * 25)
                    print(f"📈 Records: {len(spot_data):,}")
                    print(f"📅 Date Range: {spot_data['Date'].min().strftime('%d-%b-%Y')} to {spot_data['Date'].max().strftime('%d-%b-%Y')}")
                    print(f"📋 Columns: {len(spot_data.columns)}")
                    
                except Exception as e:
                    print(f"⚠️ Warning: Could not save data to file: {e}")
                    print("📊 Data downloaded successfully but not saved to disk")
                
            else:
                print(f"❌ No data returned from yfinance for symbol: {symbol} ({yf_symbol}).")
                spot_data = pd.DataFrame()
        except Exception as e:
            print(f"❌ Error downloading spot data for {symbol}: {e}")
            spot_data = pd.DataFrame()
    
    # Create basic plot if requested and data is available
    if plot_data and not spot_data.empty:
        create_spot_plot(spot_data, symbol)
    
    return spot_data


def resolve_symbol(symbol: str) -> str:
    """
    Resolve symbol using predefined mappings or return as-is.
    
    Args:
        symbol (str): Input symbol
        
    Returns:
        str: Yahoo Finance compatible symbol
    """
    symbol_upper = symbol.upper()
    
    # Check if symbol exists in our mappings
    if symbol_upper in SYMBOL_MAPPINGS:
        resolved = SYMBOL_MAPPINGS[symbol_upper]
        print(f"🔄 Symbol mapping: {symbol} → {resolved}")
        return resolved
    
    # Return as-is if no mapping found
    print(f"💡 Using symbol as-is: {symbol}")
    return symbol


def create_spot_plot(spot_data: pd.DataFrame, symbol: str) -> None:
    """
    Create a basic matplotlib plot of spot data.
    
    Args:
        spot_data (pd.DataFrame): Spot data with Date and OHLC columns
        symbol (str): Symbol name for plot title
    """
    if spot_data.empty:
        print("❌ No spot data available to plot.")
        return
    
    print(f"\n📈 Plotting {symbol} Spot Price Data (Yahoo Finance)")
    plt.figure(figsize=(14, 6))
    
    # Check which columns are available
    required_cols = ['Open', 'High', 'Low', 'Close']
    available_cols = [col for col in required_cols if col in spot_data.columns]
    
    if 'Date' in spot_data.columns and len(available_cols) >= 2:
        # Drop rows with NaN in available columns
        plot_data = spot_data.dropna(subset=['Date'] + available_cols)
        print(f"Using columns: {available_cols}")
        print(f"Plot data shape: {plot_data.shape}")
        
        # Plot Close price (or first available price column)
        price_col = 'Close' if 'Close' in available_cols else available_cols[0]
        plt.plot(plot_data['Date'], plot_data[price_col], label=f'{price_col} Price', color='blue', linewidth=2)
        
        # Add fill_between if High and Low are available
        if 'High' in available_cols and 'Low' in available_cols:
            plt.fill_between(plot_data['Date'], plot_data['Low'], plot_data['High'], 
                           color='skyblue', alpha=0.2, label='Daily Range')
        
        plt.title(f'{symbol} Spot Price (Yahoo Finance)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print(f"❌ Required columns not found. Available columns: {list(spot_data.columns)}")
        print("Cannot create plot without proper price data.")


def load_multiple_symbols(
    symbols: List[str],
    data_path: str,
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    force_download: bool = False,
    plot_data: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load spot data for multiple symbols.
    
    Args:
        symbols (List[str]): List of symbols to download
        data_path (str): Directory path where CSV files will be saved/loaded
        start_date (str): Start date for data download (YYYY-MM-DD format)
        end_date (Optional[str]): End date for data download (YYYY-MM-DD format)
        force_download (bool): If True, forces fresh download even if files exist
        plot_data (bool): If True, creates plots for each symbol
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with symbol as key and DataFrame as value
    """
    
    print(f"📊 Loading data for {len(symbols)} symbols")
    print("-" * 40)
    
    results = {}
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{i}/{len(symbols)} - Processing {symbol}...")
        try:
            data = load_spot_data(
                symbol=symbol,
                data_path=data_path,
                start_date=start_date,
                end_date=end_date,
                force_download=force_download,
                plot_data=plot_data
            )
            results[symbol] = data
            
            if not data.empty:
                print(f"✅ {symbol}: {len(data):,} records loaded")
            else:
                print(f"❌ {symbol}: No data loaded")
                
        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")
            results[symbol] = pd.DataFrame()
    
    print(f"\n🎯 MULTI-SYMBOL LOADING SUMMARY:")
    print("-" * 35)
    successful = sum(1 for df in results.values() if not df.empty)
    print(f"✅ Successfully loaded: {successful}/{len(symbols)} symbols")
    
    return results


def get_spot_summary(spot_data: pd.DataFrame, symbol: str = "Symbol") -> dict:
    """
    Get summary statistics for spot data.
    
    Args:
        spot_data (pd.DataFrame): Spot data
        symbol (str): Symbol name for the summary
        
    Returns:
        dict: Summary statistics including data shape, date range, and price statistics
    """
    if spot_data.empty:
        return {"error": "No data available", "symbol": symbol}
    
    summary = {
        "symbol": symbol,
        "shape": spot_data.shape,
        "columns": list(spot_data.columns),
        "records": len(spot_data)
    }
    
    if 'Date' in spot_data.columns:
        summary["date_range"] = {
            "start": spot_data['Date'].min(),
            "end": spot_data['Date'].max()
        }
    
    # Price statistics for available price columns
    price_cols = ['Open', 'High', 'Low', 'Close']
    available_price_cols = [col for col in price_cols if col in spot_data.columns]
    
    if available_price_cols:
        summary["price_stats"] = {}
        for col in available_price_cols:
            summary["price_stats"][col] = {
                "mean": spot_data[col].mean(),
                "std": spot_data[col].std(),
                "min": spot_data[col].min(),
                "max": spot_data[col].max()
            }
    
    return summary


def get_available_symbols() -> Dict[str, str]:
    """
    Get dictionary of available predefined symbol mappings.
    
    Returns:
        Dict[str, str]: Dictionary of symbol mappings
    """
    return SYMBOL_MAPPINGS.copy()


def search_symbols(query: str) -> Dict[str, str]:
    """
    Search for symbols containing the query string.
    
    Args:
        query (str): Search query
        
    Returns:
        Dict[str, str]: Dictionary of matching symbol mappings
    """
    query_lower = query.lower()
    return {k: v for k, v in SYMBOL_MAPPINGS.items() if query_lower in k.lower() or query_lower in v.lower()}


# Legacy support functions for backward compatibility
def load_banknifty_data(
    data_path: str,
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    force_download: bool = False,
    plot_data: bool = True
) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    Loads Bank Nifty data using the new spot_data_loader.
    
    Args:
        data_path (str): Directory path where the CSV file will be saved/loaded
        start_date (str): Start date for data download (YYYY-MM-DD format)
        end_date (Optional[str]): End date for data download (YYYY-MM-DD format)
        force_download (bool): If True, forces fresh download even if file exists
        plot_data (bool): If True, creates a basic plot of the data
        
    Returns:
        pd.DataFrame: Bank Nifty data
    """
    print("⚠️ Using legacy function. Consider switching to load_spot_data() for more flexibility.")
    return load_spot_data(
        symbol='BANKNIFTY',
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        force_download=force_download,
        plot_data=plot_data,
        custom_filename='bank_nifty_yfinance.csv'
    )


# Factory functions for easy import
def load_and_plot_spot(symbol: str, data_path: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function that loads spot data and creates a plot.
    
    Args:
        symbol (str): Symbol to download
        data_path (str): Directory path for data storage
        **kwargs: Additional arguments passed to load_spot_data
        
    Returns:
        pd.DataFrame: Spot data
    """
    return load_spot_data(symbol, data_path, plot_data=True, **kwargs)


def quick_load(symbol: str, data_path: str, days: int = 365) -> pd.DataFrame:
    """
    Quick load function for recent data.
    
    Args:
        symbol (str): Symbol to download
        data_path (str): Directory path for data storage
        days (int): Number of days of recent data to load
        
    Returns:
        pd.DataFrame: Recent spot data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return load_spot_data(
        symbol=symbol,
        data_path=data_path,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        plot_data=False
    )
