"""
Bank Nifty Data Loader Module

This module provides functionality to download and manage Bank Nifty index data
from Yahoo Finance with caching capabilities.
"""

import yfinance as yf
import pandas as pd
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def load_banknifty_data(
    data_path: str,
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    force_download: bool = False,
    plot_data: bool = True
) -> pd.DataFrame:
    """
    Load Bank Nifty index data from Yahoo Finance with caching support.
    
    Args:
        data_path (str): Directory path where the CSV file will be saved/loaded
        start_date (str): Start date for data download (YYYY-MM-DD format)
        end_date (Optional[str]): End date for data download (YYYY-MM-DD format)
        force_download (bool): If True, forces fresh download even if file exists
        plot_data (bool): If True, creates a basic plot of the data
        
    Returns:
        pd.DataFrame: Bank Nifty data with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
    Raises:
        Exception: If data download fails or file operations fail
    """
    
    # Define file path for saving Bank Nifty data
    bank_nifty_file = os.path.join(data_path, "bank_nifty_yfinance.csv")
    bank_nifty = pd.DataFrame()
    
    # Check if file already exists and load it (unless force download is True)
    if os.path.exists(bank_nifty_file) and not force_download:
        print("📂 Loading existing Bank Nifty data from file...")
        try:
            bank_nifty = pd.read_csv(bank_nifty_file)
            bank_nifty['Date'] = pd.to_datetime(bank_nifty['Date'])
            print(f"✅ Loaded Bank Nifty data from: {os.path.basename(bank_nifty_file)}")
            print(f"📋 Data Shape: {bank_nifty.shape}")
            print(f"📅 Date Range: {bank_nifty['Date'].min().strftime('%d-%b-%Y')} to {bank_nifty['Date'].max().strftime('%d-%b-%Y')}")
            print(f"📊 Columns: {list(bank_nifty.columns)}")
        except Exception as e:
            print(f"❌ Error loading existing file: {e}")
            print("🔄 Will download fresh data...")
            bank_nifty = pd.DataFrame()
    else:
        if force_download:
            print("🔄 Force download requested. Downloading fresh data...")
        else:
            print("📥 No existing Bank Nifty file found. Downloading fresh data...")
    
    # Download fresh data if file doesn't exist, couldn't be loaded, or force download
    if bank_nifty.empty:
        try:
            print("🌐 Downloading Bank Nifty data from Yahoo Finance...")
            # Download historical data for Bank Nifty index (symbol: "^NSEBANK" on Yahoo Finance)
            bank_nifty = yf.download("^NSEBANK", start=start_date, end=end_date)
            
            if bank_nifty is not None and not bank_nifty.empty:
                bank_nifty.reset_index(inplace=True)
                
                # Debug: Print column structure to understand the data format
                print("📋 Downloaded Bank Nifty Data Structure:")
                print(f"Shape: {bank_nifty.shape}")
                
                # Handle multi-level columns if present (yfinance sometimes returns multi-level columns)
                if isinstance(bank_nifty.columns, pd.MultiIndex):
                    # Flatten multi-level columns
                    bank_nifty.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in bank_nifty.columns]
                
                # Standardize column names
                column_mapping = {}
                for col in bank_nifty.columns:
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
                bank_nifty.rename(columns=column_mapping, inplace=True)
                print(f"Final columns: {list(bank_nifty.columns)}")
                
                # Save to CSV file
                try:
                    # Ensure data directory exists
                    os.makedirs(data_path, exist_ok=True)
                    
                    # Save to CSV
                    bank_nifty.to_csv(bank_nifty_file, index=False)
                    print(f"💾 Bank Nifty data saved to: {bank_nifty_file}")
                    print(f"📁 File size: {os.path.getsize(bank_nifty_file):,} bytes")
                    
                    # Display summary of saved data
                    print(f"\n📊 SAVED DATA SUMMARY:")
                    print("-" * 25)
                    print(f"📈 Records: {len(bank_nifty):,}")
                    print(f"📅 Date Range: {bank_nifty['Date'].min().strftime('%d-%b-%Y')} to {bank_nifty['Date'].max().strftime('%d-%b-%Y')}")
                    print(f"📋 Columns: {len(bank_nifty.columns)}")
                    
                except Exception as e:
                    print(f"⚠️ Warning: Could not save data to file: {e}")
                    print("📊 Data downloaded successfully but not saved to disk")
                
            else:
                print("❌ No data returned from yfinance for Bank Nifty.")
                bank_nifty = pd.DataFrame()
        except Exception as e:
            print(f"❌ Error downloading Bank Nifty data: {e}")
            bank_nifty = pd.DataFrame()
    
    # Create basic plot if requested and data is available
    if plot_data and not bank_nifty.empty:
        create_banknifty_plot(bank_nifty)
    
    return bank_nifty


def create_banknifty_plot(bank_nifty: pd.DataFrame) -> None:
    """
    Create a basic matplotlib plot of Bank Nifty data.
    
    Args:
        bank_nifty (pd.DataFrame): Bank Nifty data with Date and OHLC columns
    """
    if bank_nifty.empty:
        print("❌ No Bank Nifty data available to plot.")
        return
    
    print("\n📈 Plotting Bank Nifty Index Data (Yahoo Finance)")
    plt.figure(figsize=(14, 6))
    
    # Check which columns are available
    required_cols = ['Open', 'High', 'Low', 'Close']
    available_cols = [col for col in required_cols if col in bank_nifty.columns]
    
    if 'Date' in bank_nifty.columns and len(available_cols) >= 2:
        # Drop rows with NaN in available columns
        plot_data = bank_nifty.dropna(subset=['Date'] + available_cols)
        print(f"Using columns: {available_cols}")
        print(f"Plot data shape: {plot_data.shape}")
        
        # Plot Close price (or first available price column)
        price_col = 'Close' if 'Close' in available_cols else available_cols[0]
        plt.plot(plot_data['Date'], plot_data[price_col], label=f'{price_col} Price', color='blue', linewidth=2)
        
        # Add fill_between if High and Low are available
        if 'High' in available_cols and 'Low' in available_cols:
            plt.fill_between(plot_data['Date'], plot_data['Low'], plot_data['High'], 
                           color='skyblue', alpha=0.2, label='Daily Range')
        
        plt.title('Bank Nifty Index Price (Yahoo Finance)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Index Price (₹)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print(f"❌ Required columns not found. Available columns: {list(bank_nifty.columns)}")
        print("Cannot create plot without proper price data.")


def get_banknifty_summary(bank_nifty: pd.DataFrame) -> dict:
    """
    Get summary statistics for Bank Nifty data.
    
    Args:
        bank_nifty (pd.DataFrame): Bank Nifty data
        
    Returns:
        dict: Summary statistics including data shape, date range, and price statistics
    """
    if bank_nifty.empty:
        return {"error": "No data available"}
    
    summary = {
        "shape": bank_nifty.shape,
        "columns": list(bank_nifty.columns),
        "records": len(bank_nifty)
    }
    
    if 'Date' in bank_nifty.columns:
        summary["date_range"] = {
            "start": bank_nifty['Date'].min(),
            "end": bank_nifty['Date'].max()
        }
    
    # Price statistics for available price columns
    price_cols = ['Open', 'High', 'Low', 'Close']
    available_price_cols = [col for col in price_cols if col in bank_nifty.columns]
    
    if available_price_cols:
        summary["price_stats"] = {}
        for col in available_price_cols:
            summary["price_stats"][col] = {
                "mean": bank_nifty[col].mean(),
                "std": bank_nifty[col].std(),
                "min": bank_nifty[col].min(),
                "max": bank_nifty[col].max()
            }
    
    return summary


# Factory function for easy import
def load_and_plot_banknifty(data_path: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function that loads Bank Nifty data and creates a plot.
    
    Args:
        data_path (str): Directory path for data storage
        **kwargs: Additional arguments passed to load_banknifty_data
        
    Returns:
        pd.DataFrame: Bank Nifty data
    """
    return load_banknifty_data(data_path, plot_data=True, **kwargs)
