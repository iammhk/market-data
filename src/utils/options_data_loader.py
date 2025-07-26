"""
Options Data Loader Module

This module provides functionality to load and process Bank Nifty options data
from CSV files, including data cleaning, type conversion, and separation of
Call (CE) and Put (PE) options.
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any


def load_banknifty_options_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process Bank Nifty options data from CSV files.
    
    This function reads all OPTIDX_BANKNIFTY CSV files from the specified directory,
    cleans the data, converts data types, and separates Call and Put options.
    
    Args:
        data_path (str): Path to the directory containing the CSV files
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - df_call: DataFrame with Call (CE) options data
            - df_put: DataFrame with Put (PE) options data  
            - options_merged: DataFrame with all options data combined
            
    Raises:
        FileNotFoundError: If the data directory doesn't exist
        ValueError: If no valid CSV files are found
    """
    
    print("📂 LOADING BANK NIFTY OPTIONS DATA")
    print("=" * 40)
    
    # Validate data path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find all OPTIDX_BANKNIFTY CSV files
    csv_pattern = os.path.join(data_path, "OPTIDX_BANKNIFTY*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise ValueError(f"No OPTIDX_BANKNIFTY CSV files found in {data_path}")
    
    print(f"📋 Found {len(csv_files)} Bank Nifty options files:")
    for i, file in enumerate(csv_files, 1):
        filename = os.path.basename(file)
        print(f"  {i}. {filename}")
    
    # Read and merge all CSV files
    all_options_data = []
    
    for file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(file)
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            # Add source file information
            df['Source_File'] = os.path.basename(file)
            
            all_options_data.append(df)
            print(f"✅ Loaded: {os.path.basename(file)} - {len(df)} records")
            
        except Exception as e:
            print(f"❌ Error loading {os.path.basename(file)}: {str(e)}")
    
    if not all_options_data:
        raise ValueError("No data files were successfully loaded")
    
    # Combine all dataframes
    options_merged = pd.concat(all_options_data, ignore_index=True)
    
    # Clean and process the merged data
    options_merged = _clean_options_data(options_merged)
    
    # Separate Call (CE) and Put (PE) options
    df_call, df_put = _separate_call_put_options(all_options_data, csv_files, options_merged)
    
    # Display summary information
    _display_data_summary(options_merged, df_call, df_put)
    
    # Perform data quality checks
    _perform_data_quality_checks(options_merged)
    
    print(f"\n✅ Successfully loaded and separated Bank Nifty options data!")
    print(f"📊 Datasets available:")
    print(f"   📞 Call Options (CE): {len(df_call):,} records")
    print(f"   📉 Put Options (PE): {len(df_put):,} records")
    print(f"   📋 Total Options: {len(options_merged):,} records")
    
    return df_call, df_put, options_merged


def _clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process options data.
    
    Args:
        df (pd.DataFrame): Raw options data
        
    Returns:
        pd.DataFrame: Cleaned options data
    """
    # Replace all "-" with np.nan in the entire dataframe
    df = df.replace('-', np.nan)
    
    # Define numeric columns that need type conversion
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'No. of contracts', 
                      'Turnover * in   ₹ Lakhs', 'Open Int', 'Change in OI', 'Strike Price']
    
    # Convert numeric columns to appropriate data types
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    
    # Convert Expiry column to datetime
    if 'Expiry' in df.columns:
        df['Expiry'] = pd.to_datetime(df['Expiry'], format='%d-%b-%Y')
    
    # Sort by Date and Expiry
    df = df.sort_values(['Date', 'Expiry']).reset_index(drop=True)
    
    return df


def _separate_call_put_options(all_options_data: List[pd.DataFrame], 
                              csv_files: List[str], 
                              options_merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate Call (CE) and Put (PE) options data.
    
    Args:
        all_options_data (List[pd.DataFrame]): List of individual dataframes
        csv_files (List[str]): List of CSV file paths
        options_merged (pd.DataFrame): Merged options data
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Call and Put options dataframes
    """
    # Define numeric columns for type conversion
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'No. of contracts', 
                      'Turnover * in   ₹ Lakhs', 'Open Int', 'Change in OI', 'Strike Price']
    
    # Initialize empty dataframes
    df_call = pd.DataFrame()
    df_put = pd.DataFrame()
    
    # Check if there's an option type column
    if 'Option_typ' in options_merged.columns:
        df_call = options_merged[options_merged['Option_typ'] == 'CE'].copy()
        df_put = options_merged[options_merged['Option_typ'] == 'PE'].copy()
    else:
        # Separate based on filename patterns (CE/PE)
        ce_data = []
        pe_data = []
        
        for file in csv_files:
            filename = os.path.basename(file).upper()
            if 'CE' in filename:
                ce_files = [df for df in all_options_data if df['Source_File'].iloc[0] == os.path.basename(file)]
                for df in ce_files:
                    # Replace "-" with np.nan in CE data
                    df_clean_ce = df.replace('-', np.nan)
                    ce_data.append(df_clean_ce)
            elif 'PE' in filename:
                pe_files = [df for df in all_options_data if df['Source_File'].iloc[0] == os.path.basename(file)]
                for df in pe_files:
                    # Replace "-" with np.nan in PE data
                    df_clean_pe = df.replace('-', np.nan)
                    pe_data.append(df_clean_pe)
        
        if ce_data:
            df_call = pd.concat(ce_data, ignore_index=True)
            df_call['Date'] = pd.to_datetime(df_call['Date'], format='%d-%b-%Y')
            df_call['Expiry'] = pd.to_datetime(df_call['Expiry'], format='%d-%b-%Y')
            # Convert numeric columns
            for col in numeric_columns:
                if col in df_call.columns:
                    df_call[col] = pd.to_numeric(df_call[col], errors='coerce')
            df_call = df_call.sort_values(['Date', 'Expiry']).reset_index(drop=True)
        
        if pe_data:
            df_put = pd.concat(pe_data, ignore_index=True)
            df_put['Date'] = pd.to_datetime(df_put['Date'], format='%d-%b-%Y')
            df_put['Expiry'] = pd.to_datetime(df_put['Expiry'], format='%d-%b-%Y')
            # Convert numeric columns
            for col in numeric_columns:
                if col in df_put.columns:
                    df_put[col] = pd.to_numeric(df_put[col], errors='coerce')
            df_put = df_put.sort_values(['Date', 'Expiry']).reset_index(drop=True)
    
    return df_call, df_put


def _display_data_summary(options_merged: pd.DataFrame, 
                         df_call: pd.DataFrame, 
                         df_put: pd.DataFrame) -> None:
    """
    Display summary information about the loaded data.
    
    Args:
        options_merged (pd.DataFrame): Merged options data
        df_call (pd.DataFrame): Call options data
        df_put (pd.DataFrame): Put options data
    """
    print(f"\n📊 MERGED DATASET SUMMARY:")
    print("-" * 30)
    print(f"📈 Total Options Records: {len(options_merged):,}")
    print(f"📞 Call Options (CE): {len(df_call):,}")
    print(f"📉 Put Options (PE): {len(df_put):,}")
    
    if not options_merged.empty:
        print(f"📅 Date Range: {options_merged['Date'].min().strftime('%d-%b-%Y')} to {options_merged['Date'].max().strftime('%d-%b-%Y')}")
        print(f"🎯 Expiry Range: {options_merged['Expiry'].min().strftime('%d-%b-%Y')} to {options_merged['Expiry'].max().strftime('%d-%b-%Y')}")
        print(f"📋 Unique Expiries: {options_merged['Expiry'].nunique()}")
        print(f"📄 Source Files: {options_merged['Source_File'].nunique()}")
        
        # Display column information
        print(f"\n📋 DATASET COLUMNS:")
        print("-" * 20)
        for i, col in enumerate(options_merged.columns, 1):
            print(f"  {i:2d}. {col}")


def _perform_data_quality_checks(options_merged: pd.DataFrame) -> None:
    """
    Perform data quality checks on the merged data.
    
    Args:
        options_merged (pd.DataFrame): Merged options data
    """
    print(f"\n🔍 DATA QUALITY CHECK:")
    print("-" * 25)
    
    # Check for missing values in merged data
    missing_data = options_merged.isnull().sum()
    if missing_data.sum() > 0:
        print("✅ Missing Values Summary (after replacing '-' with NaN):")
        for col, count in missing_data[missing_data > 0].items():
            percentage = (count / len(options_merged)) * 100
            print(f"   {col}: {count:,} missing values ({percentage:.1f}%)")
    else:
        print("✅ No missing values found after data cleaning")
    
    # Check for duplicate records
    if not options_merged.empty:
        duplicates = options_merged.duplicated(['Date', 'Expiry']).sum()
        if duplicates > 0:
            print(f"⚠️ Found {duplicates} potential duplicate records")
        else:
            print("✅ No duplicate records found")
        
        # Display expiry breakdown
        print(f"\n📊 EXPIRY BREAKDOWN:")
        print("-" * 20)
        expiry_counts = options_merged['Expiry'].value_counts().sort_index()
        for expiry, count in expiry_counts.items():
            print(f"  {expiry.strftime('%d-%b-%Y')}: {count:3d} records")


def get_data_summary(df_call: pd.DataFrame, df_put: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary dictionary of the loaded data.
    
    Args:
        df_call (pd.DataFrame): Call options data
        df_put (pd.DataFrame): Put options data
        
    Returns:
        Dict[str, any]: Dictionary containing data summary statistics
    """
    summary = {
        'total_call_records': len(df_call),
        'total_put_records': len(df_put),
        'total_records': len(df_call) + len(df_put),
        'call_columns': list(df_call.columns) if not df_call.empty else [],
        'put_columns': list(df_put.columns) if not df_put.empty else [],
        'date_range': {
            'start': None,
            'end': None
        },
        'expiry_range': {
            'start': None,
            'end': None
        }
    }
    
    # Add date ranges if data is available
    if not df_call.empty and 'Date' in df_call.columns:
        summary['date_range']['start'] = df_call['Date'].min()
        summary['date_range']['end'] = df_call['Date'].max()
        
    if not df_put.empty and 'Date' in df_put.columns:
        if summary['date_range']['start'] is None:
            summary['date_range']['start'] = df_put['Date'].min()
            summary['date_range']['end'] = df_put['Date'].max()
        else:
            summary['date_range']['start'] = min(summary['date_range']['start'], df_put['Date'].min())
            summary['date_range']['end'] = max(summary['date_range']['end'], df_put['Date'].max())
    
    return summary
