"""
Data Manager
Module for managing market data storage, retrieval, and analysis.
"""

import sqlite3
import pandas as pd
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class DataManager:
    """Class for managing market data storage and retrieval."""
    
    def __init__(self, data_directory: Optional[str] = None, database_file: Optional[str] = None):
        """
        Initialize data manager.
        
        Args:
            data_directory: Directory for storing data files
            database_file: SQLite database file name
        """
        if data_directory:
            self.data_dir = Path(data_directory)
        else:
            # Default to data directory in project root
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / 'data'
        
        self.data_dir.mkdir(exist_ok=True)
        
        # Database setup
        db_name = database_file or 'market_data.db'
        self.db_path = self.data_dir / db_name
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create stock_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stock_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        source TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date, source)
                    )
                ''')
                
                # Create indices_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS indices_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        index_name TEXT NOT NULL,
                        date DATE NOT NULL,
                        value REAL NOT NULL,
                        change_points REAL,
                        change_percent REAL,
                        source TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(index_name, date, source)
                    )
                ''')
                
                # Create news_data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        summary TEXT,
                        url TEXT,
                        published_date DATETIME,
                        source TEXT,
                        symbols TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create watchlist table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS watchlist (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL UNIQUE,
                        name TEXT,
                        exchange TEXT,
                        sector TEXT,
                        added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def store_stock_data(self, data: Union[pd.DataFrame, Dict[str, Any]], source: str = "unknown"):
        """
        Store stock data in database.
        
        Args:
            data: Stock data as DataFrame or dictionary
            source: Data source identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if isinstance(data, pd.DataFrame):
                    # Handle DataFrame
                    df = data.copy()
                    df['source'] = source
                    df['timestamp'] = datetime.now()
                    
                    # Reset index to get date as column if it's in index
                    if df.index.name == 'date' or 'Date' in str(df.index.names):
                        df = df.reset_index()
                    
                    # Ensure required columns exist
                    required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
                    df_columns = df.columns.str.lower()
                    
                    # Map common column names
                    column_mapping = {
                        'adj close': 'close',
                        'adjusted close': 'close'
                    }
                    
                    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df_columns})
                    df.columns = df.columns.str.lower()
                    
                    # Select only available columns
                    available_columns = [col for col in required_columns if col in df.columns]
                    if 'symbol' not in df.columns:
                        logger.warning("Symbol column not found in DataFrame")
                        return
                    
                    df_subset = df[available_columns + ['source', 'timestamp']]
                    
                    # Insert data
                    df_subset.to_sql('stock_data', conn, if_exists='append', index=False)
                    logger.info(f"Stored {len(df_subset)} stock data records from {source}")
                
                elif isinstance(data, dict):
                    # Handle dictionary data
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO stock_data 
                        (symbol, date, open, high, low, close, volume, source, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data.get('symbol', ''),
                        data.get('date', datetime.now().date()),
                        data.get('open'),
                        data.get('high'),
                        data.get('low'),
                        data.get('close'),
                        data.get('volume'),
                        source,
                        datetime.now()
                    ))
                    conn.commit()
                    logger.info(f"Stored single stock data record for {data.get('symbol', 'unknown')}")
                
        except Exception as e:
            logger.error(f"Error storing stock data: {str(e)}")
    
    def store_index_data(self, index_name: str, value: float, change_points: float = None, 
                        change_percent: float = None, source: str = "unknown"):
        """
        Store index data in database.
        
        Args:
            index_name: Name of the index
            value: Current value
            change_points: Change in points
            change_percent: Change in percentage
            source: Data source
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO indices_data 
                    (index_name, date, value, change_points, change_percent, source, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    index_name,
                    datetime.now().date(),
                    value,
                    change_points,
                    change_percent,
                    source,
                    datetime.now()
                ))
                conn.commit()
                logger.info(f"Stored index data for {index_name}")
                
        except Exception as e:
            logger.error(f"Error storing index data: {str(e)}")
    
    def get_stock_data(self, symbol: str, days: int = 30, source: str = None) -> Optional[pd.DataFrame]:
        """
        Retrieve stock data from database.
        
        Args:
            symbol: Stock symbol
            days: Number of days to retrieve
            source: Filter by data source (optional)
            
        Returns:
            DataFrame with stock data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM stock_data 
                    WHERE symbol = ? AND date >= date('now', '-{} days')
                '''.format(days)
                
                params = [symbol]
                
                if source:
                    query += ' AND source = ?'
                    params.append(source)
                
                query += ' ORDER BY date DESC'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    logger.info(f"Retrieved {len(df)} records for {symbol}")
                    return df
                else:
                    logger.warning(f"No data found for {symbol}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving stock data for {symbol}: {str(e)}")
            return None
    
    def get_index_data(self, index_name: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Retrieve index data from database.
        
        Args:
            index_name: Index name
            days: Number of days to retrieve
            
        Returns:
            DataFrame with index data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM indices_data 
                    WHERE index_name = ? AND date >= date('now', '-{} days')
                    ORDER BY date DESC
                '''.format(days)
                
                df = pd.read_sql_query(query, conn, params=[index_name])
                
                if not df.empty:
                    logger.info(f"Retrieved {len(df)} records for {index_name}")
                    return df
                else:
                    logger.warning(f"No data found for {index_name}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving index data for {index_name}: {str(e)}")
            return None
    
    def export_to_csv(self, symbol: str, filename: Optional[str] = None) -> str:
        """
        Export stock data to CSV file.
        
        Args:
            symbol: Stock symbol
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        try:
            data = self.get_stock_data(symbol)
            if data is None or data.empty:
                logger.warning(f"No data available for export: {symbol}")
                return ""
            
            if not filename:
                filename = f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv"
            
            file_path = self.data_dir / filename
            data.to_csv(file_path, index=False)
            
            logger.info(f"Data exported to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {str(e)}")
            return ""
    
    def add_to_watchlist(self, symbol: str, name: str = "", exchange: str = "", sector: str = ""):
        """
        Add symbol to watchlist.
        
        Args:
            symbol: Stock symbol
            name: Company name
            exchange: Exchange name
            sector: Sector name
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO watchlist (symbol, name, exchange, sector, added_date, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (symbol, name, exchange, sector, datetime.now(), True))
                conn.commit()
                logger.info(f"Added {symbol} to watchlist")
                
        except Exception as e:
            logger.error(f"Error adding {symbol} to watchlist: {str(e)}")
    
    def get_watchlist(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get watchlist symbols.
        
        Args:
            active_only: Return only active symbols
            
        Returns:
            List of watchlist entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM watchlist'
                if active_only:
                    query += ' WHERE is_active = TRUE'
                query += ' ORDER BY added_date DESC'
                
                cursor = conn.cursor()
                cursor.execute(query)
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                watchlist = [dict(zip(columns, row)) for row in rows]
                logger.info(f"Retrieved {len(watchlist)} watchlist entries")
                
                return watchlist
                
        except Exception as e:
            logger.error(f"Error retrieving watchlist: {str(e)}")
            return []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of stored data.
        
        Returns:
            Dictionary with data summary statistics
        """
        summary = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Stock data summary
                cursor.execute('SELECT COUNT(*) as total_records FROM stock_data')
                summary['total_stock_records'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT symbol) as unique_symbols FROM stock_data')
                summary['unique_symbols'] = cursor.fetchone()[0]
                
                # Index data summary
                cursor.execute('SELECT COUNT(*) as total_records FROM indices_data')
                summary['total_index_records'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT index_name) as unique_indices FROM indices_data')
                summary['unique_indices'] = cursor.fetchone()[0]
                
                # Watchlist summary
                cursor.execute('SELECT COUNT(*) as watchlist_count FROM watchlist WHERE is_active = TRUE')
                summary['active_watchlist_count'] = cursor.fetchone()[0]
                
                # Latest data dates
                cursor.execute('SELECT MAX(date) as latest_stock_date FROM stock_data')
                result = cursor.fetchone()
                summary['latest_stock_date'] = result[0] if result[0] else 'No data'
                
                cursor.execute('SELECT MAX(date) as latest_index_date FROM indices_data')
                result = cursor.fetchone()
                summary['latest_index_date'] = result[0] if result[0] else 'No data'
                
                logger.info("Generated data summary")
                
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
        
        return summary
