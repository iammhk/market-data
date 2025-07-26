"""
Zerodha KiteConnect API Fetcher
Module for fetching data using Zerodha's KiteConnect API.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ZerodhaFetcher:
    """Class to fetch data from Zerodha KiteConnect API."""
    
    def __init__(self, api_key: Optional[str] = None, access_token: Optional[str] = None):
        """
        Initialize Zerodha Fetcher.
        
        Args:
            api_key: Zerodha API key
            access_token: Access token for authentication
        """
        self.api_key = api_key
        self.access_token = access_token
        self.kite = None
        
        if api_key and access_token:
            self._initialize_kite_connect()
    
    def _initialize_kite_connect(self):
        """Initialize KiteConnect instance."""
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            logger.info("KiteConnect initialized successfully")
        except ImportError:
            logger.error("kiteconnect package not installed. Run: pip install kiteconnect")
        except Exception as e:
            logger.error(f"Failed to initialize KiteConnect: {str(e)}")
    
    def set_credentials(self, api_key: str, access_token: str):
        """
        Set API credentials.
        
        Args:
            api_key: Zerodha API key
            access_token: Access token
        """
        self.api_key = api_key
        self.access_token = access_token
        self._initialize_kite_connect()
    
    def get_instruments(self, exchange: str = "NSE") -> Optional[List[Dict[str, Any]]]:
        """
        Get list of instruments from specified exchange.
        
        Args:
            exchange: Exchange name (NSE, BSE, etc.)
            
        Returns:
            List of instruments or None if failed
        """
        if not self.kite:
            logger.error("KiteConnect not initialized. Please set credentials first.")
            return None
        
        try:
            instruments = self.kite.instruments(exchange)
            logger.info(f"Successfully fetched {len(instruments)} instruments from {exchange}")
            return instruments
        except Exception as e:
            logger.error(f"Error fetching instruments from {exchange}: {str(e)}")
            return None
    
    def get_quote(self, instruments: List[str]) -> Optional[Dict[str, Any]]:
        """
        Get quote for specified instruments.
        
        Args:
            instruments: List of instrument tokens or symbols
            
        Returns:
            Dictionary containing quote data or None if failed
        """
        if not self.kite:
            logger.error("KiteConnect not initialized. Please set credentials first.")
            return None
        
        try:
            quotes = self.kite.quote(instruments)
            logger.info(f"Successfully fetched quotes for {len(instruments)} instruments")
            return quotes
        except Exception as e:
            logger.error(f"Error fetching quotes: {str(e)}")
            return None
    
    def get_historical_data(self, instrument_token: str, from_date: datetime, 
                          to_date: datetime, interval: str = "day") -> Optional[pd.DataFrame]:
        """
        Get historical data for an instrument.
        
        Args:
            instrument_token: Instrument token
            from_date: Start date
            to_date: End date
            interval: Data interval (minute, day, etc.)
            
        Returns:
            DataFrame with historical data or None if failed
        """
        if not self.kite:
            logger.error("KiteConnect not initialized. Please set credentials first.")
            return None
        
        try:
            historical_data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if historical_data:
                df = pd.DataFrame(historical_data)
                logger.info(f"Successfully fetched {len(df)} records of historical data")
                return df
            else:
                logger.warning("No historical data returned")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None
    
    def get_positions(self) -> Optional[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            Dictionary containing position data or None if failed
        """
        if not self.kite:
            logger.error("KiteConnect not initialized. Please set credentials first.")
            return None
        
        try:
            positions = self.kite.positions()
            logger.info("Successfully fetched positions")
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            return None
    
    def get_holdings(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get current holdings.
        
        Returns:
            List of holdings or None if failed
        """
        if not self.kite:
            logger.error("KiteConnect not initialized. Please set credentials first.")
            return None
        
        try:
            holdings = self.kite.holdings()
            logger.info(f"Successfully fetched {len(holdings)} holdings")
            return holdings
        except Exception as e:
            logger.error(f"Error fetching holdings: {str(e)}")
            return None
    
    def search_instruments(self, query: str, exchange: str = "NSE") -> List[Dict[str, Any]]:
        """
        Search for instruments by name or symbol.
        
        Args:
            query: Search query
            exchange: Exchange to search in
            
        Returns:
            List of matching instruments
        """
        if not self.kite:
            logger.error("KiteConnect not initialized. Please set credentials first.")
            return []
        
        try:
            instruments = self.get_instruments(exchange)
            if not instruments:
                return []
            
            # Search for instruments matching the query
            matching_instruments = []
            query_lower = query.lower()
            
            for instrument in instruments:
                if (query_lower in instrument.get('tradingsymbol', '').lower() or 
                    query_lower in instrument.get('name', '').lower()):
                    matching_instruments.append(instrument)
            
            logger.info(f"Found {len(matching_instruments)} instruments matching '{query}'")
            return matching_instruments
            
        except Exception as e:
            logger.error(f"Error searching instruments: {str(e)}")
            return []
    
    def place_order(self, tradingsymbol: str, quantity: int, transaction_type: str,
                   order_type: str = "MARKET", product: str = "MIS", **kwargs) -> Optional[str]:
        """
        Place an order (for demo purposes - use with caution in live trading).
        
        Args:
            tradingsymbol: Trading symbol
            quantity: Order quantity
            transaction_type: BUY or SELL
            order_type: Order type (MARKET, LIMIT, etc.)
            product: Product type (MIS, CNC, NRML)
            **kwargs: Additional order parameters
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self.kite:
            logger.error("KiteConnect not initialized. Please set credentials first.")
            return None
        
        try:
            order_params = {
                'tradingsymbol': tradingsymbol,
                'quantity': quantity,
                'transaction_type': transaction_type,
                'order_type': order_type,
                'product': product,
                **kwargs
            }
            
            order_id = self.kite.place_order(**order_params)
            logger.info(f"Order placed successfully. Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def get_orders(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get list of orders.
        
        Returns:
            List of orders or None if failed
        """
        if not self.kite:
            logger.error("KiteConnect not initialized. Please set credentials first.")
            return None
        
        try:
            orders = self.kite.orders()
            logger.info(f"Successfully fetched {len(orders)} orders")
            return orders
        except Exception as e:
            logger.error(f"Error fetching orders: {str(e)}")
            return None
