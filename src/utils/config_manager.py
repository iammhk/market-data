"""
Configuration Manager
Module for managing application configuration and settings.
"""

import os
import configparser
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigManager:
    """Class for managing application configuration."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to config directory in project root
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / 'config'
        
        self.config_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        env_file = self.config_dir.parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
        
        self.config = configparser.ConfigParser()
        self.config_file = self.config_dir / 'config.ini'
        
        # Load or create default configuration
        self._load_or_create_config()
    
    def _load_or_create_config(self):
        """Load existing configuration or create default one."""
        if self.config_file.exists():
            self.config.read(self.config_file)
            logger.info(f"Loaded configuration from {self.config_file}")
        else:
            self._create_default_config()
            self.save_config()
            logger.info(f"Created default configuration at {self.config_file}")
    
    def _create_default_config(self):
        """Create default configuration."""
        # NSE Configuration
        self.config['NSE'] = {
            'base_url': 'https://www.nseindia.com',
            'timeout': '10',
            'retry_attempts': '3',
            'rate_limit_delay': '1'
        }
        
        # Zerodha Configuration
        self.config['ZERODHA'] = {
            'api_key': '',  # To be filled by user
            'access_token': '',  # To be filled by user
            'sandbox': 'true'
        }
        
        # Data Storage Configuration
        self.config['DATA'] = {
            'data_directory': './data',
            'database_file': 'market_data.db',
            'csv_export': 'true',
            'json_export': 'false'
        }
        
        # Logging Configuration
        self.config['LOGGING'] = {
            'level': 'INFO',
            'log_file': 'market_data.log',
            'max_file_size': '10485760',  # 10MB
            'backup_count': '5'
        }
        
        # API Rate Limits
        self.config['RATE_LIMITS'] = {
            'nse_requests_per_minute': '30',
            'yahoo_requests_per_minute': '60',
            'zerodha_requests_per_second': '10'
        }
        
        # Default Stock Lists
        self.config['WATCHLISTS'] = {
            'nifty_50': 'true',
            'bank_nifty': 'true',
            'nifty_it': 'false',
            'custom_symbols': ''
        }
    
    def get(self, section: str, key: str, fallback: Any = None) -> str:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            fallback: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return self.config.get(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            logger.warning(f"Configuration key {section}.{key} not found, using fallback: {fallback}")
            return fallback
    
    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get integer configuration value."""
        try:
            return self.config.getint(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            logger.warning(f"Configuration key {section}.{key} not found or invalid, using fallback: {fallback}")
            return fallback
    
    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get float configuration value."""
        try:
            return self.config.getfloat(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            logger.warning(f"Configuration key {section}.{key} not found or invalid, using fallback: {fallback}")
            return fallback
    
    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean configuration value."""
        try:
            return self.config.getboolean(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            logger.warning(f"Configuration key {section}.{key} not found or invalid, using fallback: {fallback}")
            return fallback
    
    def set(self, section: str, key: str, value: str):
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get_zerodha_credentials(self) -> Dict[str, str]:
        """Get Zerodha API credentials."""
        return {
            'api_key': os.getenv('ZERODHA_API_KEY') or self.get('ZERODHA', 'api_key', ''),
            'access_token': os.getenv('ZERODHA_ACCESS_TOKEN') or self.get('ZERODHA', 'access_token', '')
        }
    
    def get_data_directory(self) -> Path:
        """Get data directory path."""
        data_dir = Path(self.get('DATA', 'data_directory', './data'))
        data_dir.mkdir(exist_ok=True)
        return data_dir
    
    def get_database_path(self) -> Path:
        """Get database file path."""
        return self.get_data_directory() / self.get('DATA', 'database_file', 'market_data.db')
    
    def get_watchlist_symbols(self) -> Dict[str, bool]:
        """Get watchlist configuration."""
        return {
            'nifty_50': self.getboolean('WATCHLISTS', 'nifty_50', True),
            'bank_nifty': self.getboolean('WATCHLISTS', 'bank_nifty', True),
            'nifty_it': self.getboolean('WATCHLISTS', 'nifty_it', False)
        }
    
    def get_custom_symbols(self) -> list:
        """Get custom symbols list."""
        symbols_str = self.get('WATCHLISTS', 'custom_symbols', '')
        if symbols_str:
            return [symbol.strip() for symbol in symbols_str.split(',') if symbol.strip()]
        return []
    
    def update_zerodha_credentials(self, api_key: str, access_token: str):
        """Update Zerodha credentials."""
        self.set('ZERODHA', 'api_key', api_key)
        self.set('ZERODHA', 'access_token', access_token)
        self.save_config()
        logger.info("Zerodha credentials updated")
    
    def get_all_config(self) -> Dict[str, Dict[str, str]]:
        """Get all configuration as dictionary."""
        config_dict = {}
        for section in self.config.sections():
            config_dict[section] = dict(self.config.items(section))
        return config_dict
    
    def export_config_json(self, file_path: Optional[str] = None) -> str:
        """
        Export configuration to JSON file.
        
        Args:
            file_path: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        if not file_path:
            file_path = self.config_dir / 'config.json'
        
        config_dict = self.get_all_config()
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration exported to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error exporting configuration: {str(e)}")
            return ""
