import configparser
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path: str = 'config.ini'):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        # If only filename is provided, use the current working directory for portability
        if config_path == 'config.ini' or not os.path.dirname(config_path):
            # Use current working directory instead of script location
            config_manager_dir = os.getcwd()
            config_path = os.path.join(config_manager_dir, 'config.ini')

        # Store absolute path to config file and its directory
        self.config_path = os.path.abspath(config_path)
        self.config_dir = os.path.dirname(self.config_path)
        #self.config = configparser.ConfigParser()
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())


        # Set default values
        self.defaults = {
            'Strategy': {
                # Moving Averages
                'sma_fast_period': '200',
                'sma_slow_period': '21',
                # ADX
                'adx_period': '14',
                'adx_threshold': '20',
                # RSI
                'rsi_period': '14',
                'rsi_ma_period': '9',
                # ATR
                'atr_period': '14',
                # SuperTrend
                'supertrend_period': '10',
                'supertrend_long_multiplier': '1.5',
                'supertrend_short_multiplier': '1.0',
                # ADR
                'adr_period': '14',
                # Risk Management
                'risk_per_trade': '0.01',
                'use_trailing_stop': 'true',
                'atr_trailing_multiplier': '2.0',
                # Trading Hours
                'start_hour': '0',
                'end_hour': '24',
                # Symbol Settings
                'symbol': 'XAUUSD',
                'timeframe': 'M1'
            },
            'Backtest': {
                'initial_cash': '10000.0',
                'commission': '0.0005',
                'max_trades': '1000'
            },
            'MT5': {
                'server': 'MetaQuotes-Demo',
                'login': '',
                'password': '',
                'executable_path': 'C:/MT5/Master/terminal64.exe'
            },
            'Database': {
                'path': 'mt5_data.db'
            }
        }
        
        # Create default config if it doesn't exist
        if not os.path.exists(self.config_path):
            self._create_default_config()
        
        self.load_config()
    
    def _create_default_config(self):
        """Create a default configuration file if it doesn't exist."""
        self.config.read_dict(self.defaults)
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)
    
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.config.read(self.config_path)
            return True
        except Exception as e:
            print(f"Error loading config file: {e}")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.

        Returns:
            bool: True if successful, False otherwise
        """
        # COMMENTED OUT FOR REPLAY MODE - NO CONFIG SAVING ALLOWED
        print("Config save disabled for replay mode")
        return True
        # try:
        #     with open(self.config_path, 'w') as configfile:
        #         self.config.write(configfile)
        #     return True
        # except Exception as e:
        #     print(f"Error saving config file: {e}")
        #     return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a section from the configuration.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary with section key-value pairs
        """
        try:
            return dict(self.config[section])
        except KeyError:
            return {}
    
    def get(self, section: str, option: str, fallback: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            option: Configuration option
            fallback: Fallback value if option is not found
            
        Returns:
            The configuration value or fallback
            
        Note:
            For Database.path, if the path is relative, it will be resolved relative to the config file.
        """
        try:
            value = self.config.get(section, option, fallback=fallback)
            
            # Handle database path resolution
            if section.lower() == 'database' and option.lower() == 'path' and value:
                # If path is not absolute, make it relative to the config file
                if not os.path.isabs(value):
                    value = os.path.join(self.config_dir, value)
                return os.path.normpath(value)
                
            # Convert string 'true'/'false' to boolean
            if isinstance(value, str) and value.lower() in ('true', 'false'):
                return value.lower() == 'true'
                
            return value
            
        except (configparser.NoSectionError, configparser.NoOptionError):
            # If this is a database path request, ensure we return the default path relative to config
            if section.lower() == 'database' and option.lower() == 'path' and fallback:
                return os.path.normpath(os.path.join(self.config_dir, str(fallback)))
            return fallback
    
    def getint(self, section: str, key: str, default: int = 0) -> int:
        """Get a value as integer."""
        try:
            return self.config.getint(section, key)
        except (ValueError, configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def getfloat(self, section: str, key: str, default: float = 0.0) -> float:
        """Get a value as float."""
        try:
            return self.config.getfloat(section, key)
        except (ValueError, configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def getboolean(self, section: str, key: str, default: bool = False) -> bool:
        """Get a value as boolean."""
        try:
            return self.config.getboolean(section, key)
        except (ValueError, configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a value in the configuration.

        Args:
            section: Section name
            key: Key name
            value: Value to set
        """
        # COMMENTED OUT FOR REPLAY MODE - NO CONFIG MODIFICATIONS ALLOWED
        print(f"Config set disabled for replay mode: {section}.{key} = {value}")
        return
        # if not self.config.has_section(section):
        #     self.config.add_section(section)
        # self.config.set(section, key, str(value))

# Global config instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """
    Get the global config manager instance.
    
    Returns:
        ConfigManager instance
    """
    return config_manager
