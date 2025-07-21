import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Union, Dict

# Set up logger
logger = logging.getLogger(__name__)

class MT5ConnectionManager:
    """
    Handles all MetaTrader 5 connection management, including initialization,
    reconnection, and data retrieval.
    """
    def __init__(self, config):
        """
        Initialize the MT5 connection manager.
        
        Args:
            config: Configuration object with MT5 and LiveTrader sections
        """
        self.config = config
        self._mt5_initialized = False
        self._last_connection_check = 0
        self._connection_check_interval = 300  # 5 minutes between connection checks
        
        # Get retry settings from LiveTrader section
        self._max_connection_attempts = int(config.get('LiveTrader', 'max_retries', fallback=5))
        self._connection_retry_delay = int(config.get('LiveTrader', 'retry_delay', fallback=3))
        self._min_reconnect_interval = 60  # Minimum seconds between reconnection attempts
        self._last_connection_attempt = 0
        
        # Timeframe mapping
        self._tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M3': mt5.TIMEFRAME_M3,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }
    
    def check_connection(self) -> bool:
        """
        Check if MT5 terminal is connected and account is authorized.
        
        Returns:
            bool: True if connected and authorized, False otherwise
        """
        try:
            # Check if terminal is connected and account is authorized
            if not mt5.terminal_info() is None and mt5.account_info() is not None:
                self._mt5_initialized = True
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking MT5 connection: {e}")
            self._mt5_initialized = False
            return False

    def initialize(self) -> bool:
        """
        Initialize connection to MT5 terminal with retry logic.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        current_time = time.time()
        
        # Check if we recently attempted to connect
        if current_time - self._last_connection_attempt < self._min_reconnect_interval:
            logger.debug("Skipping MT5 initialization - too soon since last attempt")
            return self._mt5_initialized
            
        self._last_connection_attempt = current_time
        
        # Use get_section instead of has_section
        mt5_config = self.config.get_section('MT5')
        
        # Initialize MT5 with config parameters
        for attempt in range(1, self._max_connection_attempts + 1):
            try:
                # Shutdown any existing connection first
                if mt5.terminal_info() is not None:
                    mt5.shutdown()
                
                # Initialize with config parameters if available
                if mt5_config.get('executable_path'):
                    mt5.initialize(
                        path=mt5_config.get('executable_path'),
                        portable=mt5_config.get('portable', 'false').lower() == 'true',
                        timeout=int(mt5_config.get('timeout', '60000')),
                        login=int(mt5_config.get('login', 0)),
                        password=mt5_config.get('password', ''),
                        server=mt5_config.get('server', '')
                    )
                else:
                    mt5.initialize()
                
                # Verify connection
                if self.check_connection():
                    logger.info("✅ MT5 connection established successfully")
                    return True
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"MT5 initialization attempt {attempt}/{self._max_connection_attempts} failed: {error_msg}")
                if attempt < self._max_connection_attempts:
                    time.sleep(self._connection_retry_delay)
                
        logger.error("Failed to initialize MT5 after multiple attempts")
        return False

    def ensure_connected(self) -> bool:
        """
        Ensure MT5 terminal is connected and authorized.
        Performs periodic connection checks and automatic reconnection if needed.
        
        Returns:
            bool: True if connected, False otherwise
        """
        current_time = time.time()
        
        # Check if we need to verify the connection
        if (self._mt5_initialized and 
            current_time - self._last_connection_check < self._connection_check_interval):
            return True
            
        self._last_connection_check = current_time
        
        # Check if already connected
        if self.check_connection():
            return True
            
        # Try to reconnect
        logger.warning("MT5 connection lost. Attempting to reconnect...")
        return self.initialize()
    
    def select_symbol(self, symbol: str) -> bool:
        """
        Ensure symbol is selected in Market Watch.
        
        Args:
            symbol: Symbol to select
            
        Returns:
            bool: True if symbol was selected successfully, False otherwise
        """
        try:
            if not self.ensure_connected():
                logger.error(f"Cannot select symbol {symbol}: MT5 not connected")
                return False
                
            return mt5.symbol_select(symbol, True)
        except Exception as e:
            logger.error(f"Error selecting symbol {symbol}: {e}")
            return False
    
    def get_data(self, symbol: str, timeframe: str, bars: int = 1000, 
               from_date: Optional[Union[str, pd.Timestamp, datetime]] = None,
               to_date: Optional[Union[str, pd.Timestamp, datetime]] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from MetaTrader 5 with connection management.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Timeframe string (e.g., 'M1', 'H1', 'D1')
            bars: Number of bars to fetch (used if date range not specified)
            from_date: Start date for data (optional)
            to_date: End date for data (optional)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Ensure MT5 is connected
        if not self.ensure_connected():
            logger.error("Cannot fetch data: MT5 not connected")
            return None
            
        # Ensure symbol is selected
        if not self.select_symbol(symbol):
            logger.error(f"Cannot fetch data: Symbol {symbol} not available")
            return None
            
        # Convert timeframe string to MT5 timeframe constant
        if timeframe not in self._tf_map:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None
            
        try:
            # Convert timeframe string to MT5 timeframe enum
            tf = self._tf_map[timeframe]
            
            # Prepare request parameters
            if bars is not None:
                logger.debug(f"Fetching last {bars} bars for {symbol} {timeframe}")
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            elif from_date is not None and to_date is not None:
                logger.debug(f"Fetching data for {symbol} {timeframe} from {from_date} to {to_date}")
                rates = mt5.copy_rates_range(symbol, tf, from_date, to_date)
            else:
                logger.error("Either bars or date range must be specified")
                return None
                
            if rates is None or len(rates) == 0:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Standardize column names
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            logger.debug(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching MT5 data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information.
        
        Returns:
            Dict with account information or None if failed
        """
        if not self.ensure_connected():
            logger.error("Cannot get account info: MT5 not connected")
            return None
            
        try:
            account_info = mt5.account_info()
            if account_info is not None:
                # Convert to dictionary
                return {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.margin_free,
                    'leverage': account_info.leverage,
                    'name': account_info.name,
                    'server': account_info.server,
                    'currency': account_info.currency,
                    'company': account_info.company,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def shutdown(self):
        """
        Shut down the MT5 connection and clean up resources.
        """
        try:
            if mt5.terminal_info() is not None:
                mt5.shutdown()
                self._mt5_initialized = False
                logger.info("MT5 connection terminated")
        except Exception as e:
            logger.error(f"Error shutting down MT5: {e}")
