import math
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import MetaTrader5 as mt5
# Use absolute import for better compatibility when importing from outside the package
from mt5_manager import MT5ConnectionManager
from datetime import datetime, timedelta, timezone
import pytz
import json
import configparser
from pathlib import Path
from tabulate import tabulate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import time
import logging
import sys
import os
    
def save_internal_sl(magic, stop_loss, filename="internal_sls.json"):
    import os
    print(f"[DEBUG] Saving SL: magic={magic}, stop_loss={stop_loss}, file={os.path.abspath(filename)}")
    data = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    data[str(magic)] = stop_loss
    try:
        with open(filename, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[ERROR] Could not write to {filename}: {e}")

def load_internal_sl(magic, filename="internal_sls.json"):
    """Load the internal SL for a position from a local JSON file."""
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
    return data.get(str(magic))

class UnicodeStreamHandler(logging.StreamHandler):
    """
    A handler class that writes logging records to a stream with proper Unicode encoding.
    This ensures that Unicode characters are properly displayed in the console.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Import config manager
from config_manager import get_config

# Import trade logger
from trade_logger import trade_logger

# Get config instance
config = get_config()

import os

# Create logs directory if it doesn't exist
# Use current working directory instead of script location for portability
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Set up log file path
log_file = os.path.join(log_dir, 'strategy.log')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        UnicodeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"

@dataclass
class TradePosition:
    """Enhanced trade position with magic number tracking and HIDDEN stop loss"""
    magic_number: int
    direction: TradeDirection
    entry_price: float
    stop_loss: float  # HIDDEN/DYNAMIC stop loss - NOT placed as actual order in system
    take_profit: float
    timestamp: datetime
    symbol: str
    position_size: float
    trailing_stop: bool = False
    atr_trailing_multiplier: float = 0.0
    highest_high: float = 0.0
    lowest_low: float = 0.0
    exit_price: float = None
    exit_time: datetime = None
    exit_reason: str = None
    is_closed: bool = False
    hidden_stop_loss: bool = True  # Flag indicating this is a hidden stop (not placed in system)
    trailing_activated: bool = False  # Add this field

    def __post_init__(self):
        """Initialize position tracking values"""
        if self.direction == TradeDirection.LONG:
            self.highest_high = self.entry_price
            self.lowest_low = float('inf')
        else:  # SHORT
            self.lowest_low = self.entry_price
            self.highest_high = 0.0
    
    def should_close(self, current_bar: pd.Series) -> bool:
        """
        Check if the position should be closed based on current price action.
        For stop loss: uses bar's LOW (LONG) or HIGH (SHORT) to catch intrabar breaches.
        For take profit: uses bar's CLOSE as before.
        For SuperTrend exit: checks if Prev Close crosses SuperTrend line.
        
        IMPORTANT: 
        1. Skips exit checking on the same candle where the position was opened
        2. Prioritizes the tighter stop loss (SL_HIT) over SuperTrend Exit when SuperTrend would result in a larger loss
        """
        # Skip exit checking on the same candle where the position was opened
        # This prevents unrealistic scenarios where a trade is entered and immediately hit SL/TP
        if hasattr(current_bar, 'name') and current_bar.name == self.timestamp:
            return False
        
        # Initialize variables to track potential exit conditions
        supertrend_exit = False
        supertrend_exit_price = None
        sl_exit = False
        sl_exit_price = None
        tp_exit = False
        tp_exit_price = None
        
        # Check SuperTrend exit condition if the data is available
        if 'st_long' in current_bar and 'st_short' in current_bar and hasattr(current_bar, 'name'):
            prev_close = current_bar.get('prev_close', None)
            
            # If prev_close is not available, try to use the previous bar's close
            if prev_close is None and hasattr(current_bar, 'prev_close'):
                prev_close = current_bar.prev_close
                
            if prev_close is not None:
                if self.direction == TradeDirection.LONG and prev_close < current_bar['st_long']:
                    # For LONG positions: exit if previous close crosses below st_long
                    supertrend_exit = True
                    supertrend_exit_price = current_bar['Close']
                elif self.direction == TradeDirection.SHORT and prev_close > current_bar['st_long']:
                    # For SHORT positions: exit if previous close crosses above st_long
                    supertrend_exit = True
                    supertrend_exit_price = current_bar['Close']
        
        # Check stop loss and take profit conditions
        if self.direction == TradeDirection.LONG:
            # Check if bar's LOW breaches stop loss
            if current_bar['Low'] <= self.stop_loss:
                sl_exit = True
                sl_exit_price = self.stop_loss
            # Check if CLOSE breaches take profit
            if current_bar['Close'] >= self.take_profit:
                tp_exit = True
                tp_exit_price = current_bar['Close']
        elif self.direction == TradeDirection.SHORT:
            # Check if bar's HIGH breaches stop loss
            if current_bar['High'] >= self.stop_loss:
                sl_exit = True
                sl_exit_price = self.stop_loss
            # Check if CLOSE breaches take profit
            if current_bar['Close'] <= self.take_profit:
                tp_exit = True
                tp_exit_price = current_bar['Close']
        
        # Prioritize exits based on profitability
        if tp_exit:
            # Take profit has highest priority
            self.exit_price = tp_exit_price
            self.exit_reason = XAUUSDStrategy.EXIT_TP_HIT
            return True
        elif sl_exit and supertrend_exit:
            # Both stop loss and supertrend exit triggered - compare which is better
            if self.direction == TradeDirection.LONG:
                # For LONG: higher exit price is better (smaller loss)
                if sl_exit_price >= supertrend_exit_price:
                    self.exit_price = sl_exit_price
                    # Check if this is a trailing stop exit or regular stop loss
                    if hasattr(self, 'trailing_stop') and self.trailing_stop and hasattr(self, 'trailing_activated') and self.trailing_activated:
                        self.exit_reason = XAUUSDStrategy.EXIT_TSL_ATR if hasattr(self, 'atr_trailing_multiplier') and self.atr_trailing_multiplier > 0 else XAUUSDStrategy.EXIT_TRAILING_STOP
                    else:
                        self.exit_reason = XAUUSDStrategy.EXIT_SL_HIT
                else:
                    self.exit_price = supertrend_exit_price
                    self.exit_reason = XAUUSDStrategy.EXIT_TSL_SUPERTREND
            else:  # SHORT
                # For SHORT: lower exit price is better (smaller loss)
                if sl_exit_price <= supertrend_exit_price:
                    self.exit_price = sl_exit_price
                    # Check if this is a trailing stop exit or regular stop loss
                    if hasattr(self, 'trailing_stop') and self.trailing_stop and hasattr(self, 'trailing_activated') and self.trailing_activated:
                        self.exit_reason = XAUUSDStrategy.EXIT_TSL_ATR if hasattr(self, 'atr_trailing_multiplier') and self.atr_trailing_multiplier > 0 else XAUUSDStrategy.EXIT_TRAILING_STOP
                    else:
                        self.exit_reason = XAUUSDStrategy.EXIT_SL_HIT
                else:
                    self.exit_price = supertrend_exit_price
                    self.exit_reason = XAUUSDStrategy.EXIT_TSL_SUPERTREND
            return True
        elif sl_exit:
            # Only stop loss triggered
            self.exit_price = sl_exit_price
            # Check if this is a trailing stop exit or regular stop loss
            if hasattr(self, 'trailing_stop') and self.trailing_stop and hasattr(self, 'trailing_activated') and self.trailing_activated:
                self.exit_reason = XAUUSDStrategy.EXIT_TSL_ATR if hasattr(self, 'atr_trailing_multiplier') and self.atr_trailing_multiplier > 0 else XAUUSDStrategy.EXIT_TRAILING_STOP
            else:
                self.exit_reason = XAUUSDStrategy.EXIT_SL_HIT
            return True
        elif supertrend_exit:
            # Only supertrend exit triggered
            self.exit_price = supertrend_exit_price
            self.exit_reason = XAUUSDStrategy.EXIT_TSL_SUPERTREND
            return True
        
        return False    

    def update_trailing_stop(self, current_price: float, atr: float):
        """
        Update the trailing stop based on price action and ATR.
        Now includes soft TSL logic.
        """
        if not self.trailing_stop:
            return
            
        # Store the original stop loss for comparison
        original_stop = self.stop_loss
        
        if self.direction == TradeDirection.LONG:
            # For long positions, trail the stop up as price increases
            self.highest_high = max(self.highest_high, current_price) if self.highest_high is not None else current_price
            
            # Calculate profit in pips
            point_size = self._strategy.point_size if hasattr(self._strategy, 'point_size') else 0.01
            profit_pips = (current_price - self.entry_price) / point_size  # Get from config
            risk_pips = (self.entry_price - self.stop_loss) / point_size
            
            # Check if soft TSL conditions are met
            soft_tsl_triggered = False
            if hasattr(self, '_strategy') and self._strategy:
                if self._strategy._soft_tsl_enabled:
                    # Check pip-based trigger
                    if profit_pips >= self._strategy._soft_tsl_trigger_pips:
                        soft_tsl_triggered = True
                    # Check risk-reward trigger
                    elif risk_pips > 0 and (profit_pips / risk_pips) >= self._strategy._soft_tsl_trigger_rr:
                        soft_tsl_triggered = True
            
            # Only update trailing stop if soft TSL is triggered (or if soft TSL is disabled)
            if soft_tsl_triggered or not getattr(self._strategy, '_soft_tsl_enabled', False):
                new_stop = self.highest_high - (atr * self.atr_trailing_multiplier)
                
                # Only move stop up, never down
                if new_stop > self.stop_loss:
                    old_stop = self.stop_loss
                    self.stop_loss = new_stop
                    logger.debug(f"Updated LONG trailing stop: {old_stop:.2f} -> {self.stop_loss:.2f} "
                            f"(Price: {current_price:.2f}, ATR: {atr:.2f}, Multiplier: {self.atr_trailing_multiplier})")
                self.trailing_activated = True  # Activate trailing
            else:
                self.trailing_activated = False
            save_internal_sl(self.magic_number, self.stop_loss)
        elif self.direction == TradeDirection.SHORT:
            # For short positions, trail the stop down as price decreases
            self.lowest_low = min(self.lowest_low, current_price) if self.lowest_low is not None else current_price
            
            # Calculate profit in pips
            point_size = self._strategy.point_size if hasattr(self._strategy, 'point_size') else 0.01
            profit_pips = (self.entry_price - current_price) / point_size  # Get from config
            risk_pips = (self.stop_loss - self.entry_price) / point_size
            
            # Check if soft TSL conditions are met
            soft_tsl_triggered = False
            if hasattr(self, '_strategy') and self._strategy:
                if self._strategy._soft_tsl_enabled:
                    # Check pip-based trigger
                    if profit_pips >= self._strategy._soft_tsl_trigger_pips:
                        soft_tsl_triggered = True
                    # Check risk-reward trigger
                    elif risk_pips > 0 and (profit_pips / risk_pips) >= self._strategy._soft_tsl_trigger_rr:
                        soft_tsl_triggered = True
            
            # Only update trailing stop if soft TSL is triggered (or if soft TSL is disabled)
            if soft_tsl_triggered or not getattr(self._strategy, '_soft_tsl_enabled', False):
                new_stop = self.lowest_low + (atr * self.atr_trailing_multiplier)
                
                # Only move stop down, never up
                if new_stop < self.stop_loss or self.stop_loss == 0:
                    old_stop = self.stop_loss
                    self.stop_loss = new_stop
                    logger.debug(f"Updated SHORT trailing stop: {old_stop:.2f} -> {self.stop_loss:.2f} "
                            f"(Price: {current_price:.2f}, ATR: {atr:.2f}, Multiplier: {self.atr_trailing_multiplier})")
                self.trailing_activated = True  # Activate trailing
            else:
                self.trailing_activated = False
            save_internal_sl(self.magic_number, self.stop_loss)
        # Log if the stop loss was updated
        if self.stop_loss != original_stop:
            logger.info(f"Trailing stop updated for {self.direction} position: {original_stop:.2f} -> {self.stop_loss:.2f}")

@dataclass
class TradeSignal:
    direction: TradeDirection
    entry: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    symbol: str
    trailing_stop: bool
    atr_trailing_multiplier: float
    highest_high: float
    lowest_low: float
    position_size: float = field(init=False)  # Will be calculated based on risk
    
    def __post_init__(self):
        """
        Initialize trade signal with proper position sizing and trailing stop levels.
        """
        # Initialize highest_high and lowest_low based on direction
        if self.direction == TradeDirection.LONG:
            self.highest_high = self.entry  # Initialize with entry price for long positions
            self.lowest_low = float('inf')   # Will be updated as price moves up
        else:  # SHORT
            self.lowest_low = self.entry     # Initialize with entry price for short positions
            self.highest_high = 0            # Will be updated as price moves down
            
        # Get the strategy instance that created this signal
        strategy = getattr(self, '_strategy', None)
        if strategy is None:
            # If no strategy is provided, use minimum lot size
            self.position_size = 0.01  # Default minimum lot size
            return
            
        # Calculate position size based on risk parameters
        try:
            risk_amount = (strategy.account_balance * strategy.risk_per_trade_pct / 100)
        except (AttributeError, TypeError):
            # Fallback if strategy attributes are missing or invalid
            self.position_size = 0.01  # Default minimum lot size
            return
        
        # Calculate stop loss in points
        if self.stop_loss is not None:
            # Get point size from strategy config
            point_value = strategy.point_size if hasattr(strategy, 'point_size') else 0.01
            stop_loss_points = abs(self.entry - self.stop_loss) / point_value  # Convert to points
            
            # Calculate position size considering pip value and commission
            if stop_loss_points > 0 and hasattr(strategy, '_pip_value_per_001_lot') and strategy._pip_value_per_001_lot > 0:
                # Convert pip value to per 1.0 lot (from per 0.01 lot)
                pip_value_per_lot = strategy._pip_value_per_001_lot * 100  # Now it's per 1.0 lot
                
                # Calculate maximum position size without commission
                # risk_amount = position_size * stop_loss_points * pip_value_per_lot
                max_position = risk_amount / (stop_loss_points * pip_value_per_lot)
                
                # Calculate commission for this position size (round turn)
                commission = max_position * strategy.commission_per_lot
                
                # Adjust risk amount by subtracting commission
                adjusted_risk = max(0.01, risk_amount - commission)
                
                # Recalculate position size with adjusted risk
                self.position_size = adjusted_risk / (stop_loss_points * pip_value_per_lot)
                
                # Recalculate commission with new position size (more accurate)
                commission = self.position_size * strategy.commission_per_lot
                
                # Final adjustment to ensure we don't exceed risk after commission
                if (self.position_size * stop_loss_points * pip_value_per_lot) + commission > risk_amount:
                    adjusted_risk = max(0.01, risk_amount - commission)
                    self.position_size = adjusted_risk / (stop_loss_points * pip_value_per_lot)
            else:
                self.position_size = strategy.min_lot_size
                
            # Apply min/max lot size constraints
            self.position_size = max(strategy.min_lot_size, min(strategy.max_lot_size, self.position_size))
        else:
            self.position_size = strategy.min_lot_size
    
    exit_reason: str = None
    exit_time: datetime = None
    exit_price: float = None
    
class XAUUSDStrategy:
    # Exit reason constants
    EXIT_TP_HIT = 'TP_HIT'  # Take profit hit
    EXIT_SL_HIT = 'SL_HIT'  # Stop loss hit
    EXIT_TRAILING_STOP = 'TRAILING_STOP'  # Generic trailing stop
    EXIT_TSL_ATR = 'TSL_ATR'  # Trailing stop loss (ATR-based)
    EXIT_TSL_STATIC = 'TSL_STATIC'  # Static trailing stop loss
    EXIT_TSL_SUPERTREND = 'TSL_SUPERTREND'  # SuperTrend-based trailing stop
    EXIT_TRAILING_ACTIVATED = 'TRAILING_ACTIVATED'  # Trailing stop activated
    EXIT_SIGNAL_REVERSAL = 'SIGNAL_REVERSAL'  # Signal reversal exit
    EXIT_LOW_CONFIDENCE = 'LOW_CONFIDENCE'  # Low confidence in signal
    EXIT_MANUAL = 'MANUAL'  # Manual exit
    # Removed: EXIT_VOLATILITY and EXIT_VOLATILITY_EXIT - ATR-based trading handles volatility
    EXIT_SESSION_END = 'SESSION_END'  # End of trading session
    EXIT_MAX_DRAWDOWN = 'MAX_DRAWDOWN'  # Max drawdown reached
    EXIT_MAX_LOSS = 'MAX_LOSS'  # Max loss per trade reached
    EXIT_TIME_EXIT = 'TIME_EXIT'  # Time-based exit
    EXIT_TIME_EXPIRY = 'TIME_EXPIRY'  # Time expiry exit
    EXIT_OTHER = 'OTHER'  # Other exit reason

    def _clean_config_value(self, value: str) -> str:
        """Clean config value by removing comments and extra whitespace.
        
        Args:
            value: Raw value from config file
            
        Returns:
            str: Cleaned value with comments and extra whitespace removed
        """
        if not value:
            return value
        # Split on '#' and take the first part, then strip whitespace
        return value.split('#')[0].strip()
        
    def __init__(self, config_path: str = None, **kwargs):
        """
        Initialize the XAUUSD trading strategy with parameters from config file or arguments.
        
        Args:
            config_path: Optional path to config file. If None, uses default config.
            **kwargs: Override any config parameters
        """
        # Store config path for reloading
        self._config_path = config_path

        # Initialize config
        self._config = None
        self._init_config()

        # Initialize basic timezone settings (without server timezone detection yet)
        # This will set up default and execution timezones, but not detect server timezone
        self.init_timezones()

        # Track last trade exit information
        self.last_trade_exit = {
            'price': None,      # Last exit price
            'time': None,       # Time of last exit
            'reason': None,     # Reason for exit (TP_HIT, SL_HIT, etc.)
            'direction': None   # Direction of the exited trade (LONG or SHORT)
        }
        self._tp_block_active = False  # Block entries after TP
        self._tp_block_start_time = None  # When the block started
        self._signal_cooldown_bars = 3  # Number of bars to block after TP exit
        
        # Store any kwargs for parameter overrides
        self._kwargs = kwargs
        
        # Initialize all parameters
        self._init_parameters()

        # Print loaded parameters for debugging
        self._print_parameters()

        # Initialize set to track seen signals and avoid duplicates
        self._seen_signals = set()
        
        # Initialize trade tracking variables
        self._open_positions = {}
        self._next_magic_number = 1
        self._trade_logger = None
        
        # Signal flags for delayed entry
        self._pending_long_signal = False
        self._pending_short_signal = False
        self._pending_signal_timestamp = None
        
        # Trading style attribute
        self.trading_style = None
        
        # Daily drawdown tracking variables
        self._daily_peak_balance = None
        self._daily_start_balance = None
        self._last_drawdown_check_date = None
        self._daily_drawdown_triggered = False
        
        # Initialize MT5 connection manager
        self._mt5_manager = None
        try:
            logger.info("Initializing MT5 connection manager")
            self._mt5_manager = MT5ConnectionManager(self._config)
            if not self._mt5_manager.initialize():
                logger.warning("Failed to initialize MT5 connection during strategy startup")
            else:
                logger.info("Successfully connected to MT5")
                
                # Enable Auto Trading
                if not self._mt5_manager.check_auto_trading_enabled():
                    logger.warning("Please enable AutoTrading in MetaTrader before proceeding.")

                # Now that MT5 is connected, we can detect the server timezone if needed
                if self._auto_detect_server_timezone and self._server_timezone is None:
                    logger.info("MT5 connected, now detecting server timezone...")
                    self.detect_server_timezone()
                    logger.info(f"Server timezone detected: {self._server_timezone}")
        except Exception as e:
            logger.error(f"Error initializing MT5 connection: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _init_config(self):
        """Initialize configuration from file or defaults"""
        if self._config_path:
            from config_manager import ConfigManager
            self._config = ConfigManager(self._config_path)
        else:
            from config_manager import get_config
            self._config = get_config()

    def init_timezones(self):
        """Initialize timezones from config"""
        logger.info("Initializing timezones from config")

        # Get execution timezone (always required)
        execution_tz_str = self._config.get("Strategy", "execution_timezone")
        try:
            self._execution_timezone = pytz.timezone(execution_tz_str)
            logger.info(f"Execution timezone set to: {execution_tz_str}")
        except Exception as e:
            logger.error(f"Invalid execution timezone: {execution_tz_str}, error: {e}. Using UTC.")
            self._execution_timezone = pytz.UTC

        # Get default timezone (with support for 'server_timezone' keyword)
        default_tz_str = self._config.get("Strategy", "default_timezone")
        if default_tz_str.lower() == '${execution_timezone}':
            # Use execution timezone as default
            self._default_timezone = self._execution_timezone
            logger.info("Default timezone set to execution timezone")
        else:
            try:
                self._default_timezone = pytz.timezone(default_tz_str)
                logger.info(f"Default timezone set to: {default_tz_str}")
            except Exception as e:
                logger.error(f"Invalid default timezone: {default_tz_str}, error: {e}. Using execution timezone.")
                self._default_timezone = self._execution_timezone

        # Initialize server timezone to None - will be detected later if needed
        self._server_timezone = None

        # Auto-detection flag from config
        self._auto_detect_server_timezone = self._config.getboolean("Strategy", "auto_detect_server_timezone")
        logger.info(f"Auto-detect server timezone: {self._auto_detect_server_timezone}")
        
        # If auto-detection is disabled, try to use the server_timezone from config
        if not self._auto_detect_server_timezone:
            server_tz_str = self._config.get("Strategy", "server_timezone")
            if server_tz_str and server_tz_str.lower() != 'none':
                try:
                    self._server_timezone = pytz.timezone(server_tz_str)
                    logger.info(f"Using server timezone from config: {server_tz_str}")
                except Exception as e:
                    logger.error(f"Invalid server timezone in config: {server_tz_str}, error: {e}")
                    self._server_timezone = None
        
        # Note: We don't attempt to detect server timezone here even if auto_detect is enabled
        # This will be done after MT5 is connected, either in __init__ or in run_strategy

    def shutdown(self):
        """
        Shut down the MT5 connection and clean up resources.
        """
        if self._mt5_manager is not None:
            self._mt5_manager.shutdown()
            logger.info("MT5 connection closed")
            
    @property
    def mt5_manager(self):
        """
        Get the MT5 connection manager instance.
        
        Returns:
            MT5ConnectionManager: The MT5 connection manager instance or None if not initialized.
        """
        return self._mt5_manager
        
    def get_mt5_data(self, symbol=None, timeframe=None, bars=500, from_date=None, to_date=None):
        """
        Fetch data from MT5 using the MT5ConnectionManager.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD'). If None, uses the strategy's symbol.
            timeframe: Timeframe string (e.g., 'M1', 'H1', 'D1'). If None, uses the strategy's timeframe.
            bars: Number of bars to fetch. Default is 500.
            from_date: Start date for historical data. If None, fetches the most recent bars.
            to_date: End date for historical data. If None, fetches up to the current time.
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data or None if fetching fails.
        """
        try:
            # Use strategy's symbol and data_timeframe if not provided
            symbol = symbol or self.symbol
            timeframe = timeframe or self.data_timeframe
            
            logger.info(f"Fetching {bars} bars of {symbol} {timeframe} data from MT5...")
            
            # Check if MT5 connection is initialized
            if self._mt5_manager is None:
                logger.error("MT5 connection manager is not initialized")
                return None
            
            if not self._mt5_manager.check_connection():
                logger.warning("MT5 is not connected. Attempting to reconnect...")
                if not self._mt5_manager.initialize():
                    logger.error("Failed to reconnect to MT5")
                    return None
            
            # Fetch data from MT5
            df = self._mt5_manager.get_data(symbol, timeframe, bars, from_date, to_date)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
                return df
            else:
                logger.error(f"Failed to fetch data for {symbol} {timeframe} or data is empty")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching MT5 data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _init_parameters(self):
        """Initialize all strategy parameters from config and kwargs"""
        # Strategy parameters with defaults
        self._sma_fast_period = self._get_param('sma_fast_period', int)
        self._sma_slow_period = self._get_param('sma_slow_period', int)
        self._adx_period = self._get_param('adx_period', int)
        self._adxr_period = self._get_param('adxr_period', int)
        self._rsi_period = self._get_param('rsi_period', int)
        self._rsi_ma_period = self._get_param('rsi_ma_period', int)
        self._atr_period = self._get_param('atr_period', int)
        
        # Trading style parameters
        self._atr_swing_threshold = 1.30  # ATR threshold for swing vs scalp trading style
        self._current_trading_style = 'scalp'  # Default trading style
        # self._config_trading_style = self._get_param('trading_style', str, 'both')  # Config-defined trading style: 'scalp', 'swing', or 'both'

        # SuperTrend parameters with defaults
        self._st_long_period = self._get_param('st_long_period', int)
        self._st_long_multiplier = self._get_param('st_long_multiplier', float)
        self._st_short_period = self._get_param('st_short_period', int)
        self._st_short_multiplier = self._get_param('st_short_multiplier', float)
        
        # Trailing stop settings with defaults
        self._use_trailing_stop = self._get_param('use_trailing_stop', bool)
        self._atr_trailing_multiplier = self._get_param('atr_trailing_multiplier', float)
        
        # CPR Indicator settings
        self._cpr_prior_day_high = self._get_param('cpr_prior_day_high', float)
        self._cpr_prior_day_low = self._get_param('cpr_prior_day_low', float)
        self._cpr_prior_day_close = self._get_param('cpr_prior_day_close', float)
        self._cpr_pivot_width = self._get_param('cpr_pivot_width', int)
        self._cpr_tc_width = self._get_param('cpr_tc_width', int)
        self._cpr_bc_width = self._get_param('cpr_bc_width', int)
        
        # Trade logic settings
        self._trade_logic_base = self._get_param('trade_logic_base', bool)
        self._trade_logic_base_multi_timeframe = self._get_param('trade_logic_base_multi_timeframe', bool)
        self._trade_logic_marketstructure = self._get_param('trade_logic_marketstructure', bool)
        self._trade_logic_pivots = self._get_param('trade_logic_pivots', bool)
        self._trade_logic_cpr = self._get_param('trade_logic_CPR', bool)

        # Signal strength calculator settings (hardcoded - hidden from user)
        self._min_signal_strength_pct = 50.0  # Lowered from 75% to 30% for testing
        self._signal_strength_weighting = True  # Always use weighted scoring
                    
        # Trading hours with defaults (24-hour format)
        self._start_hour = self._get_param('start_hour', int)               # 00:00
        self._end_hour = self._get_param('end_hour', int)                   # 23:59
        
        # Position sizing with defaults
        self._initial_account_balance = self._get_param('account_balance', float)
        # Initialize account balance - will be updated during backtesting
        self.account_balance = self._initial_account_balance
        self._risk_per_trade_pct = self._get_param('risk_per_trade_pct', float)  # 1% default risk
        # ATR-based trading replaces fixed stop loss and take profit
        # self._stop_loss_usd = self._get_param('stop_loss_usd', float)  # Removed - using ATR-based
        # self._take_profit_usd = self._get_param('take_profit_usd', float)  # Removed - using ATR-based
        self._min_lot_size = self._get_param('min_lot_size', float)  # 0.01 lots minimum
        self._max_lot_size = self._get_param('max_lot_size', float)  # 100 lots maximum
        self._commission_per_lot = self._get_param('commission_per_lot', float)  # $7 per lot default
        
        # Daily drawdown settings
        self._daily_drawdown_limit = self._get_param('daily_drawdown_limit', bool)  # Whether to enable daily drawdown limit
        self._daily_drawdown_limit_pct = self._get_param('daily_drawdown_limit_pct', float)  # 2% default daily drawdown limit
        self._daily_peak_balance = self._account_balance  # Track peak balance for the day
        self._last_trade_day = None  # Track the last trading day

        # Trade management settings
        self._max_open_trades = self._get_param('max_open_trades', int)  # Maximum concurrent positions
        self._signal_cooldown_bars = self._get_param('signal_cooldown_bars', int)  # Bars between signals
        self._signal_cooldown_minutes = self._get_param('signal_cooldown_minutes', int, 5)  # Minutes between any trades
        self._tp_cooldown_minutes = self._get_param('tp_cooldown_minutes', int, 5)  # Minutes to wait after TP before allowing new entries

        # Order execution settings
        self._deviation = self._get_param('deviation', int, 20)  # Maximum price deviation/slippage in points
        self._magic_number = self._get_param('magic_number', int, 4091956)  # Magic number for order identification
        self._comment = self._get_param('comment', str, 'bot_gkk')  # Order comment for trade identification

        # Entry confluence requirements
        self._require_trend_confluence = self._get_param('require_trend_confluence', bool)
        self._require_momentum_confluence = self._get_param('require_momentum_confluence', bool)
        self._min_adx_for_entry = self._get_param('min_adx_for_entry', float)
        self._rsi_oversold_level = self._get_param('rsi_oversold_level', float)
        self._rsi_overbought_level = self._get_param('rsi_overbought_level', float)

        # Trading mode configuration
        self._trading_mode = self._get_param('trading_mode', str, 'testing').lower()

        # Trade logging configuration
        self._trades_log_in_db = self._get_param('trades_log_in_db', bool, True)

        # Only use ForexFactory for actual live trading (not backtesting)
        # Check if we're doing backtesting by looking at date range
        is_backtesting = self._is_backtesting_mode()
        self._use_forexfactory = (self._trading_mode == 'live' and not is_backtesting)

        # Market condition filters
        self._avoid_news_times = self._get_param('avoid_news_times', bool)
        self._market_session_filter = self._get_param('market_session_filter', bool)
        self._avoid_bank_holidays = self._get_param('avoid_bank_holidays', bool)
        self._volatility_filter = self._get_param('volatility_filter', bool)
        self._min_atr_for_entry = self._get_param('min_atr_for_entry', float)
        self._max_atr_for_entry = self._get_param('max_atr_for_entry', float)

        # Track open positions and last signal times
        self._open_positions = {}  # Dict to track open positions by magic number        
        self._completed_trades = []  # List to track completed trades for reporting
        self._last_signal_time = {}  # Dict to track last signal time by direction
        self._last_trade_time = None  # Track last trade time for time-based cooldown
        self._next_magic_offset = 0  # Counter for generating unique magic numbers

        # ForexFactory data caching
        self._ff_news_cache = {}  # Cache for news data
        self._ff_holidays_cache = {}  # Cache for holiday data
        self._ff_cache_expiry = 3600  # Cache expiry in seconds (1 hour)

        # Log trading mode
        logger.info(f"Trading Mode: {self._trading_mode.upper()}")
        if is_backtesting:
            logger.info("Detected: BACKTESTING mode (historical data)")
        if self._use_forexfactory:
            logger.info("📡 ForexFactory integration: ENABLED (Live trading)")
        else:
            reason = "Backtesting detected" if is_backtesting else "Testing mode configured"
            logger.info(f"ForexFactory integration: DISABLED ({reason} - using static rules)")
        
        # Get pip value with a safe default (0.10 for XAUUSD)
        self._pip_value_per_001_lot = self._get_param('pip_value_per_001_lot', float)
        
        # Ensure pip_value_per_001_lot is valid
        if self._pip_value_per_001_lot is None or self._pip_value_per_001_lot <= 0:
            print("Warning: Invalid pip_value_per_001_lot, using default value of 0.10")
            self._pip_value_per_001_lot = 0.10
            
        # Calculate pip value per standard lot (1.0 lot)
        self._pip_value_per_lot = self._pip_value_per_001_lot * 100.0  # Convert from 0.01 lot to 1.0 lot
        
        # Date range with defaults (today - 30 days to today)
        today = datetime.now().date()
        thirty_days_ago = today - timedelta(days=30)
        
        self._from_date = self._get_param('from_date', str, thirty_days_ago.strftime('%Y-%m-%d'))
        self._to_date = self._get_param('to_date', str, today.strftime('%Y-%m-%d'))
        
        # Symbol and timeframes with defaults
        self._symbol = self._get_param('symbol', str)
        self._data_timeframe = self._get_param('timeframe', str)
        self._execution_timeframe = self._get_param('execution_timeframe', str)
        self._high_time_frame = self._get_param('high_time_frame', str, 'M5')
    
    def _get_param(self, name, param_type, default=None):
        """
        Get parameter value from kwargs, then config, with type conversion.
        
        Args:
            name: Parameter name
            param_type: Type to convert to (int, float, str, bool)
            default: Default value if not found
            
        Returns:
            Parameter value with appropriate type
        """
        # First check kwargs
        if name in self._kwargs and self._kwargs[name] is not None:
            value = self._kwargs[name]
            logger.debug(f"_get_param: Found {name} in kwargs: {value}")
        else:
            # Then check config
            try:
                value = self._config.get('Strategy', name, fallback=None)
                logger.debug(f"_get_param: Found {name} in config: {value}")
                if value is None:
                    logger.debug(f"_get_param: {name} not found, using default: {default}")
                    return default
            except (configparser.NoOptionError, configparser.NoSectionError) as e:
                logger.debug(f"_get_param: Error reading {name} from config: {e}, using default: {default}")
                return default
        
        # If we have a value, try to convert it
        if value is not None:
            try:
                # Handle comments in config values (split on '#' and take first part)
                if isinstance(value, str):
                    value = value.split('#')[0].strip()
                    if not value:  # If nothing left after removing comment
                        return default
                
                if param_type == bool:
                    if isinstance(value, str):
                        return value.lower() in ('true', 'yes', '1', 't', 'on')
                    return bool(value)
                elif param_type == int:
                    return int(float(value))  # Handle float strings for int params
                elif param_type == float:
                    return float(value)
                elif param_type == str:
                    return str(value).strip()
                else:
                    return param_type(value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert {name}='{value}' to {param_type.__name__}, using default: {default}")
                return default
        
        return default
    
    def get_parameter(self, name, default=None):
        """
        Get current parameter value, checking config first, then instance attributes.
        
        Args:
            name: Parameter name (without leading underscore)
            default: Default value if not found
            
        Returns:
            Current parameter value or default
        """
        # Try to get from instance attributes first (prefixed with _)
        attr_name = f'_{name}'
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
            
        # Try to get from config
        try:
            return self._get_param(name, type(default) if default is not None else str, default)
        except (configparser.NoOptionError, configparser.NoSectionError):
            return default
    
    def reload_config(self, config_path: str = None):
        """
        Reload configuration from file and update parameters.
        
        Args:
            config_path: Optional path to config file. If None, uses existing path.
        """
        if config_path:
            self._config_path = config_path
            
        # Re-initialize config
        self._init_config()
        
        # Initialize timezones
        self.init_timezones()
        
        # Re-initialize parameters
        self._init_parameters()
        
        # Recalculate any derived values
        self._pip_value_per_lot = self._pip_value_per_001_lot * 100
        
        # Print updated parameters for debugging
        self._print_parameters()
        
        return True
        
    # Property getters for backward compatibility
    @property
    def sma_fast_period(self):
        return self._sma_fast_period
        
    @property
    def sma_slow_period(self):
        return self._sma_slow_period
        
    @property
    def adx_period(self):
        return self._adx_period

    @property
    def adxr_period(self):
        return self._adxr_period
            
    @property
    def rsi_period(self):
        return self._rsi_period
        
    @property
    def rsi_ma_period(self):
        return self._rsi_ma_period
        
    @property
    def atr_period(self):
        return self._atr_period
        
    @property
    def st_long_period(self):
        return self._st_long_period
        
    @property
    def st_long_multiplier(self):
        return self._st_long_multiplier
        
    @property
    def st_short_period(self):
        return self._st_short_period
        
    @property
    def st_short_multiplier(self):
        return self._st_short_multiplier
        
    @property
    def use_trailing_stop(self):
        return self._use_trailing_stop
        
    @property
    def atr_trailing_multiplier(self):
        return self._atr_trailing_multiplier
        
    # ATR-Based Trading Properties
    @property
    def atr_based_trading(self):
        return self._get_param('atr_based_trading', bool, True)
        
    @property
    def atr_stop_multiplier(self):
        return self._get_param('atr_stop_multiplier', float, 2.0)
        
    @property
    def atr_tp_multiplier(self):
        return self._get_param('atr_tp_multiplier', float, 4.0)
        
    @property
    def atr_position_sizing(self):
        return self._get_param('atr_position_sizing', bool, True)

    @property
    def atr_swing_threshold(self):
        return self._get_param('atr_swing_threshold', float, 1.30)

    @property
    def swing_risk_per_trade_pct(self):
        return self._get_param('swing_risk_per_trade_pct', float, 1.0)

    @property
    def atr_min_multiplier(self):
        return self._get_param('atr_min_multiplier', float, 1.0)
        
    @property
    def atr_max_multiplier(self):
        return self._get_param('atr_max_multiplier', float, 5.0)
        
    @property
    def atr_volatility_filter(self):
        return self._get_param('atr_volatility_filter', bool, True)
        
    @property
    def atr_min_value(self):
        return self._get_param('atr_min_value', float, 5.0)
        
    @property
    def atr_max_value(self):
        return self._get_param('atr_max_value', float, 50.0)
        
    # Daily Drawdown Protection Properties
    @property
    def daily_drawdown_limit(self):
        return self._get_param('daily_drawdown_limit', bool, True)
        
    @property
    def daily_drawdown_limit_pct(self):
        return self._get_param('daily_drawdown_limit_pct', float, 2.0)

    @property
    def execution_timezone(self):
        return self._execution_timezone
    
    @property
    def server_timezone(self):
        return self._server_timezone
    
    @server_timezone.setter
    def server_timezone(self, value):
        self._server_timezone = value
        
    @property
    def default_timezone(self):
        return self._default_timezone
    
    @property
    def auto_detect_server_timezone(self):
        return self._auto_detect_server_timezone
    
    @property
    def start_hour(self):
        return self._start_hour
        
    @property
    def end_hour(self):
        return self._end_hour
        
    @property
    def account_balance(self):
        return self._account_balance
        
    @account_balance.setter
    def account_balance(self, value):
        self._account_balance = value
        
    @property
    def risk_per_trade_pct(self):
        return self._risk_per_trade_pct

    @risk_per_trade_pct.setter
    def risk_per_trade_pct(self, value):
        self._risk_per_trade_pct = value
            
    @property
    def min_lot_size(self):
        return self._min_lot_size
        
    @property
    def max_lot_size(self):
        return self._max_lot_size
        
    @property
    def commission_per_lot(self):
        return self._commission_per_lot
        
    @property
    def pip_value_per_001_lot(self):
        return self._pip_value_per_001_lot
        
    @property
    def pip_value_per_lot(self):
        return self._pip_value_per_lot
    
    @property
    def point_size(self):
        """Get the point size from Strategy section config."""
        return self._get_param('point_size', float, 0.01)
        
    @property
    def from_date(self):
        return self._from_date
        
    @property
    def to_date(self):
        return self._to_date
        
    @property
    def symbol(self):
        return self._symbol
        
    @symbol.setter
    def symbol(self, value):
        self._symbol = value
        
    @property
    def data_timeframe(self):
        return self._data_timeframe
        
    @data_timeframe.setter
    def data_timeframe(self, value):
        self._data_timeframe = value
        
    @property
    def execution_timeframe(self):
        return self._execution_timeframe
        
    @execution_timeframe.setter
    def execution_timeframe(self, value):
        self._execution_timeframe = value
        
    @property
    def config(self):
        return self._config

    @property
    def deviation(self):
        return self._deviation

    @property
    def magic_number(self):
        return self._magic_number

    @property
    def comment(self):
        return self._comment
        
    @property
    def tp_cooldown_minutes(self):
        return self._tp_cooldown_minutes
        
    @property
    def trades_log_in_db(self):
        return self._trades_log_in_db

    def _print_parameters(self):
        """Print the current strategy parameters for verification."""
        params = {
            'sma_fast_period': self.sma_fast_period,
            'sma_slow_period': self.sma_slow_period,
            'adx_period': self.adx_period,
            'adxr_period': self.adxr_period,
            'rsi_period': self.rsi_period,
            'rsi_ma_period': self.rsi_ma_period,
            'atr_period': self.atr_period,
            'st_long_period': self.st_long_period,
            'st_long_multiplier': self.st_long_multiplier,
            'st_short_period': self.st_short_period,
            'st_short_multiplier': self.st_short_multiplier,
            'use_trailing_stop': self.use_trailing_stop,
            'atr_trailing_multiplier': self.atr_trailing_multiplier,
            'trade_logic_base': self._trade_logic_base,
            'trade_logic_marketstructure': self._trade_logic_marketstructure,
            'trade_logic_pivots': self._trade_logic_pivots,
            'trade_logic_cpr': self._trade_logic_cpr,
            'trading_hours': f"{self.start_hour:02d}:00-{self.end_hour:02d}:00",
            'execution_timezone': self.execution_timezone,
            'server_timezone': self.server_timezone,
            'default_timezone': self.default_timezone,
            'auto_detect_server_timezone': self.auto_detect_server_timezone,
            'symbol': self.symbol,
            'data_timeframe': self.data_timeframe,
            'execution_timeframe': self.execution_timeframe
        }
    
    def _determine_trading_style(self, current_bar):
        """
        Determine the trading style based on ATR value and ADX value.
        
        Args:
            current_bar: Current bar data with ATR and ADX values
        
        Returns:
            str: 'swing' if ATR >= 1.30 and ADX > 20, 'scalp' otherwise
        """
        # Check if ATR and ADX are available in the current bar
        if 'atr' not in current_bar:
            logger.warning("ATR not available in current bar, defaulting to scalp style")
            return 'scalp'
        
        if 'adx' not in current_bar:
            logger.warning("ADX not available in current bar, using only ATR for trading style")
            # Fall back to ATR-only logic if ADX is not available
            if current_bar['atr'] >= self._atr_swing_threshold:
                if self._current_trading_style != 'swing':
                    logger.info(f"Switching to SWING trading style (ATR: {current_bar['atr']:.2f} >= {self._atr_swing_threshold})")
                    self._current_trading_style = 'swing'
            else:
                if self._current_trading_style != 'scalp':
                    logger.info(f"Switching to SCALP trading style (ATR: {current_bar['atr']:.2f} < {self._atr_swing_threshold})")
                    self._current_trading_style = 'scalp'
            return self._current_trading_style
            
        # Determine trading style based on ATR and ADX values
        if current_bar['atr'] >= self._atr_swing_threshold and current_bar['adx'] > 20:
            if self._current_trading_style != 'swing':
                logger.info(f"Switching to SWING trading style (ATR: {current_bar['atr']:.2f} >= {self._atr_swing_threshold}, ADX: {current_bar['adx']:.2f} > 20)")
                self._current_trading_style = 'swing'
        else:
            if self._current_trading_style != 'scalp':
                logger.info(f"Switching to SCALP trading style (ATR: {current_bar['atr']:.2f} < {self._atr_swing_threshold} or ADX: {current_bar['adx']:.2f} <= 20)")
                self._current_trading_style = 'scalp'
                
        return self._current_trading_style

    def _ensure_utc_datetime(self, dt):
        """
        Ensures a datetime is timezone-aware and in UTC.
        
        Args:
            dt: The datetime object to convert to UTC
            
        Returns:
            A timezone-aware datetime object in UTC
        """
        # Convert to datetime if it's not already
        if not isinstance(dt, datetime):
            dt = pd.to_datetime(dt)
        
        # If datetime is naive (no timezone info), assume it's in execution timezone
        if dt.tzinfo is None and self._execution_timezone is not None:
            # Localize with execution timezone
            dt = self._execution_timezone.localize(dt)
        elif dt.tzinfo is None:
            # If no execution timezone, use UTC
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to UTC if it has timezone info
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        
        return dt
        
    def detect_server_timezone(self):
        """
        Detect MT5 server timezone by comparing server time with UTC.
        Returns a pytz timezone object or fixed offset timezone.
        Falls back to default timezone if detection fails.
        """
        logger.info("Detecting server timezone from MT5 server time...")
        
        # Verify MT5 connection first
        if not self._mt5_manager or not self._mt5_manager.check_connection():
            logger.warning("MT5 connection not available for timezone detection. Using default timezone.")
            self._server_timezone = self._default_timezone
            return self._server_timezone
        
        try:
            # Try multiple methods to get the most accurate server time
            utc_now = datetime.now(timezone.utc)
            server_dt = None
            time_source = ""
            
            # Method 1: Try terminal_info().time (most reliable for server time)
            try:
                terminal_info = mt5.terminal_info()
                if terminal_info and hasattr(terminal_info, 'time'):
                    server_dt = datetime.fromtimestamp(terminal_info.time, tz=timezone.utc)
                    time_source = "terminal_info"
                    logger.info(f"Using terminal_info time: {server_dt.isoformat()}")
            except Exception as e:
                logger.warning(f"Error getting terminal_info time: {e}")
            
            # Method 2: Try account_info server time
            if server_dt is None:
                try:
                    account_info = mt5.account_info()
                    if account_info and hasattr(account_info, 'server'):
                        logger.info(f"Broker server: {account_info.server}")
                        # For some brokers, we can directly map their server name to a timezone
                        broker_name = account_info.server.lower() if hasattr(account_info, 'server') else ""
                        
                        # Force UTC+3 for specific brokers known to use this timezone
                        if any(name in broker_name for name in ['exness', 'xm', 'fbs', 'icmarket', 'tickmill', 'fxtm']):
                            logger.info(f"Broker {broker_name} is known to use UTC+3, forcing Europe/Moscow timezone")
                            self._server_timezone = pytz.timezone("Europe/Moscow")
                            return self._server_timezone
                        
                        # Still need a timestamp, so fall back to tick time
                        server_time = mt5.symbol_info_tick(self.symbol)
                        if server_time and hasattr(server_time, 'time'):
                            server_dt = datetime.fromtimestamp(server_time.time, tz=timezone.utc)
                            time_source = "account_info + tick"
                            logger.info(f"Using account_info + tick time: {server_dt.isoformat()}")
                except Exception as e:
                    logger.warning(f"Error getting account_info: {e}")
            
            # Method 3: Fall back to symbol_info_tick
            if server_dt is None:
                server_time = mt5.symbol_info_tick(self.symbol)
                if server_time is None or not hasattr(server_time, 'time'):
                    logger.warning("Could not get valid server time from any source, using default timezone")
                    self._server_timezone = self._default_timezone
                    return self._server_timezone
                server_dt = datetime.fromtimestamp(server_time.time, tz=timezone.utc)
                time_source = "symbol_info_tick"
                logger.info(f"Using symbol_info_tick time: {server_dt.isoformat()}")
            
            # Log raw times for debugging
            logger.info(f"Time source used: {time_source}")
            logger.info(f"Raw UTC time: {utc_now.isoformat()}")
            logger.info(f"Raw server time: {server_dt.isoformat()}")
            logger.info(f"Raw time difference in seconds: {(server_dt - utc_now).total_seconds()}")
            
            # Calculate offset
            offset_seconds = (server_dt - utc_now).total_seconds()
            # Don't round the hours, use integer division to get the whole hours
            offset_hours = int(offset_seconds / 3600)
            # Calculate remaining minutes without rounding
            offset_minutes = int((abs(offset_seconds) % 3600) / 60)
            
            logger.info(f"Calculated offset: {offset_hours} hours and {offset_minutes} minutes")
            
            # Handle fractional hour offsets (like India UTC+5:30)
            # Create a precise floating point representation of the offset
            offset_hours_float = offset_hours + (offset_minutes / 60) if offset_minutes > 0 else offset_hours
            
            # Map common broker timezones with tolerance
            timezone_map = {
                0: "UTC",
                1: "Europe/Paris",     # GMT+1
                2: "Europe/Kiev",      # GMT+2 (many European brokers)
                3: "Europe/Moscow",    # GMT+3 (many European brokers)
                -5: "America/New_York", # GMT-5 (some US brokers)
                -4: "America/Halifax",  # GMT-4
                5.5: "Asia/Kolkata",    # GMT+5:30 (India)
                8: "Asia/Singapore",    # GMT+8 (Singapore, Hong Kong)
                9: "Asia/Tokyo"        # GMT+9 (Japan)
            }
            
            # Add tolerance for timezone matching (±0.1 hours or 6 minutes)
            tolerance = 0.1
            detected_tz = None
            
            # Special case: if offset is close to 3 hours, force Europe/Moscow (UTC+3)
            if 2.9 <= offset_hours_float <= 3.1:
                detected_tz = "Europe/Moscow"
                logger.info(f"Offset {offset_hours_float} is close to 3 hours, forcing Europe/Moscow timezone (UTC+3)")
            else:
                # Try to match with other timezones
                for tz_offset, tz_name in timezone_map.items():
                    if abs(offset_hours_float - tz_offset) <= tolerance:
                        detected_tz = tz_name
                        logger.info(f"Matched timezone {tz_name} with offset {tz_offset} (actual offset: {offset_hours_float})")
                        break
            
            # If we found a match in the map, use named timezone
            if detected_tz:
                self._server_timezone = pytz.timezone(detected_tz)
                logger.info(f"Detected server timezone: {detected_tz} (offset: {offset_hours_float} hours)")
            else:
                # Create a fixed offset timezone with proper formatting
                if offset_minutes > 0:
                    # Format the timezone string properly with zero-padded hours and minutes
                    sign = '+' if offset_hours >= 0 else '-'
                    abs_hours = abs(offset_hours)
                    tz_str = f"UTC{sign}{abs_hours:02d}:{offset_minutes:02d}"
                    
                    # Create the timezone object
                    total_seconds = (offset_hours * 3600) + (offset_minutes * 60)
                    self._server_timezone = timezone(timedelta(seconds=total_seconds))
                    logger.info(f"Created fixed offset timezone: {tz_str}")
                else:
                    # Format the timezone string properly with zero-padded hours
                    sign = '+' if offset_hours >= 0 else '-'
                    abs_hours = abs(offset_hours)
                    tz_str = f"UTC{sign}{abs_hours:02d}:00"
                    
                    # Create the timezone object
                    self._server_timezone = timezone(timedelta(hours=offset_hours))
                    logger.info(f"Created fixed offset timezone: {tz_str}")
            
            # Verify the timezone works by converting a time
            test_time = utc_now.astimezone(self._server_timezone)
            logger.info(f"Server time (verified): {test_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
            
            # Additional verification - convert UTC to server time and back
            server_time_from_utc = utc_now.astimezone(self._server_timezone)
            utc_time_from_server = server_time_from_utc.astimezone(timezone.utc)
            logger.info(f"UTC → Server → UTC verification: {utc_now.isoformat()} → {server_time_from_utc.isoformat()} → {utc_time_from_server.isoformat()}")
            logger.info(f"Final detected server timezone offset: {self._server_timezone.utcoffset(datetime.now())}")
            return self._server_timezone
            
        except Exception as e:
            logger.error(f"Error detecting server timezone: {str(e)}")
            logger.exception("Timezone detection exception details:")
            self._server_timezone = self._default_timezone
            logger.info(f"Falling back to default timezone: {self._default_timezone}")
            return self._server_timezone

    def convert_timezone(self, dt, from_tz, to_tz):
        """Convert datetime from one timezone to another"""
        if dt.tzinfo is None:
            # Assume the datetime is in from_tz
            dt = from_tz.localize(dt)
        else:
            # Convert to from_tz first
            dt = dt.astimezone(from_tz)
        
        # Convert to target timezone
        return dt.astimezone(to_tz)
    
    def get_current_time_info(self) -> Dict[str, datetime]:
        """
        Get current time in all relevant timezones (UTC, server, execution, default).
        Returns a dictionary with timezone-aware datetime objects for each timezone.
        If a timezone is not available, falls back to UTC.
        """
        # Start with UTC time as the reference point
        utc_now = datetime.now(timezone.utc)
        result = {'utc': utc_now}
        
        # Dictionary to store timezone info for logging
        tz_info = {
            'server': {'object': self._server_timezone, 'name': 'Server'},
            'execution': {'object': self._execution_timezone, 'name': 'Execution'},
            'default': {'object': self._default_timezone, 'name': 'Default'}
        }
        
        # Process each timezone in a consistent way
        for tz_key, info in tz_info.items():
            try:
                tz_obj = info['object']
                if tz_obj is None:
                    result[tz_key] = utc_now
                    logger.debug(f"{info['name']} timezone is None, using UTC")
                else:
                    # Convert UTC time to the target timezone
                    localized_time = utc_now.astimezone(tz_obj)
                    result[tz_key] = localized_time
                    
                    # Get timezone name and offset for logging
                    if hasattr(tz_obj, 'zone'):
                        tz_name = tz_obj.zone
                    elif hasattr(tz_obj, 'key'):
                        tz_name = tz_obj.key
                    else:
                        tz_name = str(tz_obj)
                        
                    # Log the conversion with timezone offset
                    offset_str = localized_time.strftime('%z')
                    logger.debug(f"Converted UTC time to {info['name']} timezone: {tz_name} (UTC{offset_str})")
                    
            except Exception as e:
                result[tz_key] = utc_now
                logger.error(f"Error converting to {info['name']} timezone: {str(e)}")
                logger.debug(f"Timezone object type: {type(info['object'])}")
        
        # Log a summary of all timezone conversions at debug level
        logger.debug("Current time in all timezones:")
        for tz_name, dt in result.items():
            logger.debug(f"  {tz_name.capitalize()}: {dt.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
            
        return result
                
    # def get_mt5_data(self, symbol=None, timeframe=None, bars=500, from_date=None, to_date=None):
    #     """
    #     Fetch data from MT5 using the MT5ConnectionManager with proper timezone localization.

    #     Args:
    #         symbol: Trading symbol (e.g., 'XAUUSD'). If None, uses the strategy's symbol.
    #         timeframe: Timeframe string (e.g., 'M1', 'H1', 'D1'). If None, uses the strategy's timeframe.
    #         bars: Number of bars to fetch. Default is 500.
    #         from_date: Start date for historical data. If None, fetches the most recent bars.
    #         to_date: End date for historical data. If None, fetches up to the current time.

    #     Returns:
    #         pandas.DataFrame: DataFrame with OHLCV data or None if fetching fails.
    #     """
    #     try:
    #         # Use strategy's symbol and timeframe if not provided
    #         symbol = symbol or self.symbol
    #         timeframe = timeframe or self.timeframe

    #         logger.info(f"Fetching {bars} bars of {symbol} {timeframe} data from MT5...")

    #         # Check MT5 manager
    #         if self._mt5_manager is None:
    #             logger.error("MT5 connection manager is not initialized")
    #             return None

    #         if not self._mt5_manager.ensure_connected():
    #             logger.warning("MT5 is not connected. Attempting to reconnect...")
    #             if not self._mt5_manager.initialize():
    #                 logger.error("Failed to reconnect to MT5")
    #                 return None

    #         # Fetch data
    #         df = self._mt5_manager.get_data(symbol, timeframe, bars, from_date, to_date)

    #         if df is not None and not df.empty:
    #             # Safe timezone localization
    #             try:
    #                 server_tz = self.server_timezone
    #                 print("Server TimeZone:",server_tz)
    #                 df.index = df.index.tz_localize(server_tz).tz_convert(self.default_timezone)
    #             except Exception as tz_err:
    #                 logger.error(f"Timezone conversion failed: {tz_err}")
    #                 return None
                
    #             logger.info(f"Successfully fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    #             return df
    #         else:
    #             logger.error(f"Failed to fetch data for {symbol} {timeframe} or data is empty")
    #             return None

    #     except Exception as e:
    #         logger.error(f"Error fetching MT5 data: {e}")
    #         import traceback
    #         logger.error(traceback.format_exc())
    #         return None



    def load_historical_data(self, symbol: str, timeframe: str, from_date: datetime = None, to_date: datetime = None, use_db: bool = True, lookback_days=1) -> pd.DataFrame:
        """
        Load historical price data for backtesting or live trading, with automatic lookback for indicator warmup.
        If use_db is True, load from local database using MT5DataLoader.
        Otherwise, load from MT5.
        Only returns data for the requested from_date to to_date range, but loads extra lookback for indicator accuracy.
        """
        # Calculate lookback start date
        if from_date is not None:
            load_from_date = (pd.to_datetime(from_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            load_from_date = None
        load_to_date = to_date

        if use_db:
            try:
                from create_ohlc_table_load_data import MT5DataLoader
                loader = MT5DataLoader()
                df = loader.load_from_database(from_date=load_from_date, to_date=load_to_date)                
                if df is None or df.empty:
                    logger.warning(f"No data loaded from database for {symbol} {timeframe}.")
                    return df
                else:
                    # --- Standardize DataFrame to match live mode ---
                    column_map = {
                        'opne': 'Open', 'open': 'Open',
                        'clsoe': 'Close', 'close': 'Close',
                        'hihg': 'High', 'high': 'High',
                        'lwo': 'Low', 'low': 'Low',
                        'tick_volume': 'Volume', 'volume': 'Volume'
                    }
                    for k, v in column_map.items():
                        if k in df.columns:
                            df[v] = df[k]
                    df = df.loc[:, ~df.columns.duplicated()]
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col not in df.columns:
                            df[col] = 0.0
                    df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'])
                            df.set_index('time', inplace=True)
                        elif 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                    # --- End standardization ---
                    logger.info(f"Loaded {len(df)} bars from database for {symbol} {timeframe} (with lookback).")
                    print(f"Loaded {len(df)} rows for {symbol} {timeframe} from {load_from_date} to {load_to_date}")
                # Return the full DataFrame (with lookback) for indicator calculation.
                # After indicator calculation, filter to requested range for signals/metrics.
                return df
            except Exception as e:
                logger.error(f"Error loading historical data from database: {e}")
                return None
        else:
            import MetaTrader5 as mt5
            try:
                if not mt5.initialize():
                    logger.error("MT5 initialize failed")
                    return None
                rates = mt5.copy_rates_range(symbol, getattr(mt5, f"TIMEFRAME_{timeframe}"), load_from_date, load_to_date)
                if rates is None or len(rates) == 0:
                    logger.warning(f"No data loaded from MT5 for {symbol} {timeframe}.")
                    return None
                df = pd.DataFrame(rates)
                logger.info(f"Loaded {len(df)} bars from MT5 for {symbol} {timeframe} (with lookback).")
                return df
            except Exception as e:
                logger.error(f"Error loading historical data from MT5: {e}")
                return None
         
    def display_indicator_values(self, df: pd.DataFrame):
        """
        Display the latest values of all indicators in the console.
        
        Args:
            df: DataFrame containing the indicator values
        """
        if df.empty:
            print("No data available to display indicator values.")
            return
            
        # Get the last row (most recent candle)
        latest = df.iloc[-1].copy()
        
        # Format the output
        print("\n" + "=" * 50)
        print(f"=== INDICATOR VALUES - {latest.name} ===")
        print("=" * 50)
        
        # Price and Volume
        print("\n=== PRICE & VOLUME ===")
        print(f"Open:       {latest.get('Open', 'N/A'):.2f}")
        print(f"High:       {latest.get('High', 'N/A'):.2f}")
        print(f"Low:        {latest.get('Low', 'N/A'):.2f}")
        print(f"Close:      {latest.get('Close', 'N/A'):.2f}")
        print(f"Volume:     {latest.get('Volume', 'N/A'):.0f}")
        
        # Moving Averages
        print("\n=== MOVING AVERAGES ===")
        print(f"SMA Fast ({self.sma_fast_period}): {latest.get('sma_fast', 'N/A'):.2f}")
        print(f"SMA Slow ({self.sma_slow_period}): {latest.get('sma_slow', 'N/A'):.2f}")
        
        # ADX and DI
        print("\n=== TREND STRENGTH ===")
        print(f"ADX ({self.adx_period}):        {latest.get('adx', 'N/A'):.2f}")
        print(f"ADXR ({self.adxr_period}):       {latest.get('adxr', 'N/A'):.2f}")
        print(f"DI+ ({self.adx_period}):        {latest.get('di_plus', 'N/A'):.2f}")
        print(f"DI- ({self.adx_period}):        {latest.get('di_minus', 'N/A'):.2f}")
        
        # RSI
        print("\n=== MOMENTUM ===")
        print(f"RSI ({self.rsi_period}):        {latest.get('rsi', 'N/A'):.2f}")
        print(f"RSI MA ({self.rsi_ma_period}):   {latest.get('rsi_ma', 'N/A'):.2f}")
        
        # ATR
        print("\n=== VOLATILITY ===")
        print(f"ATR ({self.atr_period}):       {latest.get('atr', 'N/A'):.2f}")
        
        # SuperTrend
        print("\n=== SUPER TREND ===")
        st_long_dir = "UP" if latest.get('st_long_direction', 0) > 0 else "DOWN"
        st_short_dir = "UP" if latest.get('st_short_direction', 0) > 0 else "DOWN"
        
        print(f"Long ST ({self.st_long_period}x{self.st_long_multiplier}): {latest.get('st_long', 'N/A'):.2f} [{st_long_dir}]")
        print(f"Short ST ({self.st_short_period}x{self.st_short_multiplier}): {latest.get('st_short', 'N/A'):.2f} [{st_short_dir}]")
        
        # For backward compatibility
        print(f"SuperTrend ({self.st_long_period}x{self.st_long_multiplier}): {latest.get('supertrend', 'N/A'):.2f} [{st_long_dir}]")
        
        # CPR
        print("\n=== CPR ===")
        print(f"Pivot:      {latest.get('cpr_pivot', 'N/A'):.2f}")
        print(f"Top Central: {latest.get('cpr_tc', 'N/A'):.2f}")
        print(f"Bottom Central: {latest.get('cpr_bc', 'N/A'):.2f}")
        print(f"Inside CPR: {latest.get('inside_cpr', 'N/A')}")
        
        # Signal Status
        print("\n=== SIGNAL STATUS ===")
        print(f"Trend:       {'Bullish' if latest.get('trend', 0) > 0 else 'Bearish' if latest.get('trend', 0) < 0 else 'Neutral'}")
        print(f"Signal:      {latest.get('signal', 'N/A')}")
        print("\n" + "=" * 50 + "\n")
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the strategy.
        """     
        # Ensure required columns exist with correct case
        if 'Open' not in df.columns and 'open' in df.columns:
            df['Open'] = df['open']        
        if 'Close' not in df.columns and 'close' in df.columns:
            df['Close'] = df['close']
        if 'High' not in df.columns and 'high' in df.columns:
            df['High'] = df['high']
        if 'Low' not in df.columns and 'low' in df.columns:
            df['Low'] = df['low']
        if 'Volume' not in df.columns and 'volume' in df.columns:
            df['Volume'] = df['volume']
            
        # Calculate SMAs using periods from config
        df['sma_fast'] = ta.sma(df['Close'], length=self.sma_fast_period)
        df['sma_slow'] = ta.sma(df['Close'], length=self.sma_slow_period)
       
        # Calculate ADX and DI indicators using TA-Lib for better MT5 compatibility
        import talib
        df['adx'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=self.adx_period)
        df['di_plus'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=self.adx_period)
        df['di_minus'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=self.adx_period)
        # Calculate ADXR using the dedicated method
        df['adxr'] = self.calculate_adxr(df, period=self.adxr_period)

        # Calculate RSI and RSI MA using TA-Lib for better MT5 compatibility
        df['rsi'] = talib.RSI(df['Close'], timeperiod=self.rsi_period)
        df['rsi_ma'] = talib.SMA(df['rsi'], timeperiod=self.rsi_ma_period)
        
        # Calculate ATR
        df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=self.atr_period)
        
        # Calculate Long SuperTrend (tighter with higher multiplier)
        st_long = ta.supertrend(df['High'], df['Low'], df['Close'], 
                              length=self.st_long_period, 
                              multiplier=self.st_long_multiplier)
        df['st_long'] = st_long[f'SUPERT_{self.st_long_period}_{self.st_long_multiplier}']
        df['st_long_direction'] = st_long[f'SUPERTd_{self.st_long_period}_{self.st_long_multiplier}']
        
        # Calculate Short SuperTrend (looser with lower multiplier)
        st_short = ta.supertrend(df['High'], df['Low'], df['Close'], 
                               length=self.st_short_period, 
                               multiplier=self.st_short_multiplier)
        df['st_short'] = st_short[f'SUPERT_{self.st_short_period}_{self.st_short_multiplier}']
        df['st_short_direction'] = st_short[f'SUPERTd_{self.st_short_period}_{self.st_short_multiplier}']
        
        # For backward compatibility
        df['supertrend'] = df['st_long']
        df['supertrend_direction'] = df['st_long_direction']
        
        # Calculate CPR (Central Pivot Range) indicator
        df = self._calculate_cpr(df)
        
        # Calculate if price is above/below CPR levels
        df['above_tc'] = df['Close'] > df['cpr_tc']
        df['below_bc'] = df['Close'] < df['cpr_bc']
        df['inside_cpr'] = ~df['above_tc'] & ~df['below_bc']

        # Calculate HTF indicators if multi-timeframe logic is enabled
        if self._trade_logic_base_multi_timeframe:
        #     df.rename(columns={
        #         'open': 'Open',
        #         'high': 'High',
        #         'low': 'Low',
        #         'close': 'Close',
        #         'tick_volume': 'Volume',
        #         'volume': 'Volume'
        #     }, inplace=True)
            df = self.calculate_indicators_htf(df)
        return df

    def calculate_indicators_htf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Higher Time Frame (HTF) indicators for multi-timeframe analysis.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with HTF indicators added
        """
        if len(df) < 50:  # Need sufficient data for HTF analysis
            logger.warning("Insufficient data for HTF indicator calculation")
            return df

        try:
            # Resample to higher timeframe
            htf_df = self._resample_to_htf(df, self._high_time_frame)

            if len(htf_df) < 20:  # Need minimum HTF bars
                logger.warning(f"Insufficient HTF data after resampling to {self._high_time_frame}")
                return df

            # Calculate HTF indicators
            htf_df = self._calculate_htf_indicators(htf_df)

            # Merge HTF indicators back to original timeframe
            df = self._merge_htf_indicators(df, htf_df)

            logger.debug(f"HTF indicators calculated for {self._high_time_frame} timeframe")
            return df

        except Exception as e:
            logger.error(f"Error calculating HTF indicators: {e}")
            return df

    def _resample_to_htf(self, df: pd.DataFrame, htf: str) -> pd.DataFrame:
        """Resample dataframe to higher timeframe."""
        timeframe_map = {
            'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
            'H1': '1H', 'H4': '4H', 'D1': '1D'
        }

        if htf not in timeframe_map:
            logger.warning(f"Unsupported HTF: {htf}, using M5")
            htf = 'M5'

        freq = timeframe_map[htf]

        # Resample OHLC data
        htf_df = df.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        return htf_df

    def _calculate_htf_indicators(self, htf_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all HTF indicators."""
        # SMA indicators
        htf_df['sma_fast_htf'] = ta.sma(htf_df['Close'], length=self.sma_fast_period)
        htf_df['sma_slow_htf'] = ta.sma(htf_df['Close'], length=self.sma_slow_period)

        # SuperTrend indicators
        st_long = ta.supertrend(
            high=htf_df['High'],
            low=htf_df['Low'],
            close=htf_df['Close'],
            length=self.st_long_period,
            multiplier=self.st_long_multiplier
        )

        st_short = ta.supertrend(
            high=htf_df['High'],
            low=htf_df['Low'],
            close=htf_df['Close'],
            length=self.st_short_period,
            multiplier=self.st_short_multiplier
        )

        if st_long is not None and not st_long.empty:
            htf_df['super_trend_long_htf'] = st_long.iloc[:, 0]
            htf_df['super_trend_long_direction_htf'] = st_long.iloc[:, 1]

        if st_short is not None and not st_short.empty:
            htf_df['super_trend_short_htf'] = st_short.iloc[:, 0]
            htf_df['super_trend_short_direction_htf'] = st_short.iloc[:, 1]

        # RSI indicators
        htf_df['rsi_htf'] = ta.rsi(htf_df['Close'], length=self.rsi_period)
        htf_df['rsi_ma_htf'] = ta.sma(htf_df['rsi_htf'], length=self.rsi_ma_period)

        # ADX family indicators
        adx_data = ta.adx(
            high=htf_df['High'],
            low=htf_df['Low'],
            close=htf_df['Close'],
            length=self.adx_period
        )

        if adx_data is not None and not adx_data.empty:
            htf_df['adx_htf'] = adx_data[f'ADX_{self.adx_period}']
            htf_df['di_plus_htf'] = adx_data[f'DMP_{self.adx_period}']
            htf_df['di_minus_htf'] = adx_data[f'DMN_{self.adx_period}']

            ## Calculate ADXR (ADX Rating)
            #if len(htf_df['adx_htf'].dropna()) >= self.adx_period:
            #    htf_df['adxr_htf'] = (htf_df['adx_htf'] + htf_df['adx_htf'].shift(self.adx_period - 1)) / 2
            # Calculate SMA of ADX (instead of ADXR)
            if len(htf_df['adx_htf'].dropna()) >= self.adxr_period:
                htf_df['adxr_htf'] = htf_df['adx_htf'].rolling(window=self.adxr_period).mean()
                
        return htf_df

    def _merge_htf_indicators(self, df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
        """Merge HTF indicators back to original timeframe using forward fill."""
        # Create a mapping of HTF timestamps to original timestamps
        htf_indicators = [
            'sma_fast_htf', 'sma_slow_htf', 'super_trend_long_htf', 'super_trend_short_htf',
            'super_trend_long_direction_htf', 'super_trend_short_direction_htf',
            'rsi_htf', 'rsi_ma_htf', 'adx_htf', 'di_plus_htf', 'di_minus_htf', 'adxr_htf'
        ]

        # Reindex HTF data to original timeframe and forward fill
        for indicator in htf_indicators:
            if indicator in htf_df.columns:
                try:
                    # Reindex to original timeframe and forward fill
                    htf_series = htf_df[indicator].reindex(df.index).ffill().infer_objects(copy=False)
                    df[indicator] = htf_series
                except Exception as e:
                    logger.warning(f"Error merging HTF indicator {indicator}: {e}")
                    # Create empty series with NaN values as fallback
                    df[indicator] = pd.Series(index=df.index, dtype=float)

        return df

    # def _init_market_structure_params(self):
    #     """Initialize market structure analysis parameters."""
    #     self._ms_lookback = self._get_param('ms_lookback', int)
    #     self._ms_vol_window = self._get_param('ms_volatility_window', int)
    #     self._ms_adx_thresh = self._get_param('ms_adx_threshold', int)
    #     self._ms_min_confidence = self._get_param('ms_min_confidence', float)
    #     self._ms_use_volume = self._get_param('ms_use_volume', bool)
    #     self._ms_smoothing = self._get_param('ms_smoothing', int)
    #     self._ms_min_trend_bars = self._get_param('ms_min_trend_bars', int)
    #     self._ms_req_conditions = self._get_param('ms_required_conditions', int)
    #     self._ms_use_htf = self._get_param('ms_enable_htf_confirmation', bool)

    # def _analyze_market_structure(self, df):
    #     """
    #     Analyze market structure for the given dataframe.
    #     Returns: (signal, confidence)
    #     """
    #     if not self._trade_logic_marketstructure or len(df) < self._ms_min_trend_bars * 2:
    #         return "NO_SIGNAL", 0.0
            
    #     # Apply smoothing - handle case sensitivity in column names
    #     df = df.copy()
        
    #     # Get the correct case for column names
    #     actual_columns = {col.lower(): col for col in df.columns}
    #     high_col = actual_columns.get('high', 'high')
    #     low_col = actual_columns.get('low', 'low')
    #     close_col = actual_columns.get('close', 'close')
        
    #     # Create smoothed versions of OHLC data
    #     for col, actual_col in [(high_col, 'high'), (low_col, 'low'), (close_col, 'close')]:
    #         if col in df.columns:  # Only process if column exists
    #             df[f'{actual_col}_smooth'] = df[col].rolling(self._ms_smoothing).mean()
        
    #     # Adaptive lookback based on volatility
    #     if close_col in df.columns:
    #         returns = df[close_col].pct_change().dropna()
    #         if len(returns) >= self._ms_vol_window:
    #             vol = returns.rolling(self._ms_vol_window).std().iloc[-1]
    #             lookback = min(
    #                 max(int(self._ms_lookback * (1 + vol * 100)), 
    #                     self._ms_min_trend_bars * 2),
    #                 len(df) // 2
    #             )
    #         else:
    #             lookback = self._ms_lookback
    #     else:
    #         lookback = self._ms_lookback
            
    #     recent = df.tail(lookback).copy()
        
    #     # 1. Trend Analysis - use the correct smoothed column names
    #     highs = recent.get('high_smooth', recent.get(high_col, pd.Series())).values
    #     lows = recent.get('low_smooth', recent.get(low_col, pd.Series())).values
    #     close_values = recent.get('close_smooth', recent.get(close_col, pd.Series())).values
        
    #     if len(highs) == 0 or len(lows) == 0 or len(close_values) == 0:
    #         return "NO_SIGNAL", 0.0
            
    #     x = np.arange(len(highs))
        
    #     def robust_trend(y):
    #         try:
    #             if len(y) < 2:
    #                 return 0
    #             slope = np.polyfit(x, y, 1)[0]
    #             return slope / (np.mean(y) or 1)  # Normalized slope, handle division by zero
    #         except Exception as e:
    #             print(f"Error in robust_trend: {e}")
    #             return 0
        
    #     trend_strength = {
    #         'high': robust_trend(highs),
    #         'low': robust_trend(lows),
    #         'close': robust_trend(close_values)
    #     }
        
    #     # 2. Range Detection
    #     range_conditions = []
        
    #     # Price channel containment
    #     channel_width = highs.max() - lows.min()
    #     if channel_width > 0:
    #         in_channel = ((recent[close_col] >= lows.min()) & 
    #                      (recent[close_col] <= highs.max() + channel_width*0.1))
    #         range_conditions.append(in_channel.mean() > 0.7)
        
    #     # Determine trend direction
    #     uptrend = trend_strength['close'] > 0.0005
    #     downtrend = trend_strength['close'] < -0.0005
    #     signal, confidence = "NO_SIGNAL", 0.5
        
    #     if uptrend and trend_strength['high'] > 0 and trend_strength['low'] > 0:
    #         signal, confidence = "BULLISH", 0.6 + min(0.3, abs(trend_strength['close']) * 100)
    #     elif downtrend and trend_strength['high'] < 0 and trend_strength['low'] < 0:
    #         signal, confidence = "BEARISH", 0.6 + min(0.3, abs(trend_strength['close']) * 100)
        
    #     # Higher timeframe confirmation
    #     if self._ms_use_htf and len(df) >= lookback * 2:
    #         try:
    #             htf_signal, _ = self._analyze_market_structure(df.iloc[:-lookback//2])
    #             if htf_signal != signal and htf_signal != "NO_SIGNAL":
    #                 confidence *= 0.8  # Reduce confidence if HTF disagrees
    #         except Exception as e:
    #             print(f"Error in HTF analysis: {e}")
        
    #     return signal, max(0.1, min(0.99, confidence))

    # def _check_market_structure(self, current_bar, prev_bar, check_entries):
    #     """Check market structure conditions for entry."""
    #     if not self._trade_logic_marketstructure:
    #         return False
            
    #     # Get market structure signal from the last calculated bar
    #     if not hasattr(self, '_last_ms_signal') or not hasattr(self, '_last_ms_confidence'):
    #         return False
            
    #     signal = self._last_ms_signal
    #     confidence = self._last_ms_confidence
        
    #     if confidence < self._ms_min_confidence:
    #         return False
            
    #     if signal == "BULLISH":
    #         check_entries['ms_bullish'] = True
    #         return True
    #     elif signal == "BEARISH":
    #         check_entries['ms_bearish'] = True
    #         return True
            
    #     return False
    
    # def _check_base_logic_long(self, current_bar, prev_bar, check_entries):
    #     """Check base conditions for long entry with confluence requirements."""
    #     if not self._trade_logic_base or not check_entries:
    #         return False

    #     # Market condition filters
    #     if not self._check_market_conditions(current_bar):
    #         return False

    #     # Basic trend conditions
    #     sma_fast = current_bar.get('sma_fast', 0)
    #     sma_slow = current_bar.get('sma_slow', 0)
    #     price = current_bar['Close']

    #     # Trend alignment: Price above both SMAs and fast SMA above slow SMA
    #     trend_bullish = price > sma_fast > sma_slow
    #     if not trend_bullish:
    #         return False

    #     # SuperTrend confirmation
    #     st_long = current_bar.get('st_long', 0)
    #     st_direction = current_bar.get('st_long_direction', 0)
    #     supertrend_bullish = st_direction == 1 and price > st_long

    #     # Momentum confluence
    #     momentum_ok = True
    #     if self._require_momentum_confluence:
    #         momentum_ok = self._check_momentum_confluence_long(current_bar)

    #     # Volatility filter
    #     volatility_ok = True
    #     if self._volatility_filter:
    #         atr = current_bar.get('atr', 0)
    #         volatility_ok = self._min_atr_for_entry <= atr <= self._max_atr_for_entry

    #     return trend_bullish and supertrend_bullish and momentum_ok and volatility_ok

    # def _check_base_logic_short(self, current_bar, prev_bar, check_entries):
    #     """Check base conditions for short entry with confluence requirements."""
    #     if not self._trade_logic_base or not check_entries:
    #         return False

    #     # Market condition filters
    #     if not self._check_market_conditions(current_bar):
    #         return False

    #     # Basic trend conditions
    #     sma_fast = current_bar.get('sma_fast', 0)
    #     sma_slow = current_bar.get('sma_slow', 0)
    #     price = current_bar['Close']

    #     # Trend alignment: Price below both SMAs and fast SMA below slow SMA
    #     trend_bearish = price < sma_fast < sma_slow
    #     if not trend_bearish:
    #         return False

    #     # SuperTrend confirmation
    #     st_short = current_bar.get('st_short', 0)
    #     st_direction = current_bar.get('st_short_direction', 0)
    #     supertrend_bearish = st_direction == -1 and price < st_short

    #     # Momentum confluence
    #     momentum_ok = True
    #     if self._require_momentum_confluence:
    #         momentum_ok = self._check_momentum_confluence_short(current_bar)

    #     # Volatility filter
    #     volatility_ok = True
    #     if self._volatility_filter:
    #         atr = current_bar.get('atr', 0)
    #         volatility_ok = self._min_atr_for_entry <= atr <= self._max_atr_for_entry

    #     return trend_bearish and supertrend_bearish and momentum_ok and volatility_ok

    def _check_base_multi_timeframe_long(self, current_bar, prev_bar, check_entries):
        """Check multi-timeframe base conditions for long entry with signal strength calculation."""
        if not self._trade_logic_base_multi_timeframe or not check_entries:
            return False

        # Market condition filters
        if not self._check_market_conditions(current_bar):
            return False

        # Calculate signal strength
        strength_data = self._calculate_signal_strength_long(current_bar, prev_bar)

        # Check if strength meets minimum threshold
        signal_ok = strength_data['strength'] >= self._min_signal_strength_pct

        # Enhanced logging with detailed signal strength breakdown for every candle
        if 'detailed_conditions' in strength_data:
            # Get timestamp for candle identification
            candle_time = current_bar.name if hasattr(current_bar, 'name') else 'Unknown'

            # Check if we've already logged this candle to avoid duplicates
            log_key = f"LONG_{candle_time}"
            if not hasattr(self, '_logged_candles'):
                self._logged_candles = set()

            if log_key not in self._logged_candles:
                self._logged_candles.add(log_key)

                logger.info(f"[SIGNAL ANALYSIS] CANDLE: {candle_time}")
                logger.info(f"[LONG] Signal Strength: {strength_data['strength']:.1f}% | "
                           f"Threshold: {self._min_signal_strength_pct}% | "
                           f"Signal: {'PASS' if signal_ok else 'FAIL'}")

                if strength_data.get('achieved_weight') and strength_data.get('total_weight'):
                    logger.info(f"[WEIGHT] Score: {strength_data['achieved_weight']}/{strength_data['total_weight']} "
                               f"({strength_data['achieved_weight']/strength_data['total_weight']*100:.1f}%)")

                logger.info(f"[CONDITIONS] Met: {strength_data['favorable_count']}/{strength_data['total_conditions']} | "
                           f"Required: {'PASS' if strength_data.get('required_conditions_met', True) else 'FAIL'}")

                # Detailed condition breakdown with weights and contribution
                logger.info("[BREAKDOWN] DETAILED CONDITION ANALYSIS:")

                # Group and display conditions with their contributions
                condition_groups = {
                    '[LTF-TREND]': ['ltf_price_above_sma_fast', 'ltf_sma_fast_above_slow', 'ltf_supertrend_bullish'],
                    '[LTF-MOMENTUM]': ['ltf_rsi_above_rsi_ma', 'ltf_adx_strength', 'ltf_adx_above_adxr', 'ltf_di_bullish'],
                    '[HTF-TREND]': ['htf_price_above_sma_fast', 'htf_sma_fast_above_slow', 'htf_supertrend_bullish'],
                    '[HTF-MOMENTUM]': ['htf_rsi_above_rsi_ma', 'htf_adx_strength', 'htf_adx_above_adxr', 'htf_di_bullish']
                }

                for group_name, conditions in condition_groups.items():
                    group_weight = 0
                    group_achieved = 0
                    group_details = []

                    for condition in conditions:
                        if condition in strength_data['detailed_conditions']:
                            is_met = strength_data['detailed_conditions'][condition]
                            config = strength_data['condition_config'].get(condition, {})
                            weight = config.get('weight', 0)
                            required = config.get('required', False)

                            group_weight += weight
                            if is_met:
                                group_achieved += weight

                            status = '[PASS]' if is_met else '[FAIL]'
                            req_flag = '[REQ]' if required else '[OPT]'
                            contribution = f"+{weight}" if is_met else f"0/{weight}"

                            group_details.append(f"    {status} {condition.replace('_', ' ').title()}: {contribution} {req_flag}")

                    group_pct = (group_achieved / group_weight * 100) if group_weight > 0 else 0
                    logger.info(f"  {group_name}: {group_achieved}/{group_weight} ({group_pct:.1f}%)")
                    for detail in group_details:
                        logger.info(detail)

                logger.info("=" * 80)
        else:
            # Fallback for old format
            conditions_str = ", ".join([f"{k}={v}" for k, v in strength_data.get('conditions', {}).items()])
            logger.info(f"Multi-TF LONG: Strength={strength_data['strength']:.1f}% "
                       f"[{strength_data['favorable_count']}/{strength_data['total_conditions']}] "
                       f"Min={self._min_signal_strength_pct}% Signal={signal_ok}")
            logger.info(f"Multi-TF LONG Conditions: {conditions_str}")

        return signal_ok

    def _check_base_multi_timeframe_short(self, current_bar, prev_bar, check_entries):
        """Check multi-timeframe base conditions for short entry with signal strength calculation."""
        if not self._trade_logic_base_multi_timeframe or not check_entries:
            return False

        # Market condition filters
        if not self._check_market_conditions(current_bar):
            return False

        # Calculate signal strength
        strength_data = self._calculate_signal_strength_short(current_bar, prev_bar)

        # Check if strength meets minimum threshold
        signal_ok = strength_data['strength'] >= self._min_signal_strength_pct

        # Enhanced logging with detailed signal strength breakdown for every candle
        if 'detailed_conditions' in strength_data:
            # Get timestamp for candle identification
            candle_time = current_bar.name if hasattr(current_bar, 'name') else 'Unknown'

            # Check if we've already logged this candle to avoid duplicates
            log_key = f"SHORT_{candle_time}"
            if not hasattr(self, '_logged_candles'):
                self._logged_candles = set()

            if log_key not in self._logged_candles:
                self._logged_candles.add(log_key)

                logger.info(f"[SIGNAL ANALYSIS] CANDLE: {candle_time}")
                logger.info(f"[SHORT] Signal Strength: {strength_data['strength']:.1f}% | "
                           f"Threshold: {self._min_signal_strength_pct}% | "
                           f"Signal: {'PASS' if signal_ok else 'FAIL'}")

                if strength_data.get('achieved_weight') and strength_data.get('total_weight'):
                    logger.info(f"[WEIGHT] Score: {strength_data['achieved_weight']}/{strength_data['total_weight']} "
                               f"({strength_data['achieved_weight']/strength_data['total_weight']*100:.1f}%)")

                logger.info(f"[CONDITIONS] Met: {strength_data['favorable_count']}/{strength_data['total_conditions']} | "
                           f"Required: {'PASS' if strength_data.get('required_conditions_met', True) else 'FAIL'}")

                # Detailed condition breakdown with weights and contribution
                logger.info("[BREAKDOWN] DETAILED CONDITION ANALYSIS:")

                # Group and display conditions with their contributions
                condition_groups = {
                    '[LTF-TREND]': ['ltf_price_below_sma_fast', 'ltf_sma_fast_below_slow', 'ltf_supertrend_bearish'],
                    '[LTF-MOMENTUM]': ['ltf_rsi_below_rsi_ma', 'ltf_adx_strength', 'ltf_adx_above_adxr', 'ltf_di_bearish'],
                    '[HTF-TREND]': ['htf_price_below_sma_fast', 'htf_sma_fast_below_slow', 'htf_supertrend_bearish'],
                    '[HTF-MOMENTUM]': ['htf_rsi_below_rsi_ma', 'htf_adx_strength', 'htf_adx_above_adxr', 'htf_di_bearish']
                }

                for group_name, conditions in condition_groups.items():
                    group_weight = 0
                    group_achieved = 0
                    group_details = []

                    for condition in conditions:
                        if condition in strength_data['detailed_conditions']:
                            is_met = strength_data['detailed_conditions'][condition]
                            config = strength_data['condition_config'].get(condition, {})
                            weight = config.get('weight', 0)
                            required = config.get('required', False)

                            group_weight += weight
                            if is_met:
                                group_achieved += weight

                            status = '[PASS]' if is_met else '[FAIL]'
                            req_flag = '[REQ]' if required else '[OPT]'
                            contribution = f"+{weight}" if is_met else f"0/{weight}"

                            group_details.append(f"    {status} {condition.replace('_', ' ').title()}: {contribution} {req_flag}")

                    group_pct = (group_achieved / group_weight * 100) if group_weight > 0 else 0
                    logger.info(f"  {group_name}: {group_achieved}/{group_weight} ({group_pct:.1f}%)")
                    for detail in group_details:
                        logger.info(detail)

                logger.info("=" * 80)
        else:
            # Fallback for old format
            conditions_str = ", ".join([f"{k}={v}" for k, v in strength_data.get('conditions', {}).items()])
            logger.info(f"Multi-TF SHORT: Strength={strength_data['strength']:.1f}% "
                       f"[{strength_data['favorable_count']}/{strength_data['total_conditions']}] "
                       f"Min={self._min_signal_strength_pct}% Signal={signal_ok}")
            logger.info(f"Multi-TF SHORT Conditions: {conditions_str}")

        return signal_ok

    def _calculate_signal_strength_long(self, current_bar, prev_bar):
        """
        Calculate signal strength for long entries based on detailed condition confluence.

        Returns:
            dict: Signal strength data with percentage, counts, and detailed condition breakdown
        """
        # Define internal condition weights and requirements (hidden from user)
        condition_config = {
            # LTF Trend Conditions
            'ltf_price_above_sma_fast': {'weight': 10, 'required': True},
            'ltf_sma_fast_above_slow': {'weight': 10, 'required': True},
            'ltf_supertrend_bullish': {'weight': 10, 'required': True},

            # LTF Momentum Conditions
            'ltf_rsi_above_rsi_ma': {'weight': 10, 'required': False},  # Changed to optional
            'ltf_adx_strength': {'weight': 10, 'required': True},
            'ltf_adx_above_adxr': {'weight': 10, 'required': False},    
            'ltf_di_bullish': {'weight': 10, 'required': True},        

            # HTF Trend Conditions
            #'htf_price_above_sma_fast': {'weight': 10, 'required': False},
            #'htf_sma_fast_above_slow': {'weight': 10, 'required': False},
            #'htf_supertrend_bullish': {'weight': 10, 'required': False},

            # HTF Momentum Conditions
            #'htf_rsi_above_rsi_ma': {'weight': 10, 'required': False},  # Changed to optional
            #'htf_adx_strength': {'weight': 10, 'required': False},
            #'htf_adx_above_adxr': {'weight': 10, 'required': False},
            #'htf_di_bullish': {'weight': 10, 'required': False}
        }

        # Evaluate detailed conditions
        detailed_conditions = self._evaluate_detailed_conditions_long(current_bar, prev_bar)

        # Count how many conditions (required + optional) are passing
        total_conditions = len(detailed_conditions)
        passed_conditions = sum(1 for v in detailed_conditions.values() if v)

        # Calculate strength as a percentage of all conditions
        strength = (passed_conditions / total_conditions) * 100 if total_conditions > 0 else 0.0

        # Check required conditions first
        required_conditions_met = all(
            detailed_conditions[condition]
            for condition, config in condition_config.items()
            if config['required']
        )

        if not required_conditions_met:
            # If required conditions not met, return 0 strength
            return {
                'strength': strength,
                'favorable_count': sum(detailed_conditions.values()),
                'total_conditions': len(detailed_conditions),
                'detailed_conditions': detailed_conditions,
                'required_conditions_met': False,
                'condition_config': condition_config,
                'weighted': True
            }

        # Calculate weighted strength for all conditions
        total_weight = sum(config['weight'] for config in condition_config.values())
        achieved_weight = sum(
            config['weight'] for condition, config in condition_config.items()
            if detailed_conditions[condition]
        )

        weighted_strength = (achieved_weight / total_weight) * 100 if total_weight > 0 else 0.0

        return {
            'strength': strength,
            'weighted_strength': weighted_strength if required_conditions_met else 0.0,  # Only show weighted if all required met
            'favorable_count': passed_conditions,
            'total_conditions': total_conditions,
            'detailed_conditions': detailed_conditions,
            'required_conditions_met': required_conditions_met,
            'condition_config': condition_config,
            'achieved_weight': achieved_weight,
            'total_weight': total_weight,
            'weighted': True
        }

    def _evaluate_detailed_conditions_long(self, current_bar, prev_bar):
        """Evaluate detailed individual conditions for long entries."""
        # Get current bar data (not previous bar)
        current_price = current_bar['Close']
        current_sma_fast = current_bar.get('sma_fast', 0)
        current_sma_slow = current_bar.get('sma_slow', 0)
        current_st_long = current_bar.get('st_long', 0)
        current_st_direction = current_bar.get('st_long_direction', 0)
        current_rsi = current_bar.get('rsi', 50)
        current_rsi_ma = current_bar.get('rsi_ma', 50)
        current_adx = current_bar.get('adx', 0)
        current_adxr = current_bar.get('adxr', 0)
        current_di_plus = current_bar.get('di_plus', 0)
        current_di_minus = current_bar.get('di_minus', 0)

        # HTF data
        # current_sma_fast_htf = current_bar.get('sma_fast_htf', 0)
        # current_sma_slow_htf = current_bar.get('sma_slow_htf', 0)
        # current_st_long_htf = current_bar.get('super_trend_long_htf', 0)
        # current_st_direction_htf = current_bar.get('super_trend_long_direction_htf', 0)
        # current_rsi_htf = current_bar.get('rsi_htf', 50)
        # current_rsi_ma_htf = current_bar.get('rsi_ma_htf', 50)
        # current_adx_htf = current_bar.get('adx_htf', 0)
        # current_adxr_htf = current_bar.get('adxr_htf', 0)
        # current_di_plus_htf = current_bar.get('di_plus_htf', 0)
        # current_di_minus_htf = current_bar.get('di_minus_htf', 0)

        return {
            # LTF Trend Conditions
            'ltf_price_above_sma_fast': current_price > current_sma_fast,
            'ltf_sma_fast_above_slow': current_sma_fast > current_sma_slow,
            'ltf_supertrend_bullish': current_st_direction == 1 and current_price > current_st_long,

            # LTF Momentum Conditions
            'ltf_rsi_above_rsi_ma': current_rsi > current_rsi_ma,
            'ltf_adx_strength': current_adx >= self._min_adx_for_entry,
            'ltf_adx_above_adxr': current_adx > current_adxr,
            'ltf_di_bullish': current_di_plus > current_di_minus,

            # HTF Trend Conditions
            #'htf_price_above_sma_fast': current_price > current_sma_fast_htf if current_sma_fast_htf is not None else False,
            #'htf_sma_fast_above_slow': current_sma_fast_htf > current_sma_slow_htf if (current_sma_fast_htf is not None and current_sma_slow_htf is not None) else False,
            #'htf_supertrend_bullish': current_st_direction_htf == 1 and current_price > current_st_long_htf if current_st_long_htf is not None else False,

            # HTF Momentum Conditions
            #'htf_rsi_above_rsi_ma': current_rsi_htf > current_rsi_ma_htf,
            #'htf_adx_strength': current_adx_htf >= self._min_adx_for_entry,
            #'htf_adx_above_adxr': current_adx_htf > current_adxr_htf,
            #'htf_di_bullish': current_di_plus_htf > current_di_minus_htf
        }

    def _calculate_signal_strength_short(self, current_bar, prev_bar):
        """
        Calculate signal strength for short entries based on detailed condition confluence.

        Returns:
            dict: Signal strength data with percentage, counts, and detailed condition breakdown
        """
        # Define internal condition weights and requirements (hidden from user)
        condition_config = {
            # LTF Trend Conditions
            'ltf_price_below_sma_fast': {'weight': 10, 'required': True},
            'ltf_sma_fast_below_slow': {'weight': 10, 'required': True},
            'ltf_supertrend_bearish': {'weight': 10, 'required': True},

            # LTF Momentum Conditions
            'ltf_rsi_below_rsi_ma': {'weight': 10, 'required': False},  # Changed to optional
            'ltf_adx_strength': {'weight': 10, 'required': True},
            'ltf_adx_above_adxr': {'weight': 10, 'required': False},   
            'ltf_di_bearish': {'weight': 10, 'required': True},        

            # HTF Trend Conditions
            #'htf_price_below_sma_fast': {'weight': 10, 'required': False},
            #'htf_sma_fast_below_slow': {'weight': 10, 'required': False},
            #'htf_supertrend_bearish': {'weight': 10, 'required': False},

            # HTF Momentum Conditions
            #'htf_rsi_below_rsi_ma': {'weight': 10, 'required': False},  # Changed to optional
            #'htf_adx_strength': {'weight': 10, 'required': False},
            #'htf_adx_above_adxr': {'weight': 10, 'required': False},
            #'htf_di_bearish': {'weight': 10, 'required': False}
        }

        # Evaluate detailed conditions
        detailed_conditions = self._evaluate_detailed_conditions_short(current_bar, prev_bar)

        # Count how many conditions (required + optional) are passing
        total_conditions = len(detailed_conditions)
        passed_conditions = sum(1 for v in detailed_conditions.values() if v)
        strength = (passed_conditions / total_conditions) * 100 if total_conditions > 0 else 0.0

        # Check required conditions first
        required_conditions_met = all(
            detailed_conditions[condition]
            for condition, config in condition_config.items()
            if config['required']
        )

        if not required_conditions_met:
            # If required conditions not met, return 0 strength
            return {
                'strength': strength,
                'favorable_count': sum(detailed_conditions.values()),
                'total_conditions': len(detailed_conditions),
                'detailed_conditions': detailed_conditions,
                'required_conditions_met': False,
                'condition_config': condition_config,
                'weighted': True
            }

        # Calculate weighted strength for all conditions
        total_weight = sum(config['weight'] for config in condition_config.values())
        achieved_weight = sum(
            config['weight'] for condition, config in condition_config.items()
            if detailed_conditions[condition]
        )

        weighted_strength = (achieved_weight / total_weight) * 100 if total_weight > 0 else 0.0

        return {
            'strength': strength,
            'weighted_strength': weighted_strength if required_conditions_met else 0.0,  # Only show weighted if all required met
            'favorable_count': passed_conditions,
            'total_conditions': total_conditions,
            'detailed_conditions': detailed_conditions,
            'required_conditions_met': required_conditions_met,
            'condition_config': condition_config,
            'achieved_weight': achieved_weight,
            'total_weight': total_weight,
            'weighted': True
        }

    def _evaluate_detailed_conditions_short(self, current_bar, prev_bar):
        """Evaluate detailed individual conditions for short entries."""
        # Get current bar data (not previous bar)
        current_price = current_bar['Close']
        current_sma_fast = current_bar.get('sma_fast', 0)
        current_sma_slow = current_bar.get('sma_slow', 0)
        current_st_short = current_bar.get('st_short', 0)
        current_st_direction = current_bar.get('st_short_direction', 0)
        current_rsi = current_bar.get('rsi', 50)
        current_rsi_ma = current_bar.get('rsi_ma', 50)
        current_adx = current_bar.get('adx', 0)
        current_adxr = current_bar.get('adxr', 0)
        current_di_plus = current_bar.get('di_plus', 0)
        current_di_minus = current_bar.get('di_minus', 0)

        # HTF data
        # current_sma_fast_htf = current_bar.get('sma_fast_htf', 0)
        # current_sma_slow_htf = current_bar.get('sma_slow_htf', 0)
        # current_st_short_htf = current_bar.get('super_trend_short_htf', 0)
        # current_st_direction_htf = current_bar.get('super_trend_short_direction_htf', 0)
        # current_rsi_htf = current_bar.get('rsi_htf', 50)
        # current_rsi_ma_htf = current_bar.get('rsi_ma_htf', 50)
        # current_adx_htf = current_bar.get('adx_htf', 0)
        # current_adxr_htf = current_bar.get('adxr_htf', 0)
        # current_di_plus_htf = current_bar.get('di_plus_htf', 0)
        # current_di_minus_htf = current_bar.get('di_minus_htf', 0)

        # Helper function to safely compare values that might be None
        def safe_compare_lt(a, b):
            if a is None or b is None:
                return False
            return a < b
        
        def safe_compare_gt(a, b):
            if a is None or b is None:
                return False
            return a > b
        
        def safe_compare_gte(a, b):
            if a is None or b is None:
                return False
            return a >= b
    
        return {
            # LTF Trend Conditions
            'ltf_price_below_sma_fast': safe_compare_lt(current_price, current_sma_fast),
            'ltf_sma_fast_below_slow': safe_compare_lt(current_sma_fast, current_sma_slow),
            'ltf_supertrend_bearish': current_st_direction == -1 and safe_compare_lt(current_price, current_st_short),

            # LTF Momentum Conditions
            'ltf_rsi_below_rsi_ma': safe_compare_lt(current_rsi, current_rsi_ma),
            'ltf_adx_strength': safe_compare_gte(current_adx, self._min_adx_for_entry),
            'ltf_adx_above_adxr': safe_compare_gt(current_adx, current_adxr),
            'ltf_di_bearish': safe_compare_gt(current_di_minus, current_di_plus),

            # HTF Trend Conditions
            #'htf_price_below_sma_fast': safe_compare_lt(current_price, current_sma_fast_htf),
            #'htf_sma_fast_below_slow': safe_compare_lt(current_sma_fast_htf, current_sma_slow_htf),
            #'htf_supertrend_bearish': current_st_direction_htf == -1 and safe_compare_lt(current_price, current_st_short_htf),

            # HTF Momentum Conditions
            #'htf_rsi_below_rsi_ma': safe_compare_lt(current_rsi_htf, current_rsi_ma_htf),
            #'htf_adx_strength': safe_compare_gte(current_adx_htf, self._min_adx_for_entry),
            #'htf_adx_above_adxr': safe_compare_gt(current_adx_htf, current_adxr_htf),
            #'htf_di_bearish': safe_compare_gt(current_di_minus_htf, current_di_plus_htf)
        }

    def _check_market_conditions(self, current_bar):
        """
        Check if current bar's date/time is within trading hours, sessions, avoids news times, etc.

        Args:
            current_bar: pandas Series representing the current bar

        Returns:
            bool: True if market conditions are favorable, False otherwise
        """
        timestamp = current_bar.name
        logger.info(f"MARKET CONDITIONS CHECK for {timestamp}")
        
        # Check if it's within trading hours
        current_hour = current_bar.name.hour if hasattr(current_bar.name, 'hour') else 0
        logger.info(f"Trading hours check: Current hour: {current_hour}, Allowed range: {self._start_hour} to {self._end_hour}")
        if not (self._start_hour <= current_hour < self._end_hour):
            logger.info(f"FAILED: Current hour {current_hour} is outside trading hours ({self._start_hour}-{self._end_hour})")
            return False
        logger.info(f"PASSED: Current hour {current_hour} is within trading hours")

        # Check if it's within allowed market sessions
        if self._market_session_filter:
            # Check if current hour is within active trading sessions
            # London: 8-17 UTC, New York: 13-22 UTC, Asian: 0-9 UTC
            in_session = (0 <= current_hour <= 9 or 8 <= current_hour <= 17 or 13 <= current_hour <= 22)
            logger.info(f"Market session filter is ON. In active session: {in_session}")
            if not in_session:
                logger.info(f"FAILED: Current hour {current_hour} is not in any active trading session")
                return False
            logger.info(f"PASSED: Current hour {current_hour} is in an active trading session")

        # Check if it's near high-impact news time
        if self._avoid_news_times:
            current_minute = current_bar.name.minute if hasattr(current_bar.name, 'minute') else 0
            current_date = current_bar.name.date() if hasattr(current_bar.name, 'date') else None
            logger.info(f"News time check: Checking if {current_hour}:{current_minute} on {current_date} is near high-impact news")
            is_news_time = self._is_news_time(current_hour, current_minute, current_date)
            if is_news_time:
                logger.info(f"FAILED: Current time is near high-impact news event")
                return False
            logger.info(f"PASSED: Current time is not near any high-impact news event")

        # Check if it's a bank holiday or weekend
        logger.info(f"MARKET CONDITIONS CHECK - _avoid_bank_holidays flag is: {self._avoid_bank_holidays}")
        if self._avoid_bank_holidays:
            current_date = current_bar.name.date() if hasattr(current_bar.name, 'date') else None
            if current_date:
                logger.info(f"CHECKING DATE: {current_date}, month: {current_date.month}, day: {current_date.day}, weekday: {current_date.weekday()}")
                if self._use_forexfactory:
                    logger.info(f"Using ForexFactory for holiday detection")
                    if self._get_forexfactory_holidays(current_date):
                        logger.info(f"ForexFactory holiday detected: {current_date} - skipping trading")
                        return False
                else:
                    logger.info(f"Using internal holiday detection logic")
                    is_holiday = self._is_bank_holiday(current_date)
                    logger.info(f"Holiday check result for {current_date}: {is_holiday}")
                    if is_holiday:
                        logger.info(f"Bank holiday detected: {current_date} - skipping trading")
                        return False
                    
                    if current_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
                        logger.info(f"Weekend detected: {current_date} (weekday: {current_date.weekday()}) - skipping trading")
                        return False

        return True

    def _is_news_time(self, hour, minute, current_date=None):
        """
        Check if current time is during high-impact news events.
        Uses ForexFactory in live mode, hardcoded times in testing mode.

        Args:
            hour: Current hour (UTC)
            minute: Current minute
            current_date: datetime.date object (optional, defaults to today)

        Returns:
            bool: True if high-impact news is scheduled, False otherwise
        """
        if current_date is None:
            current_date = datetime.now().date()

        # Use ForexFactory only in live trading mode
        if self._use_forexfactory:
            try:
                # Get news events from ForexFactory
                news_events = self._get_forexfactory_news(current_date)

                current_time_minutes = hour * 60 + minute

                for event in news_events:
                    try:
                        # Parse event time (format: "12:30pm" or "2:00am")
                        event_time_str = event['time'].lower().replace(' ', '')

                        if 'am' in event_time_str or 'pm' in event_time_str:
                            # Parse 12-hour format
                            time_part = event_time_str.replace('am', '').replace('pm', '')
                            if ':' in time_part:
                                event_hour, event_minute = map(int, time_part.split(':'))
                            else:
                                event_hour = int(time_part)
                                event_minute = 0

                            # Convert to 24-hour format
                            if 'pm' in event_time_str and event_hour != 12:
                                event_hour += 12
                            elif 'am' in event_time_str and event_hour == 12:
                                event_hour = 0
                        else:
                            # Assume 24-hour format
                            if ':' in event_time_str:
                                event_hour, event_minute = map(int, event_time_str.split(':'))
                            else:
                                event_hour = int(event_time_str)
                                event_minute = 0

                        event_time_minutes = event_hour * 60 + event_minute

                        # Check if current time is within 30 minutes before or after the event
                        time_buffer = 30  # minutes
                        if abs(current_time_minutes - event_time_minutes) <= time_buffer:
                            logger.debug(f"High-impact news detected: {event['title']} at {event['time']} ({event['currency']})")
                            return True

                    except (ValueError, KeyError) as e:
                        logger.debug(f"Error parsing news event time: {e}")
                        continue

                return False

            except Exception as e:
                logger.warning(f"Error checking ForexFactory news: {e}")
                # Fall through to hardcoded logic below

        # Use hardcoded logic (testing mode or ForexFactory fallback)
        major_news_hours = [
            (12, 0, 14, 30),   # 12:00-14:30 UTC - Major economic releases
            (19, 0, 20, 30),   # 19:00-20:30 UTC - FOMC and major announcements
        ]

        current_time_minutes = hour * 60 + minute

        for start_hour, start_min, end_hour, end_min in major_news_hours:
            start_time_minutes = start_hour * 60 + start_min
            end_time_minutes = end_hour * 60 + end_min

            if start_time_minutes <= current_time_minutes <= end_time_minutes:
                mode_desc = "live fallback" if self._use_forexfactory else "testing mode"
                logger.debug(f"Avoiding trade during news time ({mode_desc}): {hour:02d}:{minute:02d} UTC")
                return True

        return False

    def _is_backtesting_mode(self):
        """
        Detect if we're running in backtesting mode by checking the trading_mode parameter.
        If trading_mode is not explicitly set, falls back to date-based detection.

        Returns:
            bool: True if in backtesting mode, False if in live trading mode
        """
        try:
            # First check the trading_mode parameter if it's set
            if hasattr(self, '_trading_mode'):
                if self._trading_mode == 'live':
                    logger.debug("Live trading mode detected via trading_mode parameter")
                    return False
                elif self._trading_mode == 'backtesting':
                    logger.debug("Backtesting mode detected via trading_mode parameter")
                    return True
                # If trading_mode is 'testing' or something else, fall through to date-based detection

            # Fall back to date-based detection if trading_mode is not explicitly set
            from datetime import datetime, date
            today = date.today()

            # Check if date attributes exist (they might not be initialized yet)
            if not hasattr(self, '_from_date') or not hasattr(self, '_to_date'):
                logger.debug("Date attributes not initialized, defaulting to backtesting mode")
                return True

            # Parse the configured date range using internal attributes
            from_date_val = self._from_date
            to_date_val = self._to_date

            if isinstance(from_date_val, str):
                try:
                    # Try full datetime format first
                    start_date = datetime.strptime(from_date_val, '%Y-%m-%d %H:%M:%S').date()
                except ValueError:
                    # Fall back to date only format
                    start_date = datetime.strptime(from_date_val, '%Y-%m-%d').date()
            else:
                start_date = from_date_val.date() if hasattr(from_date_val, 'date') else from_date_val

            if isinstance(to_date_val, str):
                try:
                    # Try full datetime format first
                    end_date = datetime.strptime(to_date_val, '%Y-%m-%d %H:%M:%S').date()
                except ValueError:
                    # Fall back to date only format
                    end_date = datetime.strptime(to_date_val, '%Y-%m-%d').date()
            else:
                end_date = to_date_val.date() if hasattr(to_date_val, 'date') else to_date_val

            # If end date is more than 1 day ago, we're backtesting
            days_ago = (today - end_date).days
            is_backtesting = days_ago > 1

            logger.debug(f"Date-based backtesting detection: end_date={end_date}, today={today}, days_ago={days_ago}, is_backtesting={is_backtesting}")

            return is_backtesting

        except Exception as e:
            logger.debug(f"Error detecting backtesting mode: {e}")
            # Default to backtesting mode (safer - won't hit ForexFactory)
            return True

    # def _get_forexfactory_news_old(self, date):
    #     """
    #     Fetch news events from ForexFactory for a specific date.

    #     Args:
    #         date: datetime.date object

    #     Returns:
    #         list: List of news events for the date
    #     """
    #     date_str = date.strftime('%Y-%m-%d')

    #     # Check cache first
    #     if date_str in self._ff_news_cache:
    #         cache_time, data = self._ff_news_cache[date_str]
    #         if time.time() - cache_time < self._ff_cache_expiry:
    #             return data

    #     try:
    #         # ForexFactory calendar URL
    #         url = f"https://www.forexfactory.com/calendar?day={date.strftime('%b%d.%Y')}"

    #         headers = {
    #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    #         }

    #         response = requests.get(url, headers=headers, timeout=10)
    #         response.raise_for_status()

    #         soup = BeautifulSoup(response.content, 'html.parser')

    #         # Parse news events
    #         news_events = []
    #         calendar_rows = soup.find_all('tr', class_='calendar_row')

    #         for row in calendar_rows:
    #             try:
    #                 # Extract time
    #                 time_cell = row.find('td', class_='calendar__time')
    #                 if not time_cell:
    #                     continue

    #                 event_time = time_cell.get_text(strip=True)
    #                 if not event_time or event_time in ['Day', 'All Day']:
    #                     continue

    #                 # Extract currency
    #                 currency_cell = row.find('td', class_='calendar__currency')
    #                 currency = currency_cell.get_text(strip=True) if currency_cell else ''

    #                 # Extract impact
    #                 impact_cell = row.find('td', class_='calendar__impact')
    #                 impact_spans = impact_cell.find_all('span', class_='calendar__impact-icon') if impact_cell else []
    #                 impact_level = len([span for span in impact_spans if 'calendar__impact-icon--screen' in span.get('class', [])])

    #                 # Extract event title
    #                 event_cell = row.find('td', class_='calendar__event')
    #                 event_title = event_cell.get_text(strip=True) if event_cell else ''

    #                 # Only include high impact events (3 red bars) for major currencies
    #                 if impact_level >= 3 and currency in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']:
    #                     news_events.append({
    #                         'time': event_time,
    #                         'currency': currency,
    #                         'impact': impact_level,
    #                         'title': event_title
    #                     })

    #             except Exception as e:
    #                 logger.debug(f"Error parsing news row: {e}")
    #                 continue

    #         # Cache the results
    #         self._ff_news_cache[date_str] = (time.time(), news_events)

    #         logger.debug(f"Fetched {len(news_events)} high-impact news events for {date_str}")
    #         return news_events

    #     except Exception as e:
    #         logger.warning(f"Failed to fetch ForexFactory news for {date_str}: {e}")
    #         return []

    def _get_forexfactory_news(self, date):
        """
        Fetch news events from ForexFactory for a specific date.

        Args:
            date: datetime.date object

        Returns:
            list: List of high-impact news events for the date
        """
        import requests
        from bs4 import BeautifulSoup
        import time
        import logging

        logger = logging.getLogger(__name__)
        date_str = date.strftime('%Y-%m-%d')

        # Check cache first
        if date_str in self._ff_news_cache:
            cache_time, data = self._ff_news_cache[date_str]
            if time.time() - cache_time < self._ff_cache_expiry:
                return data

        try:
            url = f"https://www.forexfactory.com/calendar?day={date.strftime('%b%d.%Y')}"

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com",
                "Connection": "keep-alive"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            news_events = []
            calendar_rows = soup.find_all('tr', class_='calendar_row')

            for row in calendar_rows:
                try:
                    # Time
                    time_cell = row.find('td', class_='calendar__time')
                    event_time = time_cell.get_text(strip=True) if time_cell else ''
                    if not event_time or event_time in ['Day', 'All Day']:
                        continue

                    # Currency
                    currency_cell = row.find('td', class_='calendar__currency')
                    currency = currency_cell.get_text(strip=True) if currency_cell else ''

                    # Impact
                    impact_cell = row.find('td', class_='calendar__impact')
                    impact_icons = impact_cell.find_all('span', class_='calendar__impact-icon') if impact_cell else []
                    impact_level = len([
                        icon for icon in impact_icons
                        if 'calendar__impact-icon--screen' in icon.get('class', [])
                    ])

                    # Event title
                    event_cell = row.find('td', class_='calendar__event')
                    event_title = event_cell.get_text(strip=True) if event_cell else ''

                    # Only keep high-impact events for major currencies
                    if impact_level >= 3 and currency in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']:
                        news_events.append({
                            'time': event_time,
                            'currency': currency,
                            'impact': impact_level,
                            'title': event_title
                        })

                except Exception as e:
                    logger.debug(f"Error parsing news row: {e}")
                    continue

            # Cache results
            self._ff_news_cache[date_str] = (time.time(), news_events)
            logger.debug(f"Fetched {len(news_events)} high-impact news events for {date_str}")
            return news_events

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch ForexFactory news for {date_str}: {e}")
            return []


    # def _get_forexfactory_holidays_old(self, date):
    #     """
    #     Check if date is a market holiday according to ForexFactory.

    #     Args:
    #         date: datetime.date object

    #     Returns:
    #         bool: True if it's a holiday, False otherwise
    #     """
    #     date_str = date.strftime('%Y-%m-%d')

    #     # Check cache first
    #     if date_str in self._ff_holidays_cache:
    #         cache_time, is_holiday = self._ff_holidays_cache[date_str]
    #         if time.time() - cache_time < self._ff_cache_expiry:
    #             return is_holiday

    #     try:
    #         # Check if it's a weekend first
    #         if date.weekday() >= 5:  # Saturday (5) or Sunday (6)
    #             self._ff_holidays_cache[date_str] = (time.time(), True)
    #             return True

    #         # ForexFactory calendar URL for the specific date
    #         url = f"https://www.forexfactory.com/calendar?day={date.strftime('%b%d.%Y')}"

    #         headers = {
    #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    #         }

    #         response = requests.get(url, headers=headers, timeout=10)
    #         response.raise_for_status()

    #         soup = BeautifulSoup(response.content, 'html.parser')

    #         # Look for holiday indicators
    #         holiday_indicators = [
    #             'holiday', 'bank holiday', 'market closed', 'no trading',
    #             'christmas', 'new year', 'thanksgiving', 'independence day'
    #         ]

    #         page_text = soup.get_text().lower()
    #         is_holiday = any(indicator in page_text for indicator in holiday_indicators)

    #         # Also check for very few or no events (might indicate holiday)
    #         calendar_rows = soup.find_all('tr', class_='calendar_row')
    #         if len(calendar_rows) < 3:  # Very few events might indicate holiday
    #             is_holiday = True

    #         # Cache the result
    #         self._ff_holidays_cache[date_str] = (time.time(), is_holiday)

    #         if is_holiday:
    #             logger.debug(f"Holiday detected for {date_str}")

    #         return is_holiday

    #     except Exception as e:
    #         logger.warning(f"Failed to check ForexFactory holidays for {date_str}: {e}")
    #         # Fallback to basic weekend check
    #         return date.weekday() >= 5

    def _get_forexfactory_holidays(self, date):
        """
        Check if date is a market holiday according to ForexFactory.

        Args:
            date: datetime.date object

        Returns:
            bool: True if it's a holiday, False otherwise
        """
        import requests
        from bs4 import BeautifulSoup
        import time
        import logging

        logger = logging.getLogger(__name__)
        date_str = date.strftime('%Y-%m-%d')

        # Check cache
        if date_str in self._ff_holidays_cache:
            cache_time, is_holiday = self._ff_holidays_cache[date_str]
            if time.time() - cache_time < self._ff_cache_expiry:
                return is_holiday

        # Weekends are always holidays
        if date.weekday() >= 5:
            self._ff_holidays_cache[date_str] = (time.time(), True)
            return True

        url = f"https://www.forexfactory.com/calendar?day={date.strftime('%b%d.%Y')}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com",
            "Connection": "keep-alive"
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            page_text = soup.get_text().lower()
            holiday_keywords = [
                'holiday', 'bank holiday', 'market closed', 'no trading',
                'christmas', 'new year', 'thanksgiving', 'independence day'
            ]

            is_holiday = any(word in page_text for word in holiday_keywords)

            # If no or very few events are listed, might also indicate a holiday
            events = soup.find_all('tr', class_='calendar_row')
            if len(events) < 3:
                is_holiday = True

            self._ff_holidays_cache[date_str] = (time.time(), is_holiday)
            if is_holiday:
                logger.debug(f"Holiday detected for {date_str}")
            return is_holiday

        except requests.exceptions.HTTPError as e:
            logger.warning(f"Failed to check ForexFactory holidays for {date_str}: {e}")
            # Optional: Treat request failure as non-holiday, or fallback to a different method
            self._ff_holidays_cache[date_str] = (time.time(), False)
            return False

        except Exception as e:
            logger.exception(f"Unexpected error checking ForexFactory holidays for {date_str}")
            self._ff_holidays_cache[date_str] = (time.time(), False)
            return False


    def _is_bank_holiday(self, current_date):
        """
        Check if current date is a major bank holiday that affects trading.

        Args:
            current_date: datetime.date object

        Returns:
            bool: True if it's a bank holiday, False otherwise
        """
        if not current_date:
            logger.warning("_is_bank_holiday called with None date")
            return False
            
        # Make sure we have a date object, not a datetime
        if hasattr(current_date, 'date') and callable(getattr(current_date, 'date')):
            current_date = current_date.date()
            
        year = current_date.year
        month = current_date.month
        day = current_date.day
        
        logger.info(f"BANK HOLIDAY CHECK - Date: {current_date}, Year: {year}, Month: {month}, Day: {day}, Type: {type(current_date)}")

        # Major international bank holidays that affect XAUUSD trading
        bank_holidays = [
            # Fixed date holidays
            (1, 1),    # New Year's Day
            (12, 25),  # Christmas Day
            (12, 26),  # Boxing Day (UK/Commonwealth)
            (7, 4),    # US Independence Day
            (11, 11),  # Veterans Day (US)
            # Add your new holiday here
            (6, 9),   # Holiday in Europe, Switzerland and Australia
            # Major religious holidays (approximate dates - these vary by year)
            # Good Friday (varies - typically March/April)
            # Easter Monday (varies - typically March/April)
        ]
        
        logger.info(f"Checking against {len(bank_holidays)} fixed holidays")

        # Check fixed holidays
        for holiday_month, holiday_day in bank_holidays:
            logger.info(f"Comparing with holiday: Month {holiday_month}, Day {holiday_day}")
            # Explicitly convert to integers for safe comparison
            if int(month) == int(holiday_month) and int(day) == int(holiday_day):
                logger.info(f"MATCH FOUND! Bank holiday detected: {current_date} - Month: {month}, Day: {day}")
                return True
            else:
                logger.info(f"No match: {month}=={holiday_month}? {month == holiday_month}, {day}=={holiday_day}? {day == holiday_day}")
                
        # Explicit check for June 9th
        if month == 6 and day == 9:
            logger.info(f"EXPLICIT CHECK: June 9th detected on {current_date}")
            return True

        # Check for specific moveable holidays
        # Thanksgiving (4th Thursday of November in US)
        if month == 11:
            # Find 4th Thursday
            import calendar
            cal = calendar.monthcalendar(year, month)
            thursdays = [week[3] for week in cal if week[3] != 0]  # Thursday is index 3
            if len(thursdays) >= 4 and day == thursdays[3]:  # 4th Thursday
                logger.debug(f"Thanksgiving holiday detected: {current_date}")
                return True

        # Labor Day (1st Monday of September in US)
        if month == 9:
            import calendar
            cal = calendar.monthcalendar(year, month)
            mondays = [week[0] for week in cal if week[0] != 0]  # Monday is index 0
            if len(mondays) >= 1 and day == mondays[0]:  # 1st Monday
                logger.debug(f"Labor Day holiday detected: {current_date}")
                return True

        # Memorial Day (last Monday of May in US)
        if month == 5:
            import calendar
            cal = calendar.monthcalendar(year, month)
            mondays = [week[0] for week in cal if week[0] != 0]  # Monday is index 0
            if len(mondays) >= 1 and day == mondays[-1]:  # Last Monday
                logger.debug(f"Memorial Day holiday detected: {current_date}")
                return True

        # Check if it's a weekend (Saturday or Sunday)
        weekday = current_date.weekday()  # 0=Monday, 6=Sunday
        if weekday >= 5:  # Saturday (5) or Sunday (6)
            logger.debug(f"Weekend detected: {current_date} (weekday: {weekday})")
            return True

        return False

    # def _check_momentum_confluence_long(self, current_bar):
    #     """Check momentum indicators for long confluence."""
    #     rsi = current_bar.get('rsi', 50)
    #     adx = current_bar.get('adx', 0)
    #     di_plus = current_bar.get('di_plus', 0)
    #     di_minus = current_bar.get('di_minus', 0)

    #     # RSI should be above oversold but not overbought
    #     rsi_ok = self._rsi_oversold_level < rsi < self._rsi_overbought_level

    #     # ADX should show strong trend
    #     adx_ok = adx >= self._min_adx_for_entry

    #     # DI+ should be above DI- for bullish momentum
    #     di_ok = di_plus > di_minus

    #     return rsi_ok and adx_ok and di_ok

    # def _check_momentum_confluence_short(self, current_bar):
    #     """Check momentum indicators for short confluence."""
    #     rsi = current_bar.get('rsi', 50)
    #     adx = current_bar.get('adx', 0)
    #     di_plus = current_bar.get('di_plus', 0)
    #     di_minus = current_bar.get('di_minus', 0)

    #     # RSI should be below overbought but not oversold
    #     rsi_ok = self._rsi_oversold_level < rsi < self._rsi_overbought_level

    #     # ADX should show strong trend
    #     adx_ok = adx >= self._min_adx_for_entry

    #     # DI- should be above DI+ for bearish momentum
    #     di_ok = di_minus > di_plus

    #     return rsi_ok and adx_ok and di_ok

    # def _get_entry_condition_details(self, current_bar, direction):
    #     """Get detailed analysis of entry conditions for logging."""
    #     details = {}

    #     # Price and SMA analysis
    #     price = current_bar['Close']
    #     sma_fast = current_bar.get('sma_fast', 0)
    #     sma_slow = current_bar.get('sma_slow', 0)

    #     if direction == 'LONG':
    #         details['Price vs SMA Fast'] = f"✅ {price:.2f} > {sma_fast:.2f}" if price > sma_fast else f"❌ {price:.2f} ≤ {sma_fast:.2f}"
    #         details['SMA Fast vs Slow'] = f"✅ {sma_fast:.2f} > {sma_slow:.2f}" if sma_fast > sma_slow else f"❌ {sma_fast:.2f} ≤ {sma_slow:.2f}"
    #     else:
    #         details['Price vs SMA Fast'] = f"✅ {price:.2f} < {sma_fast:.2f}" if price < sma_fast else f"❌ {price:.2f} ≥ {sma_fast:.2f}"
    #         details['SMA Fast vs Slow'] = f"✅ {sma_fast:.2f} < {sma_slow:.2f}" if sma_fast < sma_slow else f"❌ {sma_fast:.2f} ≥ {sma_slow:.2f}"

    #     # SuperTrend analysis
    #     if direction == 'LONG':
    #         st_long = current_bar.get('st_long', 0)
    #         st_direction = current_bar.get('st_long_direction', 0)
    #         details['SuperTrend Direction'] = f"✅ Bullish ({st_direction})" if st_direction == 1 else f"❌ Not Bullish ({st_direction})"
    #         details['Price vs SuperTrend'] = f"✅ {price:.2f} > {st_long:.2f}" if price > st_long else f"❌ {price:.2f} ≤ {st_long:.2f}"
    #     else:
    #         st_short = current_bar.get('st_short', 0)
    #         st_direction = current_bar.get('st_short_direction', 0)
    #         details['SuperTrend Direction'] = f"✅ Bearish ({st_direction})" if st_direction == -1 else f"❌ Not Bearish ({st_direction})"
    #         details['Price vs SuperTrend'] = f"✅ {price:.2f} < {st_short:.2f}" if price < st_short else f"❌ {price:.2f} ≥ {st_short:.2f}"

    #     # Momentum indicators
    #     rsi = current_bar.get('rsi', 50)
    #     adx = current_bar.get('adx', 0)
    #     di_plus = current_bar.get('di_plus', 0)
    #     di_minus = current_bar.get('di_minus', 0)

    #     details['RSI Level'] = f"✅ {rsi:.2f} (Valid range)" if self._rsi_oversold_level < rsi < self._rsi_overbought_level else f"❌ {rsi:.2f} (Outside range)"
    #     details['ADX Strength'] = f"✅ {adx:.2f} ≥ {self._min_adx_for_entry}" if adx >= self._min_adx_for_entry else f"❌ {adx:.2f} < {self._min_adx_for_entry}"

    #     if direction == 'LONG':
    #         details['DI Momentum'] = f"✅ DI+ ({di_plus:.2f}) > DI- ({di_minus:.2f})" if di_plus > di_minus else f"❌ DI+ ({di_plus:.2f}) ≤ DI- ({di_minus:.2f})"
    #     else:
    #         details['DI Momentum'] = f"✅ DI- ({di_minus:.2f}) > DI+ ({di_plus:.2f})" if di_minus > di_plus else f"❌ DI- ({di_minus:.2f}) ≤ DI+ ({di_plus:.2f})"

    #     # Volatility filter
    #     atr = current_bar.get('atr', 0)
    #     details['ATR Volatility'] = f"✅ {atr:.2f} (Range: {self._min_atr_for_entry}-{self._max_atr_for_entry})" if self._min_atr_for_entry <= atr <= self._max_atr_for_entry else f"❌ {atr:.2f} (Outside range)"

    #     # Trading hours
    #     current_hour = current_bar.name.hour if hasattr(current_bar.name, 'hour') else 0
    #     current_minute = current_bar.name.minute if hasattr(current_bar.name, 'minute') else 0
    #     details['Trading Hours'] = f"✅ Hour {current_hour} (Range: {self._start_hour}-{self._end_hour})" if self._start_hour <= current_hour < self._end_hour else f"❌ Hour {current_hour} (Outside range)"

    #     # Market session filter
    #     if self._market_session_filter:
    #         # Check if current hour is within active trading sessions
    #         # London: 8-17 UTC, New York: 13-22 UTC, Asian: 0-9 UTC
    #         in_session = (0 <= current_hour <= 9 or 8 <= current_hour <= 17 or 13 <= current_hour <= 22)
    #         session_name = ""
    #         if 0 <= current_hour <= 9:
    #             session_name = "Asian"
    #         elif 8 <= current_hour <= 17:
    #             session_name = "London" if current_hour < 13 else "London/NY Overlap"
    #         elif 13 <= current_hour <= 22:
    #             session_name = "New York"

    #         details['Market Session'] = f"✅ {session_name} session ({current_hour:02d}:00)" if in_session else f"❌ Outside active sessions ({current_hour:02d}:00)"
    #     else:
    #         details['Market Session'] = "⚪ Disabled"

    #     # News time avoidance
    #     if self._avoid_news_times:
    #         current_date = current_bar.name.date() if hasattr(current_bar.name, 'date') else None
    #         is_news_time = self._is_news_time(current_hour, current_minute, current_date)
    #         mode_indicator = "FF" if self._use_forexfactory else "Static"
    #         details[f'News Time Filter ({mode_indicator})'] = f"❌ High-impact news detected ({current_hour:02d}:{current_minute:02d})" if is_news_time else f"✅ No news conflict ({current_hour:02d}:{current_minute:02d})"
    #     else:
    #         details['News Time Filter'] = "⚪ Disabled"

    #     # Bank holiday avoidance
    #     if self._avoid_bank_holidays:
    #         current_date = current_bar.name.date() if hasattr(current_bar.name, 'date') else None
    #         if current_date:
    #             if self._use_forexfactory:
    #                 is_holiday = self._get_forexfactory_holidays(current_date)
    #                 details['Bank Holiday Filter (FF)'] = f"❌ Market holiday detected ({current_date})" if is_holiday else f"✅ Normal trading day ({current_date})"
    #             else:
    #                 is_weekend = current_date.weekday() >= 5
    #                 details['Bank Holiday Filter (Basic)'] = f"❌ Weekend detected ({current_date})" if is_weekend else f"✅ Weekday ({current_date})"
    #         else:
    #             details['Bank Holiday Filter'] = "⚠️ No date available"
    #     else:
    #         details['Bank Holiday Filter'] = "⚪ Disabled"

    #     return details

    def log_trade_metrics_summary(self, trades_list):
        """Log comprehensive trade metrics summary."""
        if not trades_list:
            logger.info("📊 No trades to analyze")
            return

        logger.info("=" * 80)
        logger.info("📊 TRADE METRICS SUMMARY")
        logger.info("=" * 80)

        # Basic statistics
        total_trades = len(trades_list)
        winning_trades = [t for t in trades_list if hasattr(t, 'profit_loss') and t.profit_loss > 0]
        losing_trades = [t for t in trades_list if hasattr(t, 'profit_loss') and t.profit_loss < 0]

        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        logger.info(f"📈 Trade Statistics:")
        logger.info(f"   Total Trades: {total_trades}")
        logger.info(f"   Winning Trades: {len(winning_trades)}")
        logger.info(f"   Losing Trades: {len(losing_trades)}")
        logger.info(f"   Win Rate: {win_rate:.1f}%")

        # Exit reason analysis
        exit_reasons = {}
        for trade in trades_list:
            if hasattr(trade, 'exit_reason') and trade.exit_reason:
                reason_desc = self.get_exit_reason_description(trade.exit_reason)
                exit_reasons[reason_desc] = exit_reasons.get(reason_desc, 0) + 1

        if exit_reasons:
            logger.info(f"")
            logger.info(f"🚪 Exit Reason Breakdown:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_trades * 100) if total_trades > 0 else 0
                logger.info(f"   {reason}: {count} ({percentage:.1f}%)")

        # Direction analysis
        long_trades = [t for t in trades_list if hasattr(t, 'direction') and t.direction == TradeDirection.LONG]
        short_trades = [t for t in trades_list if hasattr(t, 'direction') and t.direction == TradeDirection.SHORT]

        if long_trades or short_trades:
            logger.info(f"")
            logger.info(f"📊 Direction Analysis:")
            logger.info(f"   LONG Trades: {len(long_trades)}")
            logger.info(f"   SHORT Trades: {len(short_trades)}")

            if long_trades:
                long_winners = [t for t in long_trades if hasattr(t, 'profit_loss') and t.profit_loss > 0]
                long_win_rate = (len(long_winners) / len(long_trades) * 100) if long_trades else 0
                logger.info(f"   LONG Win Rate: {long_win_rate:.1f}%")

            if short_trades:
                short_winners = [t for t in short_trades if hasattr(t, 'profit_loss') and t.profit_loss > 0]
                short_win_rate = (len(short_winners) / len(short_trades) * 100) if short_trades else 0
                logger.info(f"   SHORT Win Rate: {short_win_rate:.1f}%")

        logger.info("=" * 80)
        
    def _resample_dataframe(self, df, timeframe):
        """Resample dataframe to target timeframe for multi-timeframe analysis"""
        if timeframe == 'M1':
            return df.copy()
            
        timeframe_map = {
            'M5': '5T', 'M15': '15T', 'M30': '30T',
            'H1': '1H', 'H4': '4H',
            'D1': '1D', 'W1': 'W-MON', 'MN1': 'MS'
        }
        
        resampled = df.resample(timeframe_map[timeframe]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled

    # def _calculate_fibonacci_levels(self, swing_high, swing_low, is_extension=False):
    #     """Calculate Fibonacci retracement and extension levels"""
    #     levels = {}
    #     diff = swing_high - swing_low
        
    #     if not is_extension:
    #         # Retracement levels
    #         for level in self._pivot_fib_levels:
    #             levels[f'fib_{int(level*1000)/1000}'] = swing_high - (diff * level)
    #     else:
    #         # Extension levels
    #         for level in self._pivot_fib_extensions:
    #             levels[f'fib_ext_{int(level*1000)/1000}'] = swing_high + (diff * (level - 1))
                
    #     return levels

    # def _calculate_pivot_type(self, df, pivot_type):
    #     """Calculate different types of pivot points"""
    #     if pivot_type == 'standard':
    #         # Standard pivot points
    #         pivot = (df['high'] + df['low'] + df['close']) / 3
    #         r1 = 2 * pivot - df['low']
    #         s1 = 2 * pivot - df['high']
    #         r2 = pivot + (df['high'] - df['low'])
    #         s2 = pivot - (df['high'] - df['low'])
            
    #     elif pivot_type == 'fibonacci':
    #         # Fibonacci pivot points
    #         pivot = (df['high'] + df['low'] + df['close']) / 3
    #         diff = df['high'] - df['low']
    #         r1 = pivot + (0.382 * diff)
    #         r2 = pivot + (0.618 * diff)
    #         r3 = pivot + (1.0 * diff)
    #         s1 = pivot - (0.382 * diff)
    #         s2 = pivot - (0.618 * diff)
    #         s3 = pivot - (1.0 * diff)
            
    #     elif pivot_type == 'camarilla':
    #         # Camarilla pivot points
    #         diff = df['high'] - df['low']
    #         r1 = df['close'] + diff * 1.1/4
    #         r2 = df['close'] + diff * 1.1/2
    #         r3 = df['close'] + diff * 1.1/1.5
    #         r4 = df['close'] + diff * 1.1/1.1
    #         s1 = df['close'] - diff * 1.1/4
    #         s2 = df['close'] - diff * 1.1/2
    #         s3 = df['close'] - diff * 1.1/1.5
    #         s4 = df['close'] - diff * 1.1/1.1
            
    #     elif pivot_type == 'woodie':
    #         # Woodie pivot points
    #         pivot = (df['high'] + df['low'] + 2 * df['close']) / 4
    #         r1 = 2 * pivot - df['low']
    #         s1 = 2 * pivot - df['high']
    #         r2 = pivot + (df['high'] - df['low'])
    #         s2 = pivot - (df['high'] - df['low'])
            
    #     elif pivot_type == 'demark':
    #         # DeMark pivot points
    #         x = df['high'] + 2 * df['low'] + df['close'] if df['close'] < df['open'] else \
    #             2 * df['high'] + df['low'] + df['close'] if df['close'] > df['open'] else \
    #             df['high'] + df['low'] + 2 * df['close']
    #         pivot = x / 4
    #         r1 = x / 2 - df['low']
    #         s1 = x / 2 - df['high']
            
    #     # Return the appropriate pivot levels based on type
    #     if pivot_type in ['standard', 'woodie']:
    #         return {'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2}
    #     elif pivot_type == 'fibonacci':
    #         return {'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}
    #     elif pivot_type == 'camarilla':
    #         return {'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 's1': s1, 's2': s2, 's3': s3, 's4': s4}
    #     elif pivot_type == 'demark':
    #         return {'pivot': pivot, 'r1': r1, 's1': s1}
            
    #     return {}

    # def _calculate_pivots(self, df, timeframe=None):
    #     """
    #     Enhanced pivot point calculation with multiple timeframes and Fibonacci levels
    #     """
    #     if timeframe is None:
    #         timeframe = self._pivot_timeframe
            
    #     # Create a copy to avoid modifying the original
    #     df = df.copy()
        
    #     # Resample if needed for multi-timeframe analysis
    #     if timeframe != 'M1':
    #         resampled = self._resample_dataframe(df, timeframe)
    #     else:
    #         resampled = df
        
    #     # Initialize pivot columns with timeframe prefix
    #     prefix = f"{timeframe.lower()}_" if timeframe != 'M1' else ""
        
    #     # Standard pivot points
    #     pivots = self._calculate_pivot_type(resampled, self._pivot_type)
        
    #     # Add pivot levels to dataframe
    #     for level, values in pivots.items():
    #         df[f'{prefix}pivot_{level}'] = np.nan
    #         df.loc[resampled.index, f'{prefix}pivot_{level}'] = values
            
    #     # Find swing highs and lows for Fibonacci levels
    #     if self._pivot_use_fibonacci:
    #         # Find swing highs and lows
    #         df[f'{prefix}swing_high'] = df['high'].rolling(
    #             window=self._pivot_lookback_left + self._pivot_lookback_right + 1,
    #             center=True
    #         ).apply(lambda x: x.iloc[self._pivot_lookback_left] if x.idxmax() == self._pivot_lookback_left else np.nan, raw=False)
            
    #         df[f'{prefix}swing_low'] = df['low'].rolling(
    #             window=self._pivot_lookback_left + self._pivot_lookback_right + 1,
    #             center=True
    #         ).apply(lambda x: x.iloc[self._pivot_lookback_left] if x.idxmin() == self._pivot_lookback_left else np.nan, raw=False)
            
    #         # Forward fill swing points to extend their influence
    #         df[f'{prefix}swing_high'] = df[f'{prefix}swing_high'].fillna(method='ffill')
    #         df[f'{prefix}swing_low'] = df[f'{prefix}swing_low'].fillna(method='ffill')
            
    #         # Calculate Fibonacci levels
    #         for i in range(1, len(df)):
    #             if not pd.isna(df[f'{prefix}swing_high'].iloc[i]) and not pd.isna(df[f'{prefix}swing_low'].iloc[i]):
    #                 fib_levels = self._calculate_fibonacci_levels(
    #                     df[f'{prefix}swing_high'].iloc[i],
    #                     df[f'{prefix}swing_low'].iloc[i]
    #                 )
                    
    #                 if self._pivot_use_fib_extensions:
    #                     fib_levels.update(self._calculate_fibonacci_levels(
    #                         df[f'{prefix}swing_high'].iloc[i],
    #                         df[f'{prefix}swing_low'].iloc[i],
    #                         is_extension=True
    #                     ))
                    
    #                 # Store Fibonacci levels
    #                 for level, value in fib_levels.items():
    #                     df.loc[df.index[i], f'{prefix}fib_{level}'] = value
        
    #     return df
        
    # def calculate_multi_timeframe_pivots(self, df):
    #     """Calculate pivots for multiple timeframes"""
    #     if not self._pivot_use_multiple_timeframes:
    #         return self._calculate_pivots(df, self._pivot_timeframe)
            
    #     # Calculate pivots for each timeframe
    #     for tf in self._pivot_timeframes:
    #         df = self._calculate_pivots(df, tf)
            
    #     return df

    # def _check_pivots_long(self, current_bar, prev_bar, check_entries):
    #     """
    #     Check pivot points for long entry conditions using the new pivot system.
        
    #     Args:
    #         current_bar: Current bar data
    #         prev_bar: Previous bar data
    #         check_entries: Dictionary to store entry conditions
            
    #     Returns:
    #         bool: True if pivot conditions for long are met
    #     """
    #     if not self._trade_logic_pivots:
    #         return False
            
    #     # Check for price above main pivot level
    #     main_timeframe = self._pivot_timeframe.lower()
    #     pivot_key = f"{main_timeframe}_pivot_pivot"
        
    #     if pivot_key not in current_bar or pd.isna(current_bar[pivot_key]):
    #         return False
            
    #     # Check for price above pivot and R1
    #     if (current_bar['Close'] > current_bar[pivot_key] and 
    #         current_bar['Close'] > current_bar.get(f"{main_timeframe}_pivot_r1", float('-inf'))):
    #         check_entries['pivot_above_r1'] = True
    #         return True
            
    #     return False

    # def _check_pivots_short(self, current_bar, prev_bar, check_entries):
    #     """
    #     Check pivot points for short entry conditions using the new pivot system.
        
    #     Args:
    #         current_bar: Current bar data
    #         prev_bar: Previous bar data
    #         check_entries: Dictionary to store entry conditions
            
    #     Returns:
    #         bool: True if pivot conditions for short are met
    #     """
    #     if not self._trade_logic_pivots:
    #         return False
            
    #     # Check for price below main pivot level
    #     main_timeframe = self._pivot_timeframe.lower()
    #     pivot_key = f"{main_timeframe}_pivot_pivot"
        
    #     if pivot_key not in current_bar or pd.isna(current_bar[pivot_key]):
    #         return False
            
    #     # Check for price below pivot and S1
    #     if (current_bar['Close'] < current_bar[pivot_key] and 
    #         current_bar['Close'] < current_bar.get(f"{main_timeframe}_pivot_s1", float('inf'))):
    #         check_entries['pivot_below_s1'] = True
    #         return True
            
    #     return False
    
    def _calculate_cpr(self, df):
        """Calculate Central Pivot Range (CPR) values"""
        if len(df) < 2:
            return df
            
        # Get prior day OHLC if available, otherwise use first bar's values
        prior_high = self._cpr_prior_day_high if self._cpr_prior_day_high > 0 else df['High'].iloc[0]
        prior_low = self._cpr_prior_day_low if self._cpr_prior_day_low > 0 else df['Low'].iloc[0]
        prior_close = self._cpr_prior_day_close if self._cpr_prior_day_close > 0 else df['Close'].iloc[0]
        
        # Calculate CPR values
        pivot = (prior_high + prior_low + prior_close) / 3
        bc = (prior_high + prior_low) / 2
        tc = pivot - bc + pivot
        
        # Calculate width-based levels
        pivot_width = self._cpr_pivot_width * (prior_high - prior_low) / 100
        tc_width = self._cpr_tc_width * (prior_high - prior_low) / 100
        bc_width = self._cpr_bc_width * (prior_high - prior_low) / 100
        
        # Add CPR values to dataframe
        df['cpr_pivot'] = pivot
        df['cpr_tc'] = tc
        df['cpr_bc'] = bc
        df['cpr_pivot_upper'] = pivot + pivot_width
        df['cpr_pivot_lower'] = pivot - pivot_width
        df['cpr_tc_upper'] = tc + tc_width
        df['cpr_tc_lower'] = tc - tc_width
        df['cpr_bc_upper'] = bc + bc_width
        df['cpr_bc_lower'] = bc - bc_width
        
        return df
        
    def _check_cpr_long(self, current_bar, prev_bar, check_entries):
        """Check CPR conditions for long entry"""
        if not self._trade_logic_cpr or not check_entries:
            return True
            
        # Long conditions:
        # 1. Price is above CPR pivot
        # 2. Price is above TC (Top Central) level
        # 3. TC is above BC (Bottom Central) level (bullish structure)
        price_above_pivot = current_bar['Close'] > current_bar.get('cpr_pivot_upper', 0)
        price_above_tc = current_bar['Close'] > current_bar.get('cpr_tc_upper', 0)
        tc_above_bc = current_bar.get('cpr_tc', 0) > current_bar.get('cpr_bc', 0)
        
        # Optional: Check for price bouncing from BC or pivot
        bounce_from_bc = False
        if prev_bar is not None:
            bounce_from_bc = (prev_bar['Low'] <= current_bar.get('cpr_bc_lower', 0) and 
                           current_bar['Close'] > current_bar.get('cpr_bc_upper', 0))
        
        return price_above_pivot and price_above_tc and tc_above_bc
    
    def _check_cpr_short(self, current_bar, prev_bar, check_entries):
        """Check CPR conditions for short entry"""
        if not self._trade_logic_cpr or not check_entries:
            return True
            
        # Short conditions:
        # 1. Price is below CPR pivot
        # 2. Price is below BC (Bottom Central) level
        # 3. BC is above TC (Top Central) level (bearish structure)
        price_below_pivot = current_bar['Close'] < current_bar.get('cpr_pivot_lower', 0)
        price_below_bc = current_bar['Close'] < current_bar.get('cpr_bc_lower', 0)
        bc_above_tc = current_bar.get('cpr_bc', 0) > current_bar.get('cpr_tc', 0)
        
        # Optional: Check for price rejecting from TC or pivot
        rejection_from_tc = False
        if prev_bar is not None:
            rejection_from_tc = (prev_bar['High'] >= current_bar.get('cpr_tc_upper', 0) and 
                              current_bar['Close'] < current_bar.get('cpr_tc_lower', 0))
        
        return price_below_pivot and price_below_bc and bc_above_tc
    
    def _check_market_structure_long(self, current_bar, prev_bar):
        """Check market structure for long entries."""
        if not self._trade_logic_marketstructure:
            return True  # Return True if market structure logic is disabled
            
        # Placeholder for market structure logic
        # Example: Check for higher highs and higher lows
        if len(self.df) < 3:
            return False
            
        # Simple market structure check (can be enhanced)
        prev_prev_bar = self.df.iloc[-3] if len(self.df) > 2 else None
        
        if prev_prev_bar is None:
            return False
            
        # Check for higher high and higher low
        higher_high = current_bar['High'] > prev_bar['High'] > prev_prev_bar['High']
        higher_low = current_bar['Low'] > prev_bar['Low'] > prev_prev_bar['Low']
        
        return higher_high and higher_low
    
    def _check_market_structure_short(self, current_bar, prev_bar):
        """Check market structure for short entries."""
        if not self._trade_logic_marketstructure:
            return True  # Return True if market structure logic is disabled
            
        # Placeholder for market structure logic
        # Example: Check for lower highs and lower lows
        if len(self.df) < 3:
            return False
            
        # Simple market structure check (can be enhanced)
        prev_prev_bar = self.df.iloc[-3] if len(self.df) > 2 else None
        
        if prev_prev_bar is None:
            return False
            
        # Check for lower high and lower low
        lower_high = current_bar['High'] < prev_bar['High'] < prev_prev_bar['High']
        lower_low = current_bar['Low'] < prev_bar['Low'] < prev_prev_bar['Low']
        
        return lower_high and lower_low

    def _log_signal_conditions(self, timestamp, direction, strength_data, pending_flag, pending_timestamp=None, st_level=None):
        """
        Helper method to log signal conditions in a standardized format.
        
        Args:
            timestamp: Timestamp of the current bar
            direction: 'LONG' or 'SHORT'
            strength_data: Dictionary with signal strength information
            pending_flag: Boolean indicating if there's a pending signal flag
            pending_timestamp: Timestamp when the flag was set (optional)
            st_level: SuperTrend level to touch for entry (optional)
        """
        # Determine emoji based on direction
        direction_emoji = "🟢" if direction == "LONG" else "🔴"
        
        # Log main condition status
        #logger.info(f"{direction_emoji} {direction} Signal Conditions:")
        logger.info(f"  Required conditions met: {'✅ YES' if strength_data['required_conditions_met'] else '❌ NO'}")
        
        # Use weighted_strength if available, otherwise use strength
        #strength_value = strength_data.get('weighted_strength', strength_data.get('strength', 0))
        #logger.info(f"  Signal strength: {strength_value:.1f}%")
        logger.info(f"  Pending flag active: {'✅ YES' if pending_flag else '❌ NO'}")
        
        if pending_flag and pending_timestamp is not None and st_level is not None:
            logger.info(f"  Flag set at: {pending_timestamp}")
            logger.info(f"  Waiting for price to touch ST level: {st_level:.2f}")
        
        # Log individual conditions
        #logger.info(f"  {direction} Condition Details:")
        
        # Check which field contains the condition details
        #if 'detailed_conditions' in strength_data:
        #    conditions_dict = strength_data['detailed_conditions']
        #else:
        #    # If detailed_conditions is not available, try to use the direct conditions field
        #    conditions_dict = strength_data.get('conditions', {})
        #    
        #for condition, config in strength_data['condition_config'].items():
        #    status = conditions_dict.get(condition, False)
        #    required_marker = "(Required)" if config['required'] else "(Optional)"
        #    logger.info(f"    {condition} {required_marker}: {'✅' if status else '❌'}")

    def _log_price_information(self, current_bar, prev_bar):
        """
        Helper method to log price and indicator information.
        
        Args:
            current_bar: Current bar data
            prev_bar: Previous bar data
        """
        logger.info(f"\n{'='*120}")
        logger.info(f"    📊 PRICE INFORMATION:")
        logger.info(f"      📈 OHLC: O:{current_bar['Open']:.2f} H:{current_bar['High']:.2f} L:{current_bar['Low']:.2f} C:{current_bar['Close']:.2f}")
        logger.info(f"      📉 Previous: O:{prev_bar['Open']:.2f} H:{prev_bar['High']:.2f} L:{prev_bar['Low']:.2f} C:{prev_bar['Close']:.2f}")
        logger.info(f"      🔄 Change: {(current_bar['Close'] - prev_bar['Close']):.2f} ({((current_bar['Close'] - prev_bar['Close'])/prev_bar['Close']*100):.2f}%)")
        logger.info(f"      📏 Range: {(current_bar['High'] - current_bar['Low']):.2f}")
        logger.info(f"      📊 Indicators: SMA Fast: {current_bar['sma_fast']:.2f}, SMA Slow: {current_bar['sma_slow']:.2f}")
        logger.info(f"      📊 SuperTrend: Long: {current_bar['st_long']:.2f}, Short: {current_bar['st_short']:.2f}")
        logger.info(f"      📊 ADX: {current_bar['adx']:.2f}, ADXR: {current_bar['adxr']:.2f}, +DI: {current_bar['di_plus']:.2f}, -DI: {current_bar['di_minus']:.2f}")
        logger.info(f"      📊 RSI: {current_bar['rsi']:.2f}, RSI MA: {current_bar['rsi_ma']:.2f}")
        logger.info(f"      📊 ATR: {current_bar['atr']:.2f}")
        logger.info(f"\n{'='*120}")
        
    def _log_exit_conditions(self, exit_conditions):
        """Log detailed exit condition status for a trade.
        
        Args:
            exit_conditions: List of dictionaries containing exit condition details
        """
        logger.info(f"\n{'='*120}")
        logger.info(f"    🚪 EXIT CONDITIONS ANALYSIS:")
        for condition in exit_conditions:
            status_icon = "🔴" if condition['status'] == 'ACTIVE' else "🟢"
            logger.info(f"      {status_icon} {condition['type']}: {condition['condition']}")
            if condition['exit_price']:
                logger.info(f"         Exit Price: {condition['exit_price']:.2f}")                            
        logger.info(f"\n{'='*120}")

    def generate_signal_for_bar(self, current_bar, prev_bar, num_open_trades):
        """
        Generate trading signals for a single bar (for backtest loop).
        Returns a list of TradeSignal objects (empty if no signal).
        
        Implements dual trading styles based on ATR:
        - Swing style (ATR >= 1.30): Immediate entries without waiting for ST_Short touch, 0.5% risk
        - Scalp style (ATR < 1.30): Requires price to touch ST_Short, default risk
        
        """
        signals = []
        timestamp = current_bar.name
        
        # Check market conditions first (including bank holidays)
        if not self._check_market_conditions(current_bar):
            logger.info(f"[{timestamp}] Market conditions check failed - no trading allowed")
            return []

        # # Initialize pending signal flags if they don't exist
        # if not hasattr(self, '_pending_long_signal'):
        #     self._pending_long_signal = False
        #     self._pending_short_signal = False
        #     self._pending_signal_timestamp = None

        # Determine current trading style based on ATR
        trading_style = self._determine_trading_style(current_bar)
        
        # # Get the trading style based on ATR
        # dynamic_trading_style = self._determine_trading_style(current_bar)
        # #print("dynamic_trading_style value:", dynamic_trading_style)
        # #print("self._config_trading_style value:", self._config_trading_style)
        # # Check if the configured trading style allows this dynamic style
        # should_run_scalp = (self._config_trading_style == 'scalp' or self._config_trading_style == 'both') and dynamic_trading_style == 'scalp'
        # should_run_swing = (self._config_trading_style == 'swing' or self._config_trading_style == 'both') and dynamic_trading_style == 'swing'
        # #print("should_run_scalp value:", should_run_scalp)
        # #print("should_run_swing value:", should_run_swing)
        # Check required indicators
        required_indicators = [
            'st_long', 'st_long_direction', 'st_short', 'st_short_direction',
            'sma_fast', 'sma_slow', 'adx', 'adxr', 'di_plus', 'di_minus', 'rsi', 'rsi_ma', 'atr'
        ]
        if not all(col in current_bar for col in required_indicators):
            logger.warning(f"[{timestamp}] Missing required indicators for signal generation")
            return []

        # Check if you can take new trades
        can_take_new_trades = num_open_trades < self._max_open_trades
        if not can_take_new_trades:
            logger.info(f"[{timestamp}] Cannot take new trades: {num_open_trades}/{self._max_open_trades} positions open")

        # Calculate signal strength and required condition status
        long_strength = self._calculate_signal_strength_long(current_bar, prev_bar)
        short_strength = self._calculate_signal_strength_short(current_bar, prev_bar)

        long_condition = long_strength['required_conditions_met']
        short_condition = short_strength['required_conditions_met']

        # # Log detailed condition status for both LONG and SHORT
        # logger.info(f"\n{'='*50}")
        # logger.info(f"[{timestamp}] SIGNAL CONDITION ANALYSIS:")
        # logger.info(f"{'='*50}")
        
        # # Log LONG and SHORT conditions using helper methods
        # self._log_signal_conditions(
        #     timestamp, "LONG", long_strength, 
        #     self._pending_long_signal, self._pending_signal_timestamp, 
        #     current_bar['st_short'] if 'st_short' in current_bar else None
        # )
        
        # self._log_signal_conditions(
        #     timestamp, "SHORT", short_strength, 
        #     self._pending_short_signal, self._pending_signal_timestamp,
        #     current_bar['st_short'] if 'st_short' in current_bar else None
        # )
        
        # Log price information
        #self._log_price_information(current_bar, prev_bar)
        #logger.info(f"{'='*50}\n")

        # --- IMPLEMENT SUPERTREND SIGNAL FLAG LOGIC ---
        
        # 1. Set signal flags when conditions are met (if not already set)
        if long_condition and not self._pending_long_signal:
            self._pending_long_signal = True
            self._pending_signal_timestamp = timestamp
            logger.info(f"[{timestamp}] ✅ Setting LONG signal flag to True")
            #logger.info(f"[{timestamp}] 🔍 Waiting for candle to touch ST_Short level: {current_bar['st_short']:.2f}")
        
        if short_condition and not self._pending_short_signal:
            self._pending_short_signal = True
            self._pending_signal_timestamp = timestamp
            logger.info(f"[{timestamp}] ✅ Setting SHORT signal flag to True")
            #logger.info(f"[{timestamp}] 🔍 Waiting for candle to touch ST_Short level: {current_bar['st_short']:.2f}")
        
        # 2. Reset signal flags if candle closes beyond ST_Long
        if self._pending_long_signal and current_bar['Close'] < current_bar['st_long']:
            self._pending_long_signal = False
            logger.info(f"[{timestamp}] ❌ Resetting LONG signal flag to False")
            logger.info(f"[{timestamp}] Reason: Candle closed below ST_Long: {current_bar['st_long']:.2f}")
        
        if self._pending_short_signal and current_bar['Close'] > current_bar['st_long']:
            self._pending_short_signal = False
            logger.info(f"[{timestamp}] ❌ Resetting SHORT signal flag to False")
            logger.info(f"[{timestamp}] Reason: Candle closed above ST_Long: {current_bar['st_long']:.2f}")
        
        # Initialize entry condition variables to avoid reference errors
        long_entry_condition = False
        short_entry_condition = False

        if trading_style == 'scalp':
        #if should_run_scalp:    
            logger.debug(f"Running scalp trading logic for bar at {current_bar.name}")
            #logger.info(f"[{timestamp}] 🔍 Waiting for candle to touch ST_Short level: {current_bar['st_short']:.2f}")

            # 3. Check if candle has touched ST_Short level for entry AND RSI conditions are met
            st_touch_long = self._pending_long_signal and current_bar['Low'] <= current_bar['st_short']
            st_touch_short = self._pending_short_signal and current_bar['High'] >= current_bar['st_short']
                        
            # Check RSI conditions separately for better logging
            rsi_condition_long = current_bar['rsi'] > current_bar['rsi_ma']
            rsi_condition_short = current_bar['rsi'] <current_bar['rsi_ma']
            
            # Log RSI condition status when ST touch condition is met
            if st_touch_long and self._pending_long_signal:
                logger.info(f"[{timestamp}] 📊 LONG RSI Check: RSI({current_bar['rsi']:.2f}) {'>' if rsi_condition_long else '<='} RSI_MA({current_bar['rsi_ma']:.2f}) - {'✅ PASSED' if rsi_condition_long else '❌ FAILED'}")
            
            if st_touch_short and self._pending_short_signal:
                logger.info(f"[{timestamp}] 📊 SHORT RSI Check: RSI({current_bar['rsi']:.2f}) {'<' if rsi_condition_short else '>='} RSI_MA({current_bar['rsi_ma']:.2f}) - {'✅ PASSED' if rsi_condition_short else '❌ FAILED'}")
            
            # Check ADX > ADXR condition
            adx_condition_long = current_bar['adx'] > current_bar['adxr']
            adx_condition_short = current_bar['adx'] > current_bar['adxr']
            
            # Log ADX condition status when ST touch condition is met
            if st_touch_long and self._pending_long_signal:
                logger.info(f"[{timestamp}] 📊 LONG ADX Check: ADX({current_bar['adx']:.2f}) {'>' if adx_condition_long else '<='} ADXR({current_bar['adxr']:.2f}) - {'✅ PASSED' if adx_condition_long else '❌ FAILED'}")
            
            if st_touch_short and self._pending_short_signal:
                logger.info(f"[{timestamp}] 📊 SHORT ADX Check: ADX({current_bar['adx']:.2f}) {'>' if adx_condition_short else '<='} ADXR({current_bar['adxr']:.2f}) - {'✅ PASSED' if adx_condition_short else '❌ FAILED'}")
            
            # Final entry conditions combining ST touch, RSI conditions, and ADX > ADXR
            long_entry_condition = st_touch_long and rsi_condition_long and adx_condition_long
            short_entry_condition = st_touch_short and rsi_condition_short and adx_condition_short
            
        if trading_style == 'swing':
        #if should_run_swing:
            logger.debug(f"Running swing trading logic for bar at {current_bar.name}")
            # Check if this is the first candle after ST direction change
            st_long_direction_changed = prev_bar is not None and current_bar['st_long_direction'] != prev_bar['st_long_direction']
            st_short_direction_changed = prev_bar is not None and current_bar['st_short_direction'] != prev_bar['st_short_direction']
                
            # Long entry on first candle after ST_Long direction change from -1 to 1 (bullish)
            long_direction_change = st_long_direction_changed and current_bar['st_long_direction'] == 1
                
            # Short entry on first candle after ST_Short direction change from 1 to -1 (bearish)
            short_direction_change = st_short_direction_changed and current_bar['st_short_direction'] == -1
            
            # Check if both SuperTrend indicators are in the same direction (for re-entry after TP_HIT)
            st_aligned_long = current_bar['st_long_direction'] == 1 and current_bar['st_short_direction'] == 1
            st_aligned_short = current_bar['st_long_direction'] == -1 and current_bar['st_short_direction'] == -1
            
            # Log SuperTrend alignment status
            if st_aligned_long:
                logger.info(f"[{timestamp}] ✅ SuperTrend indicators aligned for LONG (both = 1)")
            elif st_aligned_short:
                logger.info(f"[{timestamp}] ✅ SuperTrend indicators aligned for SHORT (both = -1)")
            else:
                logger.info(f"[{timestamp}] ❌ SuperTrend indicators not aligned: ST_Long={current_bar['st_long_direction']}, ST_Short={current_bar['st_short_direction']}")
            
            # Final entry conditions for swing style
            # Either first candle after direction change OR re-entry when SuperTrends are aligned (only after TP_HIT)
            last_exit_was_tp = self.last_trade_exit['reason'] == 'TP_HIT'
            
            # Log whether we're considering re-entry after TP_HIT
            if last_exit_was_tp:
                logger.info(f"[{timestamp}] Last trade exited with TP_HIT, allowing re-entry with aligned SuperTrends")
            
                # For LONG: Direction change OR (SuperTrend alignment only if last exit was TP_HIT)
                long_entry_condition = self._pending_long_signal and (long_direction_change or (st_aligned_long and last_exit_was_tp))
            
                # For SHORT: Direction change OR (SuperTrend alignment only if last exit was TP_HIT)
                short_entry_condition = self._pending_short_signal and (short_direction_change or (st_aligned_short and last_exit_was_tp))
            else:
                long_entry_condition = self._pending_long_signal and st_long_direction_changed
                short_entry_condition = self._pending_short_signal and st_short_direction_changed
                logger.info(f"Swing First Entry Trade Condition met: {long_entry_condition} {short_entry_condition}")
                
        # Generate LONG signal if pending and candle touches ST_Short
        if long_entry_condition and can_take_new_trades:
            entry = current_bar['Close']
            stop_loss, take_profit, position_size = self.calculate_atr_based_levels(current_bar, TradeDirection.LONG)
            if stop_loss is not None and take_profit is not None and position_size is not None:
                logger.info(f"\n{'*'*60}")
                logger.info(f"[{timestamp}] 🚀 LONG ENTRY TRIGGERED")
                logger.info(f"{'*'*60}")
                logger.info(f"  Signal flag was set at: {self._pending_signal_timestamp}")
                logger.info(f"  Entry condition met: Candle touched ST_Short level")
                logger.info(f"  ST_Short: {current_bar['st_short']:.2f}, Candle Low: {current_bar['Low']:.2f}")
                logger.info(f"  Entry price: {entry:.2f}")
                logger.info(f"  Stop loss: {stop_loss:.2f}")
                logger.info(f"  Take profit: {take_profit:.2f}")
                logger.info(f"  Position size: {position_size:.2f} lots")
                logger.info(f"{'*'*60}\n")
                
                signal = TradeSignal(
                    direction=TradeDirection.LONG,
                    entry=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=timestamp,
                    symbol=self.symbol,
                    trailing_stop=True,
                    atr_trailing_multiplier=1.5,
                    highest_high=entry,
                    lowest_low=float('inf')
                )
                signal._strategy = self
                signal.position_size = position_size
                signals.append(signal)
        elif self._pending_long_signal and not long_entry_condition:
            logger.info(f"[{timestamp}] ⏳ LONG entry waiting: Candle hasn't touched ST_Short yet")
            logger.info(f"[{timestamp}] ST_Short: {current_bar['st_short']:.2f}, Candle Low: {current_bar['Low']:.2f}")
            logger.info(f"[{timestamp}] Distance to entry: {current_bar['Low'] - current_bar['st_short']:.2f} points")

        # Generate SHORT signal if pending and candle touches ST_Short
        if short_entry_condition and can_take_new_trades:
            entry = current_bar['Close']
            stop_loss, take_profit, position_size = self.calculate_atr_based_levels(current_bar, TradeDirection.SHORT)
            if stop_loss is not None and take_profit is not None and position_size is not None:
                logger.info(f"\n{'*'*60}")
                logger.info(f"[{timestamp}] 🚀 SHORT ENTRY TRIGGERED")
                logger.info(f"{'*'*60}")
                logger.info(f"  Signal flag was set at: {self._pending_signal_timestamp}")
                logger.info(f"  Entry condition met: Candle touched ST_Short level")
                logger.info(f"  ST_Short: {current_bar['st_short']:.2f}, Candle High: {current_bar['High']:.2f}")
                logger.info(f"  Entry price: {entry:.2f}")
                logger.info(f"  Stop loss: {stop_loss:.2f}")
                logger.info(f"  Take profit: {take_profit:.2f}")
                logger.info(f"  Position size: {position_size:.2f} lots")
                logger.info(f"{'*'*60}\n")
                
                signal = TradeSignal(
                    direction=TradeDirection.SHORT,
                    entry=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=timestamp,
                    symbol=self.symbol,
                    trailing_stop=True,
                    atr_trailing_multiplier=1.5,
                    highest_high=0,
                    lowest_low=entry
                )
                signal._strategy = self
                signal.position_size = position_size
                signals.append(signal)
        elif self._pending_short_signal and not short_entry_condition:
            logger.info(f"[{timestamp}] ⏳ SHORT entry waiting: Candle hasn't touched ST_Short yet")
            logger.info(f"[{timestamp}] 🎯 ST_Short: {current_bar['st_short']:.2f}, Candle High: {current_bar['High']:.2f}")
            logger.info(f"[{timestamp}] 📏 Distance to entry: {current_bar['st_short'] - current_bar['High']:.2f} points")

        return signals
    
    #def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        # """
        # Generate trading signals based on the strategy rules and enabled trade logics.
        
        # After a take profit, the next entry will only be taken when:
        # 1. Price touches the sma_fast
        # 2. All other entry conditions are met
        
        # Also checks for daily drawdown limit before generating new signals.
        # """
        # if len(df) < 200:  # Ensure we have enough data
        #     return []

        # self.df = df  # Store df as instance variable for use in other methods
        # signals = []
        # current_bar = df.iloc[-1]
        # current_time = df.index[-1].to_pydatetime()
        
        # # Check if we can take new trades (but continue signal analysis regardless)
        # can_take_new_trades = len(self._open_positions) < self._max_open_trades
        # if not can_take_new_trades:
        #     logger.debug(f"Generate Signals: [TRADE BLOCKED] Max open trades limit reached: {len(self._open_positions)}/{self._max_open_trades} - continuing signal analysis")
        
        # # Check for new trading day and reset daily peak balance if needed
        # current_day = current_time.date()
        # if self._last_trade_day is None or current_day > self._last_trade_day:
        #     self._daily_peak_balance = self._account_balance
        #     self._last_trade_day = current_day
        #     logger.info(f"Generate Signals: New trading day {current_day.strftime('%Y-%m-%d')}. Reset daily peak balance to: ${self._daily_peak_balance:.2f}")
        
        # # Update daily peak balance if current balance is higher
        # if self._account_balance > self._daily_peak_balance:
        #     logger.info(f"Generate Signals: New daily peak balance: ${self._account_balance:.2f} (previous: ${self._daily_peak_balance:.2f})")
        #     self._daily_peak_balance = self._account_balance
        
        # # Calculate current drawdown from daily peak if drawdown limit is enabled
        # if self._daily_drawdown_limit and self._daily_peak_balance > 0:
        #     drawdown_pct = ((self._daily_peak_balance - self._account_balance) / self._daily_peak_balance) * 100
            
        #     # Check if drawdown exceeds limit
        #     if drawdown_pct >= self._daily_drawdown_limit_pct:
        #         logger.warning(
        #             f"Generate Signals: Daily drawdown limit reached: {drawdown_pct:.2f}% "
        #             f"(Limit: {self._daily_drawdown_limit_pct}%, "
        #             f"Peak: ${self._daily_peak_balance:.2f}, Current: ${self._account_balance:.2f}). "
        #             "Generate Signals: Blocking new trade entries for the rest of the day."
        #         )
        #         return []  # Return empty signals list to block new trades
        
        # # Get previous bar data if available
        # prev_bar = df.iloc[-2] if len(df) > 1 else None
        
        # # 🔍 DEBUG: Show all signal conditions
        # if current_bar['sma_fast'] > current_bar['sma_slow']:
        #     self.debug_signal_conditions_mtf("LONG", current_bar, prev_bar)
        # elif current_bar['sma_fast'] < current_bar['sma_slow']:
        #     self.debug_signal_conditions_mtf("SHORT", current_bar, prev_bar)
        # else:
        #     # Optionally, handle the case where they are equal
        #     print("No clear trend: SMA Fast equals SMA Slow")
        
        # # Check if we have the required indicator columns
        # required_indicators = ['st_long', 'st_long_direction', 'st_short', 'st_short_direction',
        #                      'sma_fast', 'sma_slow', 'adx', 'adxr', 'di_plus', 'di_minus', 'rsi', 'rsi_ma']
        
        # has_required_indicators = all(col in current_bar for col in required_indicators)
        # if not has_required_indicators:
        #     missing = [col for col in required_indicators if col not in current_bar]
        #     logger.warning(f"Generate Signals: Missing required indicator columns in generate_signals: {missing}")
        #     return []
        
        # # Check if the last trade was a take profit
        # last_was_tp = self.last_trade_exit.get('reason') == 'TP_HIT'
        
        # # Check if price has touched st_short since last trade (for both LONG and SHORT entries)
        # price_touched_st_short = False
        # if last_was_tp and self.last_trade_exit['time'] is not None:
        #     # Get bars since last trade exit
        #     since_last_trade = df[df.index > self.last_trade_exit['time']]
        #     # Check if price has touched st_short since last trade (for both directions)
        #     if not since_last_trade.empty:
        #         if self.last_trade_exit['direction'] == TradeDirection.LONG:
        #             # For long exits, check if price has come down to touch st_short (for next LONG entry)
        #             price_touched_st_short = any((row['Low'] <= row['st_short']) for _, row in since_last_trade.iterrows())
        #         else:  # SHORT
        #             # For short exits, check if price has come up to touch st_short (for next SHORT entry)
        #             price_touched_st_short = any((row['High'] >= row['st_short']) for _, row in since_last_trade.iterrows())
        
        # # Only check entry conditions if:
        # # 1. No last trade, or
        # # 2. Last trade was not a take profit, or
        # # 3. Last trade was a take profit AND price has touched st_short since then
        # check_entries = (not self.last_trade_exit['time'] or 
        #                 not last_was_tp or 
        #                 (last_was_tp and price_touched_st_short))
        
        # # CRITICAL DEBUG: Print the state of variables that determine if entries are allowed
        # logger.info("🔍 CRITICAL ENTRY CONDITION DEBUG:")
        # logger.info(f"  Last trade exit time: {self.last_trade_exit.get('time')}")
        # logger.info(f"  Last trade exit reason: {self.last_trade_exit.get('reason')}")
        # logger.info(f"  Last trade was take profit: {'✅ Yes' if last_was_tp else '❌ No'}")
        # if last_was_tp:
        #     logger.info(f"  Price touched st_short since TP: {'✅ Yes' if price_touched_st_short else '❌ No'}")
        #     if not since_last_trade.empty:
        #         # Show the current price vs st_short to understand how close we are
        #         current_price = current_bar['Close']
        #         current_st_short = current_bar.get('st_short', 0)
        #         logger.info(f"  Current price: {current_price:.2f}, Current st_short: {current_st_short:.2f}")
        #         logger.info(f"  Distance to st_short: {abs(current_price - current_st_short):.2f} points")
        #         logger.info(f"  Bars since last trade: {len(since_last_trade)}")
        # logger.info(f"  ✨ Final check_entries result: {'✅ ALLOWED' if check_entries else '❌ BLOCKED'}")
        # logger.info("  ⚠️ If check_entries is BLOCKED, no trades will be generated even with 100% signal strength")
        
        # # Additional debug: Show what happens with entry conditions
        # logger.info(f"  📊 Entry condition check: {'✅ WILL CHECK' if check_entries else '❌ WILL SKIP'}")
        # if not check_entries:
        #     logger.info("  🚫 ENTRY CONDITIONS WILL BE SKIPPED - No trades will be generated")
        # else:
        #     logger.info("  ✅ ENTRY CONDITIONS WILL BE CHECKED - Trades may be generated if conditions are met")
        
        # # Check long conditions across all enabled logics
        # # Only check if check_entries is True (respects TP cooldown)
        # # Use OR between conditions so any enabled logic can trigger a trade
        # long_conditions = []
        # if check_entries:  # Only check if entries are allowed
        #     if self._trade_logic_base:
        #         long_conditions.append(self._check_base_logic_long(current_bar, prev_bar, True))
        #     if self._trade_logic_base_multi_timeframe:
        #         long_conditions.append(self._check_base_multi_timeframe_long(current_bar, prev_bar, True))
        #     if self._trade_logic_marketstructure:
        #         long_conditions.append(self._check_market_structure_long(current_bar, prev_bar))
        #     if self._trade_logic_pivots:
        #         long_conditions.append(self._check_pivots_long(current_bar, prev_bar, {}))
        #     if self._trade_logic_cpr:
        #         long_conditions.append(self._check_cpr_long(current_bar, prev_bar, {}))
            
        # # If any enabled logic returns True, we have a long signal
        # long_condition = any(long_conditions) if long_conditions else False
        
        # # SuperTrend signal flag logic for LONG trades
        # if long_condition and not self._pending_long_signal:
        #     # Set the signal flag when a signal is generated
        #     self._pending_long_signal = True
        #     self._pending_signal_timestamp = current_time
        #     logger.info(f"Generate Signals: 🚩 Setting LONG signal flag to True at {current_time}")
        
        # # Check if a candle has touched the ST_Short level for LONG entry
        # if self._pending_long_signal:
        #     # Check if price has touched ST_Short level
        #     price_touched_st_short = current_bar['Low'] <= current_bar['st_short']
            
        #     # Reset signal flag if candle closes below ST_Long
        #     if current_bar['Close'] < current_bar['st_long']:
        #         self._pending_long_signal = False
        #         logger.info(f"Generate Signals: 🚩 Resetting LONG signal flag to False - candle closed below ST_Long")
            
        #     # Only allow entry if price has touched ST_Short
        #     if not price_touched_st_short:
        #         long_condition = False
        #         logger.info(f"Generate Signals: 🚫 LONG entry blocked - waiting for price to touch ST_Short level")
        #     else:
        #         logger.info(f"Generate Signals: ✅ LONG entry allowed - price touched ST_Short level")
        
        # # Check short conditions across all enabled logics
        # # Only check if check_entries is True (respects TP cooldown)
        # # Use OR between conditions so any enabled logic can trigger a trade
        # short_conditions = []
        # if check_entries:  # Only check if entries are allowed
        #     if self._trade_logic_base:
        #         short_conditions.append(self._check_base_logic_short(current_bar, prev_bar, True))
        #     if self._trade_logic_base_multi_timeframe:
        #         short_conditions.append(self._check_base_multi_timeframe_short(current_bar, prev_bar, True))
        #     if self._trade_logic_marketstructure:
        #         short_conditions.append(self._check_market_structure_short(current_bar, prev_bar))
        #     if self._trade_logic_pivots:
        #         short_conditions.append(self._check_pivots_short(current_bar, prev_bar, {}))
        #     if self._trade_logic_cpr:
        #         short_conditions.append(self._check_cpr_short(current_bar, prev_bar, {}))
            
        # # If any enabled logic returns True, we have a short signal
        # short_condition = any(short_conditions) if short_conditions else False
        
        # # SuperTrend signal flag logic for SHORT trades
        # if short_condition and not self._pending_short_signal:
        #     # Set the signal flag when a signal is generated
        #     self._pending_short_signal = True
        #     self._pending_signal_timestamp = current_time
        #     logger.info(f"Generate Signals: 🚩 Setting SHORT signal flag to True at {current_time}")
        
        # # Check if a candle has touched the ST_Short level for SHORT entry
        # if self._pending_short_signal:
        #     # Check if price has touched ST_Short level
        #     price_touched_st_short = current_bar['High'] >= current_bar['st_short']
            
        #     # Reset signal flag if candle closes above ST_Long
        #     if current_bar['Close'] > current_bar['st_long']:
        #         self._pending_short_signal = False
        #         logger.info(f"Generate Signals: 🚩 Resetting SHORT signal flag to False - candle closed above ST_Long")
            
        #     # Only allow entry if price has touched ST_Short
        #     if not price_touched_st_short:
        #         short_condition = False
        #         logger.info(f"Generate Signals: 🚫 SHORT entry blocked - waiting for price to touch ST_Short level")
        #     else:
        #         logger.info(f"Generate Signals: ✅ SHORT entry allowed - price touched ST_Short level")
        
        # # Debug logging
        # logger.debug(f"Generate Signals: Long conditions: base={self._check_base_logic_long(current_bar, prev_bar, check_entries) if self._trade_logic_base else 'disabled'}, "
        #            f"base_mtf={self._check_base_multi_timeframe_long(current_bar, prev_bar, check_entries) if self._trade_logic_base_multi_timeframe else 'disabled'}, "
        #            f"market_structure={self._check_market_structure_long(current_bar, prev_bar) if self._trade_logic_marketstructure else 'disabled'}, "
        #            f"pivots={self._check_pivots_long(current_bar, prev_bar, {}) if self._trade_logic_pivots else 'disabled'}, "
        #            f"cpr={self._check_cpr_long(current_bar, prev_bar, {}) if self._trade_logic_cpr else 'disabled'}")
        # logger.debug(f"Generate Signals: Short conditions: base={self._check_base_logic_short(current_bar, prev_bar, check_entries) if self._trade_logic_base else 'disabled'}, "
        #            f"base_mtf={self._check_base_multi_timeframe_short(current_bar, prev_bar, check_entries) if self._trade_logic_base_multi_timeframe else 'disabled'}, "
        #            f"market_structure={self._check_market_structure_short(current_bar, prev_bar) if self._trade_logic_marketstructure else 'disabled'}, "
        #            f"pivots={self._check_pivots_short(current_bar, prev_bar, {}) if self._trade_logic_pivots else 'disabled'}, "
        #            f"cpr={self._check_cpr_short(current_bar, prev_bar, {}) if self._trade_logic_cpr else 'disabled'}")
        # logger.debug(f"Generate Signals: Final signals - Long: {long_condition}, Short: {short_condition}")
        
        # signals = []

        # # Check trade limits and cooldown for LONG signals
        # if long_condition:
        #     current_timestamp = df.index[-1]

        #     # FIRST: Check max trades limit (most important constraint - always enforced)
        #     can_take_new_trades_now = len(self._open_positions) < self._max_open_trades
        #     if not can_take_new_trades_now:
        #         logger.info(f"Generate Signals: 🚫 LONG signal BLOCKED - Max trades limit: {len(self._open_positions)}/{self._max_open_trades} positions open")
        #         long_condition = False  # Don't create trade signal, but analysis was logged

        #     # SECOND: Check cooldown conditions (only if max trades limit allows)
        #     if long_condition:
        #         last_long_time = self._last_signal_time.get('LONG')

        #         # Check bar cooldown period
        #         if last_long_time is not None:
        #             # Calculate bars since last LONG signal
        #             bars_since_last = len(df[df.index > last_long_time])
        #             if bars_since_last < self._signal_cooldown_bars:
        #                 logger.info(f"Generate Signals: 🚫 LONG signal BLOCKED - Bar cooldown: {bars_since_last} bars since last (need {self._signal_cooldown_bars})")
        #                 long_condition = False

        #         # # Check time-based cooldown (minutes) for any trade
        #         # if long_condition and self._last_trade_time is not None:
        #         #     time_since_last_trade = (current_timestamp - self._last_trade_time).total_seconds() / 60.0
        #         #     if time_since_last_trade < self._signal_cooldown_minutes:
        #         #         logger.info(f"🚫 LONG signal BLOCKED - Time cooldown: {time_since_last_trade:.1f} minutes since last trade (need {self._signal_cooldown_minutes})")
        #         #         long_condition = False
            
        #     signal_strength_long = self._calculate_signal_strength_long(current_bar, prev_bar)
        #     if long_condition and check_entries:
        #         if not signal_strength_long['required_conditions_met']:
        #             logger.info("Generate Signals: 🚫 LONG signal BLOCKED - Not all required conditions met")
        #             long_condition = False            

        #     # Finally check if we should enter trades
        #     if long_condition and check_entries:
        #         # Log detailed entry conditions for LONG signal
        #         entry_details = self._get_entry_condition_details(current_bar, 'LONG')
        #         logger.info("=" * 60)
        #         logger.info("🟢 GENERATING LONG SIGNAL")
        #         logger.info("=" * 60)
        #         logger.info(f"📊 Entry Conditions Analysis:")
        #         for key, value in entry_details.items():
        #             print(f"{key}: {value}")
        #             ascii_value = (
        #                 value.replace('✅', '[OK]')
        #                      .replace('❌', '[X]')
        #                      .replace('⚪', '[ ]')
        #                      .replace('⚠️', '[!]')
        #                      .replace('🚪', '[EXIT]')
        #                      .replace('📊', '[STATS]')
        #                      .replace('📈', '[UP]')
        #                      .replace('📉', '[DOWN]')
        #                      .replace('≤', '<=')
        #                      .replace('≥', '>=')
        #                      .replace('–', '-')
        #             )
        #             logger.info(f"{key}: {ascii_value}")
        #         logger.info(f"💰 Entry Price: {current_bar['Close']:.2f}")
        #         logger.info(f"🛑 Stop Loss: {current_bar['Close'] - (current_bar.get('atr', 0.01) * self._atr_trailing_multiplier):.2f}")
        #         logger.info(f"🎯 Take Profit: Will be calculated based on ATR multiplier ({self.atr_tp_multiplier}x)")
        #         logger.info("=" * 60)

        #         # Note: _last_signal_time will be updated AFTER position is opened

        #         # Use ATR-based trading for consistent risk management
        #         stop_loss, take_profit, position_size = self.calculate_atr_based_levels(current_bar, TradeDirection.LONG)
                
        #         logger.debug(f"[DEBUG] Calculated SL: {stop_loss}, TP: {take_profit}, Size: {position_size}")

        #         # Check if ATR-based levels were calculated successfully
        #         if stop_loss is None or take_profit is None or position_size is None:
        #             logger.warning("Generate Signals: ATR-based levels calculation failed - skipping trade")
        #             return signals
                
        #         # Create signal with ATR-based levels
        #         signal = TradeSignal(
        #             direction=TradeDirection.LONG,
        #             entry=current_bar['Close'],
        #             stop_loss=stop_loss,
        #             take_profit=take_profit,
        #             timestamp=df.index[-1],
        #             symbol=self.symbol,
        #             trailing_stop=self.use_trailing_stop,
        #             atr_trailing_multiplier=self.atr_trailing_multiplier,
        #             highest_high=current_bar['High'],
        #             lowest_low=current_bar['Low']
        #         )
        #         signal.position_size = position_size
        #         signal._strategy = self
        #         signals.append(signal)

        #         # Create and track position with magic number
        #         position = self.open_position(
        #             direction=TradeDirection.LONG,
        #             entry_price=current_bar['Close'],
        #             stop_loss=signal.stop_loss,
        #             take_profit=signal.take_profit,
        #             position_size=position_size,
        #             timestamp=df.index[-1],
        #             symbol=signal.symbol,
        #             trailing_stop=signal.trailing_stop,
        #             atr_trailing_multiplier=signal.atr_trailing_multiplier
        #         )
                
        #         # Check if position was created successfully
        #         if position is None:
        #             logger.error("Generate Signals: Failed to create position - MT5 order_send returned None")
        #             # Skip this signal and continue with the next one
        #             return signals  # Return current signals and exit early
                
        #         #logger.info(f"   Risk: ${risk_amount:.2f}")
        #         #logger.info(f"   Commission: ${commission:.2f}")
            
        # # Check trade limits and cooldown for SHORT signals
        # if short_condition:
        #     current_timestamp = df.index[-1]

        #     # FIRST: Check max trades limit (most important constraint - always enforced)
        #     can_take_new_trades_now = len(self._open_positions) < self._max_open_trades
        #     if not can_take_new_trades_now:
        #         logger.info(f"Generate Signals: 🚫 SHORT signal BLOCKED - Max trades limit: {len(self._open_positions)}/{self._max_open_trades} positions open")
        #         short_condition = False  # Don't create trade signal, but analysis was logged

        #     # SECOND: Check cooldown conditions (only if max trades limit allows)
        #     if short_condition:
        #         last_short_time = self._last_signal_time.get('SHORT')

        #         # Check bar cooldown period
        #         if last_short_time is not None:
        #             # Calculate bars since last SHORT signal
        #             bars_since_last = len(df[df.index > last_short_time])
        #             if bars_since_last < self._signal_cooldown_bars:
        #                 logger.info(f"Generate Signals: 🚫 SHORT signal BLOCKED - Bar cooldown: {bars_since_last} bars since last (need {self._signal_cooldown_bars})")
        #                 short_condition = False

        #         # # Check time-based cooldown (minutes) for any trade
        #         # if short_condition and self._last_trade_time is not None:
        #         #     time_since_last_trade = (current_timestamp - self._last_trade_time).total_seconds() / 60.0
        #         #     if time_since_last_trade < self._signal_cooldown_minutes:
        #         #         logger.info(f"🚫 SHORT signal BLOCKED - Time cooldown: {time_since_last_trade:.1f} minutes since last trade (need {self._signal_cooldown_minutes})")
        #         #         short_condition = False
        
        #     signal_strength_short = self._calculate_signal_strength_short(current_bar, prev_bar)
        #     if short_condition and check_entries:
        #         if not signal_strength_short['required_conditions_met']:
        #             logger.info("Generate Signals: 🚫 SHORT signal BLOCKED - Not all required conditions met")
        #             short_condition = False

        #     # Finally check if we should enter trades
        #     if short_condition and check_entries:
        #         # Log detailed entry conditions for SHORT signal
        #         entry_details = self._get_entry_condition_details(current_bar, 'SHORT')
        #         logger.info("=" * 60)
        #         logger.info("🔴 GENERATING SHORT SIGNAL")
        #         logger.info("=" * 60)
        #         logger.info(f"📊 Entry Conditions Analysis:")
        #         for key, value in entry_details.items():
        #             print(f"{key}: {value}")
        #             ascii_value = (
        #                 value.replace('✅', '[OK]')
        #                      .replace('❌', '[X]')
        #                      .replace('⚪', '[ ]')
        #                      .replace('⚠️', '[!]')
        #                      .replace('🚪', '[EXIT]')
        #                      .replace('📊', '[STATS]')
        #                      .replace('📈', '[UP]')
        #                      .replace('📉', '[DOWN]')
        #                      .replace('≤', '<=')
        #                      .replace('≥', '>=')
        #                      .replace('–', '-')
        #             )
        #             logger.info(f"{key}: {ascii_value}")
        #         logger.info(f"💰 Entry Price: {current_bar['Close']:.2f}")
        #         logger.info(f"🛑 Stop Loss: Will be calculated based on ATR multiplier ({self.atr_stop_multiplier}x)")
        #         logger.info(f"🎯 Take Profit: Will be calculated based on ATR multiplier ({self.atr_tp_multiplier}x)")
        #         logger.info("=" * 60)

        #         # Note: _last_signal_time will be updated AFTER position is opened

        #         # Use ATR-based trading for consistent risk management
        #         stop_loss, take_profit, position_size = self.calculate_atr_based_levels(current_bar, TradeDirection.SHORT)
                
        #         # Check if ATR-based levels were calculated successfully
        #         if stop_loss is None or take_profit is None or position_size is None:
        #             logger.warning("Generate Signals: ATR-based levels calculation failed - skipping trade")
        #             return signals
                
        #         # Create signal with ATR-based levels
        #         signal = TradeSignal(
        #             direction=TradeDirection.SHORT,
        #             entry=current_bar['Close'],
        #             stop_loss=stop_loss,
        #             take_profit=take_profit,
        #             timestamp=df.index[-1],
        #             symbol=self.symbol,
        #             trailing_stop=self.use_trailing_stop,
        #             atr_trailing_multiplier=self.atr_trailing_multiplier,
        #             highest_high=current_bar['High'],
        #             lowest_low=current_bar['Low']
        #         )
        #         signal.position_size = position_size
        #         signal._strategy = self
        #         signals.append(signal)

        #         # Create and track position with magic number
        #         position = self.open_position(
        #             direction=TradeDirection.SHORT,
        #             entry_price=current_bar['Close'],
        #             stop_loss=signal.stop_loss,
        #             take_profit=signal.take_profit,
        #             position_size=position_size,
        #             timestamp=df.index[-1],
        #             symbol=signal.symbol,
        #             trailing_stop=signal.trailing_stop,
        #             atr_trailing_multiplier=signal.atr_trailing_multiplier
        #         )
                
        #         # Check if position was created successfully
        #         if position is None:
        #             logger.error("Generate Signals: Failed to create position - MT5 order_send returned None")
        #             # Skip this signal and continue with the next one
        #             return signals  # Return current signals and exit early
                
        #         #logger.info(f"   Risk: ${risk_amount:.2f}")
        #         #logger.info(f"   Commission: ${commission:.2f}")

        # account_info = self.mt5_manager.get_account_info()
        # if account_info:
        #     balance = account_info['balance']
        #     equity = account_info['equity']
        #     free_margin = account_info['free_margin']
        #     # ... use as needed ...
        # else:
        #     balance = equity = free_margin = None  # or handle error

        # # Print open positions status after signal generation
        # self.print_open_positions_status(current_price=current_bar['Close'],
        #     current_balance= balance,
        #     current_equity= equity,
        #     current_free_margin= free_margin,
        #     current_bar=current_bar
        # )
        # return signals
    
    def debug_signal_conditions_mtf(self, direction: str, current_bar, prev_bar):
        """
        Dynamically display all multi-timeframe conditions, their status, config, and the actual values being compared for the given direction.
        Now rounds all numeric values to 2 decimals for better readability.
        """
        def fmt(val):
            try:
                return f"{float(val):.2f}"
            except (ValueError, TypeError):
                return str(val)

        if direction.upper() == "LONG":
            strength_data = self._calculate_signal_strength_long(current_bar, prev_bar)
            values = {
                'ltf_price_above_sma_fast': (fmt(current_bar['Close']), '>', fmt(current_bar.get('sma_fast', 0))),
                'ltf_sma_fast_above_slow': (fmt(current_bar.get('sma_fast', 0)), '>', fmt(current_bar.get('sma_slow', 0))),
                'ltf_supertrend_bullish': (f"Dir={current_bar.get('st_long_direction', 0)}", '==1 and', f"{fmt(current_bar['Close'])} > {fmt(current_bar.get('st_long', 0))}"),
                'ltf_rsi_above_rsi_ma': (fmt(current_bar.get('rsi', 0)), '>', fmt(current_bar.get('rsi_ma', 0))),
                'ltf_adx_strength': (fmt(current_bar.get('adx', 0)), '>=', fmt(getattr(self, '_min_adx_for_entry', 0))),
                'ltf_adx_above_adxr': (fmt(current_bar.get('adx', 0)), '>', fmt(current_bar.get('adxr', 0))),
                'ltf_di_bullish': (fmt(current_bar.get('di_plus', 0)), '>', fmt(current_bar.get('di_minus', 0))),
                #'htf_price_above_sma_fast': (fmt(current_bar['Close']), '>', fmt(current_bar.get('sma_fast_htf', 0))),
                #'htf_sma_fast_above_slow': (fmt(current_bar.get('sma_fast_htf', 0)), '>', fmt(current_bar.get('sma_slow_htf', 0))),
                #'htf_supertrend_bullish': (f"Dir={current_bar.get('super_trend_long_direction_htf', 0)}", '==1 and', f"{fmt(current_bar['Close'])} > {fmt(current_bar.get('super_trend_long_htf', 0))}"),
                #'htf_rsi_above_rsi_ma': (fmt(current_bar.get('rsi_htf', 0)), '>', fmt(current_bar.get('rsi_ma_htf', 0))),
                #'htf_adx_strength': (fmt(current_bar.get('adx_htf', 0)), '>=', fmt(getattr(self, '_min_adx_for_entry', 0))),
                #'htf_adx_above_adxr': (fmt(current_bar.get('adx_htf', 0)), '>', fmt(current_bar.get('adxr_htf', 0))),
                #'htf_di_bullish': (fmt(current_bar.get('di_plus_htf', 0)), '>', fmt(current_bar.get('di_minus_htf', 0))),
            }
        else:
            strength_data = self._calculate_signal_strength_short(current_bar, prev_bar)
            values = {
                'ltf_price_below_sma_fast': (fmt(current_bar['Close']), '<', fmt(current_bar.get('sma_fast', 0))),
                'ltf_sma_fast_below_slow': (fmt(current_bar.get('sma_fast', 0)), '<', fmt(current_bar.get('sma_slow', 0))),
                'ltf_supertrend_bearish': (f"Dir={current_bar.get('st_short_direction', 0)}", '==-1 and', f"{fmt(current_bar['Close'])} < {fmt(current_bar.get('st_short', 0))}"),
                'ltf_rsi_below_rsi_ma': (fmt(current_bar.get('rsi', 0)), '<', fmt(current_bar.get('rsi_ma', 0))),
                'ltf_adx_strength': (fmt(current_bar.get('adx', 0)), '>=', fmt(getattr(self, '_min_adx_for_entry', 0))),
                'ltf_adx_above_adxr': (fmt(current_bar.get('adx', 0)), '>', fmt(current_bar.get('adxr', 0))),
                'ltf_di_bearish': (fmt(current_bar.get('di_minus', 0)), '>', fmt(current_bar.get('di_plus', 0))),
                #'htf_price_below_sma_fast': (fmt(current_bar['Close']), '<', fmt(current_bar.get('sma_fast_htf', 0))),
                #'htf_sma_fast_below_slow': (fmt(current_bar.get('sma_fast_htf', 0)), '<', fmt(current_bar.get('sma_slow_htf', 0))),
                #'htf_supertrend_bearish': (f"Dir={current_bar.get('super_trend_short_direction_htf', 0)}", '==-1 and', f"{fmt(current_bar['Close'])} < {fmt(current_bar.get('super_trend_short_htf', 0))}"),
                #'htf_rsi_below_rsi_ma': (fmt(current_bar.get('rsi_htf', 0)), '<', fmt(current_bar.get('rsi_ma_htf', 0))),
                #'htf_adx_strength': (fmt(current_bar.get('adx_htf', 0)), '>=', fmt(getattr(self, '_min_adx_for_entry', 0))),
                #'htf_adx_above_adxr': (fmt(current_bar.get('adx_htf', 0)), '>', fmt(current_bar.get('adxr_htf', 0))),
                #'htf_di_bearish': (fmt(current_bar.get('di_minus_htf', 0)), '>', fmt(current_bar.get('di_plus_htf', 0))),
            }

        detailed_conditions = strength_data['detailed_conditions']
        condition_config = strength_data['condition_config']
        
        # Price information section
        print("=" * 125)
        print("📊 PRICE INFORMATION:".ljust(55) + "|🔍 " + f"{direction.upper()} SIGNAL CONDITIONS")
        print(f" 📈 OHLC: O:{fmt(current_bar['Open'])} H:{fmt(current_bar['High'])} L:{fmt(current_bar['Low'])} C:{fmt(current_bar['Close'])}".ljust(55) + "|", end="")
        
        # Display first condition
        cond_items = list(detailed_conditions.items())
        if cond_items:
            cond, value = cond_items[0]
            config = condition_config.get(cond, {})
            status = "✅" if value else "❌"
            req_flag = "[REQ]" if config.get("required", False) else "[OPT]"
            weight = config.get("weight", 0)
            val_tuple = values.get(cond, ("N/A", "?", "N/A"))
            print(f" {status} {cond.replace('_', ' ').title():20} {req_flag} ({val_tuple[0]} {val_tuple[1]} {val_tuple[2]})")
        else:
            print()
        
        # Previous bar info and second condition
        print(f" 📉 Previous: O:{fmt(prev_bar['Open'])} H:{fmt(prev_bar['High'])} L:{fmt(prev_bar['Low'])} C:{fmt(prev_bar['Close'])}".ljust(55) + "|", end="")
        if len(cond_items) > 1:
            cond, value = cond_items[1]
            config = condition_config.get(cond, {})
            status = "✅" if value else "❌"
            req_flag = "[REQ]" if config.get("required", False) else "[OPT]"
            weight = config.get("weight", 0)
            val_tuple = values.get(cond, ("N/A", "?", "N/A"))
            print(f" {status} {cond.replace('_', ' ').title():20} {req_flag} ({val_tuple[0]} {val_tuple[1]} {val_tuple[2]})")
        else:
            print()
        
        # Price change and third condition
        change = float(current_bar['Close']) - float(prev_bar['Close'])
        change_pct = (change / float(prev_bar['Close'])) * 100 if float(prev_bar['Close']) != 0 else 0
        print(f" 🔄 Change: {change:.2f} ({change_pct:.2f}%)".ljust(55) + "|", end="")
        if len(cond_items) > 2:
            cond, value = cond_items[2]
            config = condition_config.get(cond, {})
            status = "✅" if value else "❌"
            req_flag = "[REQ]" if config.get("required", False) else "[OPT]"
            weight = config.get("weight", 0)
            val_tuple = values.get(cond, ("N/A", "?", "N/A"))
            print(f" {status} {cond.replace('_', ' ').title():20} {req_flag} ({val_tuple[0]} {val_tuple[1]} {val_tuple[2]})")
        else:
            print()
        
        # Range and fourth condition
        price_range = float(current_bar['High']) - float(current_bar['Low'])
        print(f" 📏 Range: {price_range:.2f}".ljust(55) + "|", end="")
        if len(cond_items) > 3:
            cond, value = cond_items[3]
            config = condition_config.get(cond, {})
            status = "✅" if value else "❌"
            req_flag = "[REQ]" if config.get("required", False) else "[OPT]"
            weight = config.get("weight", 0)
            val_tuple = values.get(cond, ("N/A", "?", "N/A"))
            print(f" {status} {cond.replace('_', ' ').title():20} {req_flag} ({val_tuple[0]} {val_tuple[1]} {val_tuple[2]})")
        else:
            print()
        
        # SMA indicators and fifth condition
        print(f" 📊 Indicators: SMA Fast: {fmt(current_bar.get('sma_fast', 0))}, SMA Slow: {fmt(current_bar.get('sma_slow', 0))}".ljust(55) + "|", end="")
        if len(cond_items) > 4:
            cond, value = cond_items[4]
            config = condition_config.get(cond, {})
            status = "✅" if value else "❌"
            req_flag = "[REQ]" if config.get("required", False) else "[OPT]"
            weight = config.get("weight", 0)
            val_tuple = values.get(cond, ("N/A", "?", "N/A"))
            print(f" {status} {cond.replace('_', ' ').title():20} {req_flag} ({val_tuple[0]} {val_tuple[1]} {val_tuple[2]})")
        else:
            print()
        
        # ADX indicators and sixth condition
        print(f" 📊 ADX: {fmt(current_bar.get('adx', 0))}, ADXR: {fmt(current_bar.get('adxr', 0))}, +DI: {fmt(current_bar.get('di_plus', 0))}, -DI: {fmt(current_bar.get('di_minus', 0))}".ljust(55) + "|", end="")
        if len(cond_items) > 5:
            cond, value = cond_items[5]
            config = condition_config.get(cond, {})
            status = "✅" if value else "❌"
            req_flag = "[REQ]" if config.get("required", False) else "[OPT]"
            weight = config.get("weight", 0)
            val_tuple = values.get(cond, ("N/A", "?", "N/A"))
            print(f" {status} {cond.replace('_', ' ').title():20} {req_flag} ({val_tuple[0]} {val_tuple[1]} {val_tuple[2]})")
        else:
            print()
        
        # SuperTrend indicators and seventh condition
        if direction.upper() == "LONG":
            print(f" 📊 SuperTrend: Long: {fmt(current_bar.get('st_long', 0))}, Short: {fmt(current_bar.get('st_short', 0))}".ljust(55) + "|", end="")
        else:
            print(f" 📊 SuperTrend: Long: {fmt(current_bar.get('st_long', 0))}, Short: {fmt(current_bar.get('st_short', 0))}".ljust(55) + "|", end="")
        if len(cond_items) > 6:
            cond, value = cond_items[6]
            config = condition_config.get(cond, {})
            status = "✅" if value else "❌"
            req_flag = "[REQ]" if config.get("required", False) else "[OPT]"
            weight = config.get("weight", 0)
            val_tuple = values.get(cond, ("N/A", "?", "N/A"))
            print(f" {status} {cond.replace('_', ' ').title():20} {req_flag} ({val_tuple[0]} {val_tuple[1]} {val_tuple[2]})")
        else:
            print()
        
        # RSI indicators and eighth condition (if available)
        print(f" 📊 RSI: {fmt(current_bar.get('rsi', 0))}, RSI MA: {fmt(current_bar.get('rsi_ma', 0))}".ljust(55) + "|")
        
        # ATR indicator
        print(f" 📊 ATR: {fmt(current_bar.get('atr', 0))}".ljust(55) + "|")
        
        print("=" * 125)
        
        # Signal strength summary        
        print(f"📊 Signal Strength: {strength_data['strength']:.1f}% | Required conditions met: {'✅ YES' if strength_data['required_conditions_met'] else '❌ NO'}")
        print(f"📈 Favorable: {strength_data['favorable_count']} / {strength_data['total_conditions']}")
        #print(f"Signal pending flag active: {'✅ YES' if pending_flag else '❌ NO'}")
        st_level = current_bar['st_short'] if 'st_short' in current_bar else None
        
        if direction.upper() == "LONG":
            print(f"🚩 Signal pending flag active: {'✅ YES' if self._pending_long_signal else '❌ NO'}")
            if self._pending_long_signal and self._pending_signal_timestamp is not None and st_level is not None:
                print(f"⏱️  Signal Flag set at: {self._pending_signal_timestamp}")                
        
        if direction.upper() == "SHORT":
            print(f"🚩 Signal pending flag active: {'✅ YES' if self._pending_short_signal else '❌ NO'}")
            if self._pending_short_signal and self._pending_signal_timestamp is not None and st_level is not None:
                print(f"⏱️  Signal Flag set at: {self._pending_signal_timestamp}")                
        if self._pending_long_signal or self._pending_short_signal:
            print(f"⏳ {direction.upper()} entry Waiting for price to touch ST level: {st_level:.2f}")            

        if self._pending_long_signal :
            print(f"🎯 ST_Short: {current_bar['st_short']:.2f}, Candle Low: {current_bar['Low']:.2f}")
            print(f"📏 Distance to entry: {current_bar['Low'] - current_bar['st_short']:.2f} points")

        if self._pending_short_signal :
            print(f"🎯 ST_Short: {current_bar['st_short']:.2f}, Candle High: {current_bar['High']:.2f}")
            print(f"📏 Distance to entry: {current_bar['st_short'] - current_bar['High']:.2f} points")
        
        # Display Conditions after touching ST_Short
        st_touch_long = self._pending_long_signal and current_bar['Low'] <= current_bar['st_short']
        st_touch_short = self._pending_short_signal and current_bar['High'] >= current_bar['st_short']	       
        
        # Check RSI conditions separately for better logging
        rsi_condition_long = current_bar['rsi'] > current_bar['rsi_ma']
        rsi_condition_short = current_bar['rsi'] <current_bar['rsi_ma']		
        # Check ADX > ADXR condition
        adx_condition_long = current_bar['adx'] > current_bar['adxr']
        adx_condition_short = current_bar['adx'] > current_bar['adxr']	
        
        # Print RSI condition status when ST touch condition is met
        if st_touch_long and self._pending_long_signal:
            print(f"🔄 ST Touch Long: {'✅ Yes' if st_touch_long else '❌ No'}")
            print(f" LONG RSI Check: RSI({current_bar['rsi']:.2f}) {'>' if rsi_condition_long else '<='} RSI_MA({current_bar['rsi_ma']:.2f}) - {'✅ PASSED' if rsi_condition_long else '❌ FAILED'}")
        if st_touch_short and self._pending_short_signal:
            print(f"🔄 ST Touch Short: {'✅ Yes' if st_touch_short else '❌ No'}")
            print(f" SHORT RSI Check: RSI({current_bar['rsi']:.2f}) {'<' if rsi_condition_short else '>='} RSI_MA({current_bar['rsi_ma']:.2f}) - {'✅ PASSED' if rsi_condition_short else '❌ FAILED'}")
        	
        # Print ADX condition status when ST touch condition is met
        if st_touch_long and self._pending_long_signal:
            print(f"🔄 ST Touch Long: {'✅ Yes' if st_touch_long else '❌ No'}")
            print(f" LONG ADX Check: ADX({current_bar['adx']:.2f}) {'>' if adx_condition_long else '<='} ADXR({current_bar['adxr']:.2f}) - {'✅ PASSED' if adx_condition_long else '❌ FAILED'}")
        if st_touch_short and self._pending_short_signal:
            print(f"🔄 ST Touch Short: {'✅ Yes' if st_touch_short else '❌ No'}")
            print(f" SHORT ADX Check: ADX({current_bar['adx']:.2f}) {'>' if adx_condition_short else '<='} ADXR({current_bar['adxr']:.2f}) - {'✅ PASSED' if adx_condition_short else '❌ FAILED'}")
        #print("-" * 60)
        
        # Account information section
        print("=" * 125)
        
        # Get account info from MT5
        account_info = mt5.account_info()
        if account_info:
            # Format account information with the requested format
            print(f"🔑 MT5 ACCOUNT: #{account_info.login} | 👤 Owner: {account_info.name} | 🌐 Server: {account_info.server}")
            print(f"💰 Balance: {account_info.balance:,.2f} | 📈 Equity: {account_info.equity:,.2f} | 💵 Free Margin: {account_info.margin_free:,.2f}")
            print("-" * 125)
            
            # Open positions section
            print("📊 OPEN POSITIONS:")
            print("-" * 125)
            
            # Check if there are any open positions
            positions = mt5.positions_get()
            if positions and len(positions) > 0:
                for pos in positions:
                    direction = "LONG" if pos.type == 0 else "SHORT"
                    print(f"Symbol: {pos.symbol} | Direction: {direction} | Volume: {pos.volume} | Open Price: {pos.price_open} | Current Price: {pos.price_current}")
            else:
                print("🔍 No open positions.")
            
            print("-" * 125)
            
            # Performance metrics - you may need to adjust these based on your actual tracking mechanism
            day_high = account_info.equity  # Placeholder, should be tracked throughout the day
            max_drawdown = 0  # Placeholder, should be calculated based on historical data
            max_drawdown_pct = 0  # Placeholder percentage
            
            print(f"📈 Performance: Day's High: {day_high:,.2f} | Max Draw Down: {max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")
        else:
            print("⚠️ MT5 account information not available")
        
        print("=" * 125)
    
    def debug_signal_conditions_mtf_testing(self, direction: str, current_bar, prev_bar):
        """
        Dynamically display all multi-timeframe conditions, their status, config, and the actual values being compared for the given direction.
        Now rounds all numeric values to 2 decimals for better readability.
        Uses logger.info for all output (testing/backtesting mode).
        """
        def fmt(val):
            try:
                return f"{float(val):.2f}"
            except (ValueError, TypeError):
                return str(val)

        if direction.upper() == "LONG":
            strength_data = self._calculate_signal_strength_long(current_bar, prev_bar)
            values = {
                'ltf_price_above_sma_fast': (fmt(current_bar['Close']), '>', fmt(current_bar.get('sma_fast', 0))),
                'ltf_sma_fast_above_slow': (fmt(current_bar.get('sma_fast', 0)), '>', fmt(current_bar.get('sma_slow', 0))),
                'ltf_supertrend_bullish': (f"Dir={current_bar.get('st_long_direction', 0)}", '==1 and', f"{fmt(current_bar['Close'])} > {fmt(current_bar.get('st_long', 0))}"),
                'ltf_rsi_above_rsi_ma': (fmt(current_bar.get('rsi', 0)), '>', fmt(current_bar.get('rsi_ma', 0))),
                'ltf_adx_strength': (fmt(current_bar.get('adx', 0)), '>=', fmt(getattr(self, '_min_adx_for_entry', 0))),
                'ltf_adx_above_adxr': (fmt(current_bar.get('adx', 0)), '>', fmt(current_bar.get('adxr', 0))),
                'ltf_di_bullish': (fmt(current_bar.get('di_plus', 0)), '>', fmt(current_bar.get('di_minus', 0))),
                'htf_price_above_sma_fast': (fmt(current_bar['Close']), '>', fmt(current_bar.get('sma_fast_htf', 0))),
                'htf_sma_fast_above_slow': (fmt(current_bar.get('sma_fast_htf', 0)), '>', fmt(current_bar.get('sma_slow_htf', 0))),
                'htf_supertrend_bullish': (f"Dir={current_bar.get('super_trend_long_direction_htf', 0)}", '==1 and', f"{fmt(current_bar['Close'])} > {fmt(current_bar.get('super_trend_long_htf', 0))}"),
                'htf_rsi_above_rsi_ma': (fmt(current_bar.get('rsi_htf', 0)), '>', fmt(current_bar.get('rsi_ma_htf', 0))),
                'htf_adx_strength': (fmt(current_bar.get('adx_htf', 0)), '>=', fmt(getattr(self, '_min_adx_for_entry', 0))),
                'htf_adx_above_adxr': (fmt(current_bar.get('adx_htf', 0)), '>', fmt(current_bar.get('adxr_htf', 0))),
                'htf_di_bullish': (fmt(current_bar.get('di_plus_htf', 0)), '>', fmt(current_bar.get('di_minus_htf', 0))),
            }
        else:
            strength_data = self._calculate_signal_strength_short(current_bar, prev_bar)
            values = {
                'ltf_price_below_sma_fast': (fmt(current_bar['Close']), '<', fmt(current_bar.get('sma_fast', 0))),
                'ltf_sma_fast_below_slow': (fmt(current_bar.get('sma_fast', 0)), '<', fmt(current_bar.get('sma_slow', 0))),
                'ltf_supertrend_bearish': (f"Dir={current_bar.get('st_short_direction', 0)}", '==-1 and', f"{fmt(current_bar['Close'])} < {fmt(current_bar.get('st_short', 0))}"),
                'ltf_rsi_below_rsi_ma': (fmt(current_bar.get('rsi', 0)), '<', fmt(current_bar.get('rsi_ma', 0))),
                'ltf_adx_strength': (fmt(current_bar.get('adx', 0)), '>=', fmt(getattr(self, '_min_adx_for_entry', 0))),
                'ltf_adx_above_adxr': (fmt(current_bar.get('adx', 0)), '>', fmt(current_bar.get('adxr', 0))),
                'ltf_di_bearish': (fmt(current_bar.get('di_minus', 0)), '>', fmt(current_bar.get('di_plus', 0))),
                'htf_price_below_sma_fast': (fmt(current_bar['Close']), '<', fmt(current_bar.get('sma_fast_htf', 0))),
                'htf_sma_fast_below_slow': (fmt(current_bar.get('sma_fast_htf', 0)), '<', fmt(current_bar.get('sma_slow_htf', 0))),
                'htf_supertrend_bearish': (f"Dir={current_bar.get('super_trend_short_direction_htf', 0)}", '==-1 and', f"{fmt(current_bar['Close'])} < {fmt(current_bar.get('super_trend_short_htf', 0))}"),
                'htf_rsi_below_rsi_ma': (fmt(current_bar.get('rsi_htf', 0)), '<', fmt(current_bar.get('rsi_ma_htf', 0))),
                'htf_adx_strength': (fmt(current_bar.get('adx_htf', 0)), '>=', fmt(getattr(self, '_min_adx_for_entry', 0))),
                'htf_adx_above_adxr': (fmt(current_bar.get('adx_htf', 0)), '>', fmt(current_bar.get('adxr_htf', 0))),
                'htf_di_bearish': (fmt(current_bar.get('di_minus_htf', 0)), '>', fmt(current_bar.get('di_plus_htf', 0))),
            }

        detailed_conditions = strength_data['detailed_conditions']
        condition_config = strength_data['condition_config']

        logger.info(f"\n🔍 {direction.upper()} MULTI-TIMEFRAME SIGNAL CONDITIONS")
        logger.info("-" * 60)
        for cond, value in detailed_conditions.items():
            config = condition_config.get(cond, {})
            status = "✅" if value else "❌"
            req_flag = "[REQ]" if config.get("required", False) else "[OPT]"
            weight = config.get("weight", 0)
            val_tuple = values.get(cond, ("N/A", "?", "N/A"))
            logger.info(f"{status} {cond.replace('_', ' ').title():35} {req_flag} (Weight: {weight})   ({val_tuple[0]} {val_tuple[1]} {val_tuple[2]})")
        logger.info("-" * 60)
        logger.info(f"Signal Strength: {strength_data['strength']:.1f}% | Required Met: {strength_data['required_conditions_met']}")
        logger.info(f"Favorable: {strength_data['favorable_count']} / {strength_data['total_conditions']}")
        logger.info("-" * 60)

    def _check_position_exit_conditions(self, position, current_bar, prev_bar=None):
        """
        Check all possible exit conditions for a position and return a list of active conditions.
        
        Args:
            position: TradePosition object
            current_bar: Current market data bar
            prev_bar: Previous market data bar (for SuperTrend checks)
            
        Returns:
            list: List of dictionaries with exit condition details
        """
        exit_conditions = []
        current_price = current_bar['Close']
        current_high = current_bar.get('High', current_price)
        current_low = current_bar.get('Low', current_price)
        
        # Handle trading mode differences
        if self._trading_mode == 'testing':
            # TESTING MODE: Use candle High/Low for realistic SL/TP detection
            high_price = current_high
            low_price = current_low
            close_price = current_price
        else:
            # LIVE MODE: Use current Close price only
            high_price = current_price
            low_price = current_price
            close_price = current_price
        
        # 1. Stop Loss Check
        if position.direction == TradeDirection.LONG:
            if low_price <= position.stop_loss:
                exit_conditions.append({
                    'type': 'Stop Loss',
                    'condition': f"Low ({low_price:.2f}) <= SL ({position.stop_loss:.2f})",
                    'status': 'ACTIVE',
                    'exit_price': position.stop_loss if self._trading_mode == 'testing' else close_price,
                    'reason': self.EXIT_SL_HIT
                })
            else:
                exit_conditions.append({
                    'type': 'Stop Loss',
                    'condition': f"Low ({low_price:.2f}) > SL ({position.stop_loss:.2f})",
                    'status': 'OK',
                    'exit_price': None,
                    'reason': None
                })
        else:  # SHORT
            if high_price >= position.stop_loss:
                exit_conditions.append({
                    'type': 'Stop Loss',
                    'condition': f"High ({high_price:.2f}) >= SL ({position.stop_loss:.2f})",
                    'status': 'ACTIVE',
                    'exit_price': position.stop_loss if self._trading_mode == 'testing' else close_price,
                    'reason': self.EXIT_SL_HIT
                })
            else:
                exit_conditions.append({
                    'type': 'Stop Loss',
                    'condition': f"High ({high_price:.2f}) < SL ({position.stop_loss:.2f})",
                    'status': 'OK',
                    'exit_price': None,
                    'reason': None
                })
        
        # 2. Take Profit Check
        if position.direction == TradeDirection.LONG:
            if high_price >= position.take_profit:
                exit_conditions.append({
                    'type': 'Take Profit',
                    'condition': f"High ({high_price:.2f}) >= TP ({position.take_profit:.2f})",
                    'status': 'ACTIVE',
                    'exit_price': position.take_profit if self._trading_mode == 'testing' else close_price,
                    'reason': self.EXIT_TP_HIT
                })
            else:
                exit_conditions.append({
                    'type': 'Take Profit',
                    'condition': f"High ({high_price:.2f}) < TP ({position.take_profit:.2f})",
                    'status': 'OK',
                    'exit_price': None,
                    'reason': None
                })
        else:  # SHORT
            if low_price <= position.take_profit:
                exit_conditions.append({
                    'type': 'Take Profit',
                    'condition': f"Low ({low_price:.2f}) <= TP ({position.take_profit:.2f})",
                    'status': 'ACTIVE',
                    'exit_price': position.take_profit if self._trading_mode == 'testing' else close_price,
                    'reason': self.EXIT_TP_HIT
                })
            else:
                exit_conditions.append({
                    'type': 'Take Profit',
                    'condition': f"Low ({low_price:.2f}) > TP ({position.take_profit:.2f})",
                    'status': 'OK',
                    'exit_price': None,
                    'reason': None
                })
        
        # 3. SuperTrend Exit Check
        if prev_bar is not None:
            if position.direction == TradeDirection.SHORT:
                if 'st_long' in prev_bar and prev_bar['Close'] > prev_bar['st_long']:
                    exit_conditions.append({
                        'type': 'SuperTrend Exit',
                        'condition': f"Prev Close ({prev_bar['Close']:.2f}) > ST_Long ({prev_bar['st_long']:.2f})",
                        'status': 'ACTIVE',
                        'exit_price': close_price,
                        'reason': self.EXIT_TSL_SUPERTREND
                    })
                else:
                    st_long_val = prev_bar.get('st_long', 'N/A')
                    exit_conditions.append({
                        'type': 'SuperTrend Exit',
                        'condition': f"Prev Close ({prev_bar['Close']:.2f}) <= ST_Long ({st_long_val})",
                        'status': 'OK',
                        'exit_price': None,
                        'reason': None
                    })
            else:  # LONG
                if 'st_long' in prev_bar and prev_bar['Close'] < prev_bar['st_long']:
                    exit_conditions.append({
                        'type': 'SuperTrend Exit',
                        'condition': f"Prev Close ({prev_bar['Close']:.2f}) < ST_Long ({prev_bar['st_long']:.2f})",
                        'status': 'ACTIVE',
                        'exit_price': close_price,
                        'reason': self.EXIT_TSL_SUPERTREND
                    })
                else:
                    st_long_val = prev_bar.get('st_long', 'N/A')
                    exit_conditions.append({
                        'type': 'SuperTrend Exit',
                        'condition': f"Prev Close ({prev_bar['Close']:.2f}) >= ST_Long ({st_long_val})",
                        'status': 'OK',
                        'exit_price': None,
                        'reason': None
                    })
        else:
            exit_conditions.append({
                'type': 'SuperTrend Exit',
                'condition': "No previous bar data available",
                'status': 'UNKNOWN',
                'exit_price': None,
                'reason': None
            })
        
        # 4. Trailing Stop Check (if enabled)
        if position.trailing_stop:
            if position.trailing_activated:
                exit_conditions.append({
                    'type': 'Trailing Stop',
                    'condition': f"Trailing activated (ATR: {current_bar.get('atr', 'N/A'):.2f})",
                    'status': 'ACTIVE',
                    'exit_price': position.stop_loss,
                    'reason': self.EXIT_TSL_ATR
                })
            else:
                exit_conditions.append({
                    'type': 'Trailing Stop',
                    'condition': "Trailing stop enabled but not activated",
                    'status': 'WAITING',
                    'exit_price': None,
                    'reason': None
                })
        else:
            exit_conditions.append({
                'type': 'Trailing Stop',
                'condition': "Trailing stop disabled",
                'status': 'DISABLED',
                'exit_price': None,
                'reason': None
            })
        
        return exit_conditions

    def print_open_positions_status(
    self,
    current_price=None,
    current_balance=None,
    current_equity=None,
    current_free_margin=None,
    current_bar=None,
    account_number=None,
    account_name=None,
    account_server=None):
        
        # Compact Account Status
        print("="*80)
        print(f"🔑 MT5 ACCOUNT: #{account_number} | 👤 Owner: {account_name} | 🌐 Server: {account_server}")
        print(f"💰 Balance: {current_balance:,.2f} | 📈 Equity: {current_equity:,.2f} | 💵 Free Margin: {current_free_margin:,.2f}")
        print("-" * 80)

        # Compact Open Positions Table
        print("📊 OPEN POSITIONS:")
        print("-" * 80)
        open_positions = getattr(self, '_open_positions', {})
        if not open_positions:
            print("🔍 No open positions.")
        else:
            # Track the last position for special formatting
            position_items = list(open_positions.items())
            last_magic, last_pos = position_items[-1] if position_items else (None, None)
            
            for magic, pos in position_items:
                direction = pos.direction.name if hasattr(pos.direction, 'name') else str(pos.direction)
                entry = pos.entry_price
                size = pos.position_size
                sl = pos.stop_loss
                tp = getattr(pos, 'take_profit', 0)
                trailing = "ON" if getattr(pos, 'trailing_stop', False) else "OFF"
                price = current_price if current_price is not None else getattr(pos, 'last_price', entry)
                pnl = (price - entry) * size * 100 if direction == 'LONG' else (entry - price) * size * 100
                pnl_icon = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                dir_icon = "🔼" if direction == "LONG" else "🔽"
                
                # Special formatting for the last position
                position_label = "📌 LATEST POSITION:" if magic == last_magic else "📎 POSITION:"
                
                if current_bar is not None:
                    o = current_bar.get('Open', 0)
                    h = current_bar.get('High', 0)
                    l = current_bar.get('Low', 0)
                    c = current_bar.get('Close', 0)
                    bar_str = f"{o:.2f}/{h:.2f}/{l:.2f}/{c:.2f}"
                else:
                    bar_str = "N/A"
                
                # First line: Position ID and direction info
                print(f"{position_label} Magic #{magic} | {dir_icon} {direction} | 📏 Size: {size:.2f} lots | {pnl_icon} P&L: {pnl:.2f}")
                
                # Second line: Price levels and current bar
                print(f"   💲 Entry: {entry:.2f} | 🛑 SL: {sl:.2f} | 🎯 TP: {tp:.2f} | 🔄 Trail: {trailing} | 📊 Bar: {bar_str}")
                
                # Add a separator between positions except after the last one
                if magic != last_magic:
                    print("   " + "-" * 74)
        print("-" * 80)

        # Exit Conditions Section
        if open_positions and current_bar is not None:
            print("\n🚪 EXIT CONDITIONS ANALYSIS:")
            print("-" * 80)
            
            # Get previous bar for SuperTrend checks
            df = getattr(self, '_current_df', None)
            prev_bar = None
            if df is not None and len(df) > 1:
                prev_bar = df.iloc[-2]
            
            for magic, pos in open_positions.items():
                print(f"\n📋 Position Magic {magic} ({pos.direction.name}):")
                print(f"   Entry: {pos.entry_price:.2f} | SL: {pos.stop_loss:.2f} | TP: {pos.take_profit:.2f}")
                
                # Check all exit conditions
                exit_conditions = self._check_position_exit_conditions(pos, current_bar, prev_bar)
                
                for condition in exit_conditions:
                    status_icon = {
                        'ACTIVE': '🔴',
                        'OK': '🟢', 
                        'WAITING': '🟡',
                        'DISABLED': '⚪',
                        'UNKNOWN': '❓'
                    }.get(condition['status'], '❓')
                    
                    print(f"   {status_icon} {condition['type']}: {condition['condition']}")
                    if condition['status'] == 'ACTIVE':
                        print(f"      → EXIT at {condition['exit_price']:.2f} ({self.get_exit_reason_description(condition['reason'])})")
        
        # Compact Performance Section
        if current_balance is not None:
            if not hasattr(self, '_todays_highest_balance'):
                self._todays_highest_balance = current_balance
            self._todays_highest_balance = max(self._todays_highest_balance, current_balance)
            drawdown = current_balance - self._todays_highest_balance
            if not hasattr(self, '_max_drawdown'):
                self._max_drawdown = 0
            self._max_drawdown = min(self._max_drawdown, drawdown)
            dd_pct = (self._max_drawdown / self._todays_highest_balance * 100) if self._todays_highest_balance else 0
            print(
                f"📈 Performance: Day's High: {self._todays_highest_balance:,.2f} | "
                f"Max Draw Down: {self._max_drawdown:,.2f} ({dd_pct:.2f}%)"
            )
        print("="*80)
        
    def generate_magic_number(self) -> int:
        """Generate a unique magic number for a new trade"""
        magic = self._magic_number + self._next_magic_offset
        self._next_magic_offset += 1
        return magic

    def restore_open_positions(self):
        """
        Restore open positions from MT5 into the internal tracking dict on startup.
        Only restores positions for the current symbol.
        Tries to restore internal SL from local storage if available.
        """
        import MetaTrader5 as mt5
        positions = mt5.positions_get()
        if positions is None:
            logger.error("Failed to get open positions from MT5")
            return
        # --- CLEANUP: Remove internal SLs for non-existent positions ---
        internal_sl_path = os.path.abspath("internal_sls.json")
        if os.path.exists(internal_sl_path):
            try:
                with open(internal_sl_path, "r") as f:
                    internal_sls = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load internal SLs for cleanup: {e}")
                internal_sls = {}
            mt5_magics = {pos.magic for pos in positions}
            removed = 0
            for magic in list(internal_sls.keys()):
                if int(magic) not in mt5_magics:
                    logger.info(f"Cleaning up orphaned internal SL for magic {magic} (no such position in MT5)")
                    del internal_sls[magic]
                    removed += 1
            if removed > 0:
                with open(internal_sl_path, "w") as f:
                    json.dump(internal_sls, f, indent=2)
                logger.info(f"Removed {removed} orphaned internal SL(s) from internal_sls.json")

        # --- CLEANUP: Remove phantom open positions from self._open_positions ---
        mt5_magics = {pos.magic for pos in positions}
        phantom_magics = [magic for magic in list(self._open_positions.keys()) if magic not in mt5_magics]
        for magic in phantom_magics:
            logger.info(f"Removing phantom open position for magic {magic} (not present in MT5)")
            del self._open_positions[magic]

        restored = 0
        for pos in positions:
            # Only restore positions for your symbol
            if pos.symbol != self.symbol:
                continue
            direction = TradeDirection.LONG if pos.type == mt5.POSITION_TYPE_BUY else TradeDirection.SHORT

            # Try to restore internal SL
            internal_sl = load_internal_sl(pos.magic)
            if internal_sl is not None and internal_sl > 0:
                stop_loss = internal_sl
                logger.info(f"Restored internal SL for magic {pos.magic}: {stop_loss}")
            else:
                stop_loss = pos.sl
                if stop_loss == 0:
                    logger.warning(f"SL for magic {pos.magic} is 0! No internal SL found. Consider closing or managing this trade manually.")

            # Recreate TradePosition object
            position = TradePosition(
                magic_number=pos.magic,
                direction=direction,
                entry_price=pos.price_open,
                stop_loss=stop_loss,
                take_profit=pos.tp,
                timestamp=pd.to_datetime(pos.time),  # or datetime.fromtimestamp(pos.time)
                symbol=pos.symbol,
                position_size=pos.volume,
                trailing_stop=False,  # Could be inferred if you store this info in comment/magic
                atr_trailing_multiplier=0.0
            )
            self._open_positions[pos.magic] = position
            logger.info(f"Restored open position from MT5: Magic {pos.magic}, {pos.symbol}, {direction}, {pos.volume} lots")
            restored += 1
        logger.info(f"Total open positions restored from MT5: {restored}")

    def open_position(self, direction: TradeDirection, entry_price: float, stop_loss: float,
                     take_profit: float, position_size: float, timestamp: datetime,
                     symbol: str, trailing_stop: bool = False, atr_trailing_multiplier: float = 0.0,
                     hidden_stop_loss: bool = True) -> TradePosition:
        """Create and track a new position with unique magic number"""
        magic = self.generate_magic_number()

        if len(self._open_positions) >= self._max_open_trades:
            logger.info(
                f"Trade blocked: {len(self._open_positions)} open trades (max allowed: {self._max_open_trades})"
            )
            return None

        # --- SEND ORDER TO MT5 IF LIVE TRADING ---
        if self._trading_mode == 'live':
            order_type = mt5.ORDER_TYPE_BUY if direction == TradeDirection.LONG else mt5.ORDER_TYPE_SELL
            sl_value = 0 if hidden_stop_loss else stop_loss
            
            # Get current market prices for the symbol
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get current tick for {symbol}")
                return None
            
            # Use ASK price for BUY orders, BID price for SELL orders
            if direction == TradeDirection.LONG:
                current_price = tick.ask  # Use ASK price for BUY orders
            else:
                current_price = tick.bid   # Use BID price for SELL orders
            
            # --- FIX: Round position_size to nearest 0.01 and clamp to min/max ---
            volume_step = 0.01
            min_lot = self.min_lot_size
            max_lot = self.max_lot_size
            position_size = max(min_lot, min(max_lot, round(position_size / volume_step) * volume_step))
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_size,
                "type": order_type,
                "price": current_price,  # Use current market price, not entry_price
                "tp": take_profit,
                "deviation": self.deviation,
                "magic": magic,
                "comment": getattr(self, 'comment','bot_gkk'),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Only add stop loss if it's not hidden (not 0)
            if sl_value > 0:
                request["sl"] = sl_value

            # Check MT5 connection
            if not mt5.initialize():
                logger.error("MT5 initialize failed")
                raise Exception("MT5 initialize failed")
            
            # Check MT5 connection status
            if not mt5.terminal_info():
                logger.error("MT5 terminal not connected")
                raise Exception("MT5 terminal not connected")

            # Check if symbol is available
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found in MT5")
                raise Exception(f"Symbol {symbol} not found")
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select symbol {symbol}")
                    raise Exception(f"Failed to select symbol {symbol}")

            # Check if trading is allowed
            if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                logger.error(f"Trading not allowed for {symbol}")
                raise Exception(f"Trading not allowed for {symbol}")

            # Check account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                raise Exception("Failed to get account info")
            logger.info(f"Account balance: {account_info.balance}, equity: {account_info.equity}")

            # Check symbol info and market status
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                raise Exception(f"Symbol {symbol} not found")
            
            logger.info(f"🧪 SYMBOL INFO:")
            logger.info(f"   Symbol: {symbol_info.name}")
            logger.info(f"   Trading Mode: {symbol_info.trade_mode}")
            logger.info(f"   Trade Stops Level: {symbol_info.trade_stops_level}")
            logger.info(f"   Trade Freeze Level: {symbol_info.trade_freeze_level}")
            logger.info(f"   Digits: {symbol_info.digits}")
            logger.info(f"   Spread: {symbol_info.spread}")
            logger.info(f"   Trade Contract Size: {symbol_info.trade_contract_size}")
            logger.info(f"   Volume Min: {symbol_info.volume_min}")
            logger.info(f"   Volume Max: {symbol_info.volume_max}")
            logger.info(f"   Volume Step: {symbol_info.volume_step}")
            
            # Check if symbol is available for trading
            if not symbol_info.visible:
                logger.warning(f"Symbol {symbol} is not visible, trying to add it...")
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to add symbol {symbol} to Market Watch")
                    raise Exception(f"Symbol {symbol} not available for trading")
            
            # Check current market tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get current tick for {symbol}")
                raise Exception(f"Cannot get current price for {symbol}")
            
            logger.info(f"🧪 CURRENT MARKET TICK:")
            logger.info(f"   Bid: {tick.bid}")
            logger.info(f"   Ask: {tick.ask}")
            logger.info(f"   Last: {tick.last}")
            logger.info(f"   Time: {tick.time}")
            
            # Check if market is open by looking at the tick time
            from datetime import datetime
            tick_time = datetime.fromtimestamp(tick.time)
            current_time = datetime.now()
            time_diff = abs((current_time - tick_time).total_seconds())
            
            logger.info(f"🧪 MARKET STATUS:")
            logger.info(f"   Tick Time: {tick_time}")
            logger.info(f"   Current Time: {current_time}")
            logger.info(f"   Time Difference: {time_diff:.1f} seconds")
            
            # If tick is more than 60 seconds old, market might be closed
            if time_diff > 60:
                logger.warning(f"⚠️ Market tick is {time_diff:.1f} seconds old - market might be closed")
                logger.warning(f"⚠️ XAUUSD trading hours: Sunday 22:00 - Friday 21:00 GMT")
                logger.warning(f"⚠️ Current time appears to be outside trading hours")
            
            # Check if we can actually place orders
            if tick.bid == 0 or tick.ask == 0:
                logger.error(f"❌ Invalid bid/ask prices - market closed or symbol not available")
                raise Exception(f"Market closed or symbol not available - Bid: {tick.bid}, Ask: {tick.ask}")

            # Log order details for debugging
            logger.info(f"🧪 SENDING MT5 ORDER:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Type: {order_type} ({'BUY' if direction == TradeDirection.LONG else 'SELL'})")
            logger.info(f"   Volume: {position_size}")
            logger.info(f"   Price: {current_price} (Market Price)")
            logger.info(f"   Stop Loss: {sl_value}")
            logger.info(f"   Take Profit: {take_profit}")
            logger.info(f"   Magic: {magic}")

            result = mt5.order_send(request)
            logger.info(f"MT5 order_send result: {result}")
            if result is None:
                logger.error("MT5 order_send failed - returned None. Check MT5 connection, symbol, or market status.")
                
                # Try to get more detailed error information
                logger.error("🧪 ATTEMPTING TO GET ERROR DETAILS:")
                try:
                    # Check if we can get any error information
                    last_error = mt5.last_error()
                    if last_error:
                        logger.error(f"   MT5 Last Error: {last_error}")
                    
                    # Try to get account info again to see if connection is still alive
                    account_check = mt5.account_info()
                    if account_check:
                        logger.info(f"   Account connection still alive: {account_check.login}")
                    else:
                        logger.error("   Account connection lost")
                        
                except Exception as e:
                    logger.error(f"   Error getting details: {e}")
                
                # Optionally: raise Exception("MT5 order_send returned None")
                return None  # or handle as you wish
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order send failed: {result.retcode} - {result.comment}")
                # Optionally: return None or raise an exception here

        # --- TRACK POSITION INTERNALLY ---
        position = TradePosition(
            magic_number=magic,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=timestamp,
            symbol=symbol,
            position_size=position_size,
            trailing_stop=trailing_stop,
            atr_trailing_multiplier=atr_trailing_multiplier,
            hidden_stop_loss=hidden_stop_loss
        )
    
        # Track the position
        self._open_positions[magic] = position

        # Log the trade entry to database
        if self.trades_log_in_db:
            trade_data = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'direction': direction.value,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'magic_number': magic
            }
            trade_logger.log_trade_entry(trade_data)

        # Update last trade time for time-based cooldown
        self._last_trade_time = timestamp

        # Update last signal time for bar-based cooldown
        self._last_signal_time[direction.name] = timestamp

        logger.info(f"{direction.name} Position Opened:")
        logger.info(f"   Magic Number: {magic}")
        logger.info(f"   Entry Price: {entry_price}")
        logger.info(f"   Hidden Stop Loss: {stop_loss} (NOT placed in system - monitored dynamically)")
        logger.info(f"   Take Profit: {take_profit}")
        logger.info(f"   Size: {position_size:.2f} lots")
        save_internal_sl(magic, stop_loss)
        return position

    def close_position(self, magic_number: int, exit_price: float = None, exit_reason: str = None, exit_time: datetime = None):
        """
        Close a position and remove it from tracking.
        Also sends a close order to MT5 if in live mode.
        """
        import MetaTrader5 as mt5
        position = self._open_positions.get(magic_number)
        if not position:
            logger.warning(f"Attempted to close non-existent position: Magic {magic_number}")
            return None

        # --- SEND CLOSE ORDER TO MT5 IF LIVE TRADING ---
        if getattr(self, '_trading_mode', 'testing').lower() == 'live':
            # Find the actual MT5 position ticket
            positions = mt5.positions_get(symbol=position.symbol)
            ticket = None
            for pos in positions:
                if pos.magic == magic_number and pos.type == (mt5.POSITION_TYPE_BUY if position.direction == TradeDirection.LONG else mt5.POSITION_TYPE_SELL):
                    ticket = pos.ticket
                    break

            if ticket is not None:
                close_type = mt5.ORDER_TYPE_SELL if position.direction == TradeDirection.LONG else mt5.ORDER_TYPE_BUY
                close_price = mt5.symbol_info_tick(position.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.position_size,
                    "type": close_type,
                    "position": ticket,
                    "price": close_price,
                    "deviation": getattr(self, 'deviation', 20),
                    "magic": magic_number,
                    "comment": f"Exit: {exit_reason or 'Manual Close'}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(close_request)
                logger.info(f"MT5 close order_send result: {result}")
                if result is None:
                    logger.error("MT5 close order_send failed - returned None. Check MT5 connection, symbol, or market status.")
                elif result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Close order failed: {result.retcode} - {getattr(result, 'comment', '')}")
                else:
                    logger.info(f"Position closed successfully: ticket={ticket}")
            else:
                logger.error(f"No matching MT5 position found to close for magic {magic_number}")

        # --- INTERNAL TRACKING (existing logic) ---
        position.exit_price = exit_price or position.entry_price
        if self._trading_mode == 'testing' and exit_time is not None:
            position.exit_time = exit_time
        else:
            position.exit_time = datetime.now()
        position.exit_reason = exit_reason or "Manual Close"
        position.is_closed = True
        self._completed_trades.append(position)
        del self._open_positions[magic_number]
        
        # Reset the signal flag when a position is closed
        if position.direction == TradeDirection.LONG and self._pending_long_signal:
            self._pending_long_signal = False
            logger.info(f"Position Closed: Resetting LONG signal flag to False")
        elif position.direction == TradeDirection.SHORT and self._pending_short_signal:
            self._pending_short_signal = False
            logger.info(f"Position Closed: Resetting SHORT signal flag to False")

        logger.info(f"Position Closed: Magic {magic_number}")
        logger.info(f"   Exit Price: {position.exit_price}")
        logger.info(f"   Exit Reason: {position.exit_reason}")
        logger.info(f"   Exit Time: {position.exit_time}")

        # Log the trade exit to database
        if self.trades_log_in_db:
            # Calculate profit/loss and commission using the same method as HTML report
            point = self.point_size  # Get point size from config
            pip_value = getattr(self, '_pip_value_per_001_lot', 1.0)  # Use the same pip value as HTML report
            
            if position.direction == TradeDirection.LONG:
                profit_loss = (position.exit_price - position.entry_price) / point * pip_value * position.position_size
            else:  # SHORT
                profit_loss = (position.entry_price - position.exit_price) / point * pip_value * position.position_size
            
            commission = position.position_size * self.commission_per_lot
            
            exit_data = {
                'exit_price': position.exit_price,
                'exit_time': position.exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_reason': position.exit_reason,
                'profit_loss': profit_loss,
                'commission': commission
            }
            trade_logger.log_trade_exit(magic_number, exit_data)
            
            # --- UPDATE ACCOUNT BALANCE IN BACKTESTING MODE ---
            if getattr(self, '_trading_mode', 'testing').lower() == 'testing':
                net_profit_loss = profit_loss - commission
                # Update account balance
                self.account_balance += net_profit_loss
                # Log the balance update
                logger.info(f"    💰 Account balance updated: ${self.account_balance:.2f} (P/L: ${net_profit_loss:.2f})")
                if commission > 0:
                    logger.info(f"    💸 Commission: ${commission:.2f}")

        return position

    def get_open_positions_count(self) -> int:
        """Get the current number of open positions."""
        return len(self._open_positions)

    def get_open_positions(self) -> dict:
        """Get all open positions"""
        return self._open_positions.copy()

    def get_position_by_magic(self, magic_number: int) -> TradePosition:
        """Get a specific position by magic number"""
        return self._open_positions.get(magic_number)
          
    def update_open_positions(self, df: pd.DataFrame, open_positions: List[TradeSignal] = None) -> List[TradeSignal]:
        """
        Update open positions with current market data and check for exit conditions.
        Now uses magic number-based position tracking.
        Returns empty list as positions are tracked internally by magic number.

        Handles trading_mode differently:
        - TESTING mode: Uses candle High/Low for SL/TP detection
        - LIVE mode: Uses current Close price for SL/TP detection
        """
        if df.empty:
            return []

        # Use internal position tracking instead of passed positions
        if not self._open_positions:
            return []

        current_bar = df.iloc[-1]
        current_price = current_bar['Close']
        current_time = df.index[-1]
        atr = current_bar.get('atr', 0.0)
        supertrend = current_bar.get('super_trend', {})

        current_close = current_bar['Close']
        current_low = current_bar['Low']
        current_high = current_bar['High']
        

        # Process all open positions tracked by magic number
        # Create a copy to avoid "dictionary changed size during iteration" error
        open_positions_copy = self._open_positions.copy()

        for magic_number, position in open_positions_copy.items():
            if position.trailing_stop and atr:
                position.update_trailing_stop(current_price, atr)

            # Check for exit conditions using unified should_close method
            exit_reason = None
            exit_price = None
            
            # Create a current bar for should_close method (unified approach)
            current_bar = pd.Series({
                'Open': current_price,
                'High': current_price,
                'Low': current_price,
                'Close': current_price
            })
            
            # Add timestamp as name to prevent same-candle exit in live mode
            # This ensures the same-candle check in should_close works properly
            if self._trading_mode == 'live':
                current_bar.name = position.timestamp
            
            # Use the same should_close method for both live and testing modes
            if position.should_close(current_bar):
                exit_reason = position.exit_reason
                exit_price = position.exit_price
                
                # Adjust exit price based on trading mode
                if self._trading_mode == 'testing':
                    # TESTING MODE: Use exact SL/TP levels for accurate backtesting
                    if exit_reason == self.EXIT_SL_HIT:
                        exit_price = position.stop_loss
                    elif exit_reason == self.EXIT_TP_HIT:
                        exit_price = position.take_profit
                else:
                    # LIVE MODE: Use market price (may have slippage)
                    exit_price = current_price
                
                logger.info(f"🎯 EXIT TRIGGERED for Magic {magic_number}: {self.get_exit_reason_description(exit_reason)}")
                logger.info(f"   Position: {position.direction.name}")
                logger.info(f"   Entry: {position.entry_price:.2f}")
                logger.info(f"   Exit Price: {exit_price:.2f} ({'SL/TP Level' if self._trading_mode == 'testing' else 'Market Price'})")
            
            # --- NEW: SuperTrend price cross exit (prev candle close crosses ST long line) ---
            if not exit_reason:
                prev_bar = df.iloc[-2] if len(df) > 1 else None
                if prev_bar is not None:
                    # Enhanced debugging for SuperTrend exit conditions
                    logger.debug(f"🔍 SuperTrend Exit Check for Magic {magic_number} ({position.direction.name}):")
                    logger.debug(f"   Current Close: {current_close:.2f}")
                    logger.debug(f"   Prev Close: {prev_bar['Close']:.2f}")
                    
                    if position.direction == TradeDirection.SHORT:
                        st_long_val = prev_bar.get('st_long', 'N/A')
                        logger.debug(f"   Prev ST_Long: {st_long_val}")
                        if 'st_long' in prev_bar and prev_bar['Close'] > prev_bar['st_long']:
                            exit_reason = self.EXIT_TSL_SUPERTREND
                            exit_price = current_close
                            logger.info(f"🔴 SuperTrend price cross exit for SHORT position Magic: {magic_number}")
                            logger.info(f"   Prev Close ({prev_bar['Close']:.2f}) > ST_Long ({prev_bar['st_long']:.2f})")
                            logger.info(f"   Exit Price: {exit_price:.2f}")
                        else:
                            logger.debug(f"   ❌ SHORT exit condition NOT met: Prev Close ({prev_bar['Close']:.2f}) <= ST_Long ({st_long_val})")
                    
                    elif position.direction == TradeDirection.LONG:
                        st_long_val = prev_bar.get('st_long', 'N/A')
                        logger.debug(f"   Prev ST_Long: {st_long_val}")
                        if 'st_long' in prev_bar and prev_bar['Close'] < prev_bar['st_long']:
                            exit_reason = self.EXIT_TSL_SUPERTREND
                            exit_price = current_close
                            logger.info(f"🔴 SuperTrend price cross exit for LONG position Magic: {magic_number}")
                            logger.info(f"   Prev Close ({prev_bar['Close']:.2f}) < ST_Long ({prev_bar['st_long']:.2f})")
                            logger.info(f"   Exit Price: {exit_price:.2f}")
                        else:
                            logger.debug(f"   ❌ LONG exit condition NOT met: Prev Close ({prev_bar['Close']:.2f}) >= ST_Long ({st_long_val})")
                else:
                    logger.debug(f"🔍 SuperTrend Exit Check for Magic {magic_number}: No previous bar data available")
                    # exit_conditions.append({
                    #     'type': 'SuperTrend Exit',
                    #     'condition': "No previous bar data available",
                    #     'status': 'UNKNOWN',
                    #     'exit_price': None,
                    #     'reason': None
                    # })

            # Enhanced debugging for all exit conditions
            if exit_reason:
                logger.info(f"🎯 EXIT TRIGGERED for Magic {magic_number}: {self.get_exit_reason_description(exit_reason)}")
                logger.info(f"   Position: {position.direction.name}")
                logger.info(f"   Entry: {position.entry_price:.2f}")
                logger.info(f"   Current Price: {current_close:.2f}")
                logger.info(f"   Exit Price: {exit_price:.2f}")
                if position.direction == TradeDirection.LONG:
                    logger.info(f"   Distance to SL: {current_low - position.stop_loss:.2f} points")
                    logger.info(f"   Distance to TP: {position.take_profit - current_high:.2f} points")
                else:  # SHORT
                    logger.info(f"   Distance to SL: {position.stop_loss - current_high:.2f} points")
                    logger.info(f"   Distance to TP: {current_low - position.take_profit:.2f} points")
            else:
                logger.debug(f"🔍 No exit conditions met for Magic {magic_number} ({position.direction.name})")
                logger.debug(f"   Current Low: {current_low:.2f}, SL: {position.stop_loss:.2f}")
                logger.debug(f"   Current High: {current_high:.2f}, TP: {position.take_profit:.2f}")
                if position.direction == TradeDirection.LONG:
                    logger.debug(f"   SL Check: {current_low:.2f} <= {position.stop_loss:.2f} = {current_low <= position.stop_loss}")
                    logger.debug(f"   TP Check: {current_high:.2f} >= {position.take_profit:.2f} = {current_high >= position.take_profit}")
                else:  # SHORT
                    logger.debug(f"   SL Check: {current_high:.2f} >= {position.stop_loss:.2f} = {current_high >= position.stop_loss}")
                    logger.debug(f"   TP Check: {current_low:.2f} <= {position.take_profit:.2f} = {current_low <= position.take_profit}")

            # If exit condition met, close the position
            if exit_reason:
                # Get human-readable exit reason
                exit_reason_desc = self.get_exit_reason_description(exit_reason)

                # Close the position using magic number with proper exit time
                if self._trading_mode == 'testing':
                    # TESTING MODE: Use candle timestamp for accurate backtesting
                    closed_position = self.close_position(magic_number, exit_price, exit_reason_desc, current_time)
                else:
                    # LIVE MODE: Use current time (default behavior)
                    closed_position = self.close_position(magic_number, exit_price, exit_reason_desc)

                # Track last trade exit for strategy
                self.last_trade_exit = {
                    'price': exit_price,
                    'time': current_time,
                    'reason': exit_reason,
                    'direction': position.direction,
                    'description': exit_reason_desc,
                    'magic_number': magic_number
                }
                logger.info(f"🔍 LAST TRADE EXIT UPDATED: {exit_reason} at {current_time}")
                logger.info(f"  Price: {exit_price}, Direction: {position.direction}")
                logger.info(f"  Last trade exit reason: {self.last_trade_exit.get('reason')}")
        
        return []  # Return empty list as positions are tracked internally

    def get_exit_reason_description(self, exit_reason):
        """
        Get a human-readable description for an exit reason code.
        
        Args:
            exit_reason: The exit reason code
            
        Returns:
            str: Human-readable description of the exit reason
        """
        exit_descriptions = {
            self.EXIT_TP_HIT: 'Take Profit Hit',
            self.EXIT_SL_HIT: 'Stop Loss Hit',
            self.EXIT_TRAILING_STOP: 'Trailing Stop',
            self.EXIT_TSL_ATR: 'ATR Trailing Stop',
            self.EXIT_TSL_STATIC: 'Static Trailing Stop',
            self.EXIT_TSL_SUPERTREND: 'SuperTrend Exit',
            self.EXIT_TRAILING_ACTIVATED: 'Trailing Stop Activated',
            self.EXIT_SIGNAL_REVERSAL: 'Signal Reversal',
            self.EXIT_LOW_CONFIDENCE: 'Low Confidence',
            self.EXIT_MANUAL: 'Manual Close',
            #self.EXIT_VOLATILITY: 'High Volatility',
            # Removed: 'Volatility Exit' - ATR-based trading handles volatility
            self.EXIT_SESSION_END: 'Session End',
            self.EXIT_MAX_DRAWDOWN: 'Max Drawdown',
            self.EXIT_MAX_LOSS: 'Max Loss Reached',
            self.EXIT_TIME_EXIT: 'Time Exit',
            self.EXIT_TIME_EXPIRY: 'Time Expiry',
            self.EXIT_OTHER: 'Other'
        }
        return exit_descriptions.get(exit_reason, exit_reason or 'N/A')

    def _get_exit_type(self, exit_reason):
        """
        Categorize exit reasons into broader types for analysis.
        
        Args:
            exit_reason: The specific exit reason code
            
        Returns:
            str: The exit type category
        """
        if exit_reason == self.EXIT_TP_HIT:
            return 'TAKE_PROFIT'
        elif exit_reason == self.EXIT_SL_HIT:
            return 'STOP_LOSS'
        elif exit_reason in [self.EXIT_TRAILING_STOP, self.EXIT_TSL_STATIC, 
                           self.EXIT_TSL_ATR, self.EXIT_TSL_SUPERTREND,
                           self.EXIT_TRAILING_ACTIVATED]:
            return 'TRAILING_STOP'
        elif exit_reason in [self.EXIT_SIGNAL_REVERSAL, self.EXIT_LOW_CONFIDENCE]:
            return 'SIGNAL_BASED'
        elif exit_reason in [self.EXIT_TIME_EXIT, self.EXIT_TIME_EXPIRY]:
            return 'TIME_BASED'
        # Removed: EXIT_VOLATILITY and EXIT_VOLATILITY_EXIT - ATR-based trading handles volatility
            return 'VOLATILITY'
        elif exit_reason == self.EXIT_SESSION_END:
            return 'SESSION_END'
        elif exit_reason == self.EXIT_MANUAL:
            return 'MANUAL'
        return 'OTHER'
        
    def _check_drawdown_exit(self, current_equity: float) -> bool:
        """Check if max drawdown threshold is reached."""
        if not hasattr(self, '_peak_equity') or current_equity > self._peak_equity:
            self._peak_equity = current_equity
            return False
            
        max_drawdown_pct = getattr(self, 'max_drawdown_pct')
        drawdown_pct = ((self._peak_equity - current_equity) / self._peak_equity) * 100
        
        if drawdown_pct >= max_drawdown_pct:
            logger.warning(f"Max drawdown reached: {drawdown_pct:.2f}% >= {max_drawdown_pct}%")
            return True
        return False
        
    def _reset_daily_drawdown(self, current_date, current_balance):
        """Reset daily drawdown tracking at the start of a new day."""
        self._daily_start_balance = current_balance
        self._daily_peak_balance = current_balance
        self._last_drawdown_check_date = current_date
        self._daily_drawdown_triggered = False
        logger.info(f"Daily drawdown reset for {current_date}: Start balance = ${current_balance:,.2f}")

    def _check_daily_drawdown(self, current_datetime, current_balance):
        """Check and update daily drawdown. Returns True if trading should be stopped for the day."""
        if not self.daily_drawdown_limit:
            return False  # Daily drawdown protection is disabled
            
        # Reset at the start of a new day
        current_date = current_datetime.date() if hasattr(current_datetime, 'date') else current_datetime
        if self._last_drawdown_check_date != current_date:
            self._reset_daily_drawdown(current_date, current_balance)

        # Update peak if new high
        if self._daily_peak_balance is None or current_balance > self._daily_peak_balance:
            self._daily_peak_balance = current_balance
            logger.debug(f"New daily peak balance: ${current_balance:,.2f}")

        # Check drawdown
        drawdown_pct = ((self._daily_peak_balance - current_balance) / self._daily_peak_balance) * 100
        if drawdown_pct >= self.daily_drawdown_limit_pct:
            if not self._daily_drawdown_triggered:
                logger.warning(f"🚨 MAX DAILY DRAWDOWN REACHED: {drawdown_pct:.2f}% >= {self.daily_drawdown_limit_pct}%")
                logger.warning(f"   Peak: ${self._daily_peak_balance:,.2f} | Current: ${current_balance:,.2f}")
                logger.warning(f"   Trading disabled for the rest of the day")
                self._daily_drawdown_triggered = True
            return True  # Stop trading for the day
        return False  # Trading allowed

    def can_trade_today(self, current_datetime, current_balance):
        """Public method to check if trading is allowed today (not stopped by daily drawdown)."""
        return not self._check_daily_drawdown(current_datetime, current_balance)
    
    def get_metrics_summary(self) -> str:
        """
        Get a formatted string summary of the trading metrics.

        Returns:
            str: Formatted metrics summary
        """
        return "No trading metrics available yet."
        
    def monitor_exit_conditions(self, df: pd.DataFrame = None):
        """
        Monitor and log all exit conditions for debugging purposes.
        This method helps identify why trades might be failing to exit.
        
        Args:
            df: DataFrame with current market data (optional)
        """
        if not self._open_positions:
            logger.info("🔍 No open positions to monitor")
            return
            
        logger.info("=" * 80)
        logger.info("🔍 EXIT CONDITIONS MONITORING")
        logger.info("=" * 80)
        
        for magic_number, position in self._open_positions.items():
            logger.info(f"📊 Position {magic_number} ({position.direction.name}):")
            logger.info(f"   Entry: {position.entry_price:.2f}")
            logger.info(f"   Stop Loss: {position.stop_loss:.2f}")
            logger.info(f"   Take Profit: {position.take_profit:.2f}")
            logger.info(f"   Trailing Stop: {'✅ Enabled' if position.trailing_stop else '❌ Disabled'}")
            
            if df is not None and not df.empty:
                current_bar = df.iloc[-1]
                prev_bar = df.iloc[-2] if len(df) > 1 else None
                
                current_price = current_bar['Close']
                current_high = current_bar.get('High', current_price)
                current_low = current_bar.get('Low', current_price)
                
                logger.info(f"   Current Price: {current_price:.2f}")
                logger.info(f"   Current High: {current_high:.2f}")
                logger.info(f"   Current Low: {current_low:.2f}")
                
                # Check SL/TP conditions
                if position.direction == TradeDirection.LONG:
                    sl_distance = current_low - position.stop_loss
                    tp_distance = position.take_profit - current_high
                    logger.info(f"   Distance to SL: {sl_distance:.2f} points")
                    logger.info(f"   Distance to TP: {tp_distance:.2f} points")
                    logger.info(f"   SL Hit: {'✅ YES' if current_low <= position.stop_loss else '❌ NO'}")
                    logger.info(f"   TP Hit: {'✅ YES' if current_high >= position.take_profit else '❌ NO'}")
                else:  # SHORT
                    sl_distance = position.stop_loss - current_high
                    tp_distance = current_low - position.take_profit
                    logger.info(f"   Distance to SL: {sl_distance:.2f} points")
                    logger.info(f"   Distance to TP: {tp_distance:.2f} points")
                    logger.info(f"   SL Hit: {'✅ YES' if current_high >= position.stop_loss else '❌ NO'}")
                    logger.info(f"   TP Hit: {'✅ YES' if current_low <= position.take_profit else '❌ NO'}")
                
                # Check SuperTrend conditions
                if prev_bar is not None:
                    if position.direction == TradeDirection.LONG:
                        st_long_val = prev_bar.get('st_long', 'N/A')
                        st_cross = prev_bar['Close'] < st_long_val if st_long_val != 'N/A' else False
                        logger.info(f"   ST_Long: {st_long_val}")
                        logger.info(f"   ST Cross: {'✅ YES' if st_cross else '❌ NO'}")
                    else:  # SHORT
                        st_long_val = prev_bar.get('st_long', 'N/A')
                        st_cross = prev_bar['Close'] > st_long_val if st_long_val != 'N/A' else False
                        logger.info(f"   ST_Long: {st_long_val}")
                        logger.info(f"   ST Cross: {'✅ YES' if st_cross else '❌ NO'}")
                else:
                    logger.info(f"   ST Cross: ⚠️ No previous bar data")
            
            logger.info("-" * 80)
        
        logger.info("=" * 80)
        
    def resample_dataframe(self, df: pd.DataFrame, from_tf: str, to_tf: str) -> pd.DataFrame:
        """
        Resample a DataFrame from one timeframe to another.
        
        Args:
            df: Input DataFrame with datetime index
            from_tf: Source timeframe (e.g., 'M1' for 1 minute)
            to_tf: Target timeframe (e.g., 'M3' for 3 minutes)
            
        Returns:
            Resampled DataFrame
        """
        # Convert timeframes to pandas frequency strings using 'min' instead of 'T'
        tf_map = {
            'M1': '1min', 'M3': '3min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
            'H1': '1H', 'H4': '4H', 'D1': '1D', 'W1': '1W', 'MN1': '1M'
        }
        
        from_freq = tf_map.get(from_tf.upper())
        to_freq = tf_map.get(to_tf.upper())
        
        if not from_freq or not to_freq:
            raise ValueError(f"Unsupported timeframe. Supported: {list(tf_map.keys())}")
            
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            
        # Prepare aggregation dictionary with available columns
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Only include tick_volume if it exists in the dataframe
        if 'tick_volume' in df.columns:
            agg_dict['tick_volume'] = 'sum'
        
        # Resample the data
        resampled = df.resample(to_freq).agg(agg_dict).dropna()
        
        return resampled
        
    def calculate_adxr(self, df, period=9):
        """
        Calculate ADXR (Average Directional Movement Rating)
        ADXR = (Current ADX + ADX from 'period' bars ago) / 2
        
        Args:
            df: DataFrame with ADX column (lowercase 'adx')
            period: Period for ADXR calculation (default: 14)
            
        Returns:
            Series with ADXR values
        """
        if 'adx' not in df.columns:
            raise ValueError("ADX must be calculated before ADXR. Make sure to calculate ADX first using ta.adx()")
            
        # Shift ADX by period to get ADX from 'period' bars ago
        shifted_adx = df['adx'].shift(period)
        
        # Calculate ADXR as the average of current ADX and shifted ADX
        #adxr = (df['adx'] + shifted_adx) / 2
        
        # Calculate SMA of ADX over 'period' bars
        adxr = df['adx'].rolling(window=period).mean()

        return adxr
        
    def calculate_atr_based_levels(self, current_bar, direction):
        """
        Calculate ATR-based stop loss and take profit levels.
        
        Args:
            current_bar: Current bar data containing ATR
            direction: TradeDirection.LONG or TradeDirection.SHORT
            
        Returns:
            tuple: (stop_loss, take_profit, position_size)
        """
        if not self.atr_based_trading:
            # Fallback to fixed levels if ATR-based trading is disabled
            entry_price = current_bar['Close']
            if direction == TradeDirection.LONG:
                stop_loss = entry_price - 50  # Fixed 50 points
                take_profit = entry_price + 100  # Fixed 100 points
            else:
                stop_loss = entry_price + 50  # Fixed 50 points
                take_profit = entry_price - 100  # Fixed 100 points
            return stop_loss, take_profit, self.min_lot_size
            
        # Get current ATR value
        atr_value = current_bar.get('atr')
        if atr_value is None or atr_value <= 0:
            logger.warning("ATR value is missing or zero! Cannot calculate stop loss.")
            return None, None, None

        # Apply volatility filter
        if self.atr_volatility_filter:
            if atr_value < self.atr_min_value or atr_value > self.atr_max_value:
                logger.warning(f"❌ ATR volatility filter: ATR={atr_value:.2f} outside range [{self.atr_min_value}, {self.atr_max_value}]")
                return None, None, None
        
        # Calculate ATR-based levels
        entry_price = current_bar['Close']
        stop_multiplier = max(self.atr_min_multiplier, min(self.atr_max_multiplier, self.atr_stop_multiplier))
        tp_multiplier = max(self.atr_min_multiplier, min(self.atr_max_multiplier, self.atr_tp_multiplier))
        if stop_multiplier <= 0 or tp_multiplier <= 0:
            logger.warning("ATR multipliers must be positive! Check your config.")
            return None, None, None
        
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - (atr_value * stop_multiplier)
            take_profit = entry_price + (atr_value * tp_multiplier)
        else:  # SHORT
            stop_loss = entry_price + (atr_value * stop_multiplier)
            take_profit = entry_price - (atr_value * tp_multiplier)

        # Determine risk percentage based on trading style
        trading_style = self._determine_trading_style(current_bar)

        # Determine risk percentage based on trading style
        original_risk_pct = self.risk_per_trade_pct
        #if hasattr(self, '_current_trading_style') and self._current_trading_style == 'swing':
        if trading_style == 'swing':
            # For swing style, use the configured swing risk percentage
            risk_pct = self.swing_risk_per_trade_pct
            logger.info(f"Using swing style risk: {risk_pct}% (increased from {original_risk_pct}%) [from config]")
        #else
        if trading_style == 'scalp':            
            # For scalp style, use the default risk per trade
            risk_pct = original_risk_pct
            logger.info(f"Using scalp style risk: {risk_pct}%")

        # Store original risk setting
        original_risk = self.risk_per_trade_pct
        
        # Temporarily set risk for position size calculation
        self.risk_per_trade_pct = risk_pct
        
        # Calculate ATR-based position size with adjusted risk
        position_size = self.calculate_atr_position_size(atr_value, stop_multiplier)
        
        # Restore original risk setting
        self.risk_per_trade_pct = original_risk
        
        # Get trading style for logging
        style = getattr(self, '_current_trading_style')
        
        logger.info(f"ATR-Based Levels ({direction.name}) - Style: {style}:")
        logger.info(f"  ATR: {atr_value:.2f} points")
        logger.info(f"  Entry: {entry_price:.2f}")
        logger.info(f"  Stop Loss: {stop_loss:.2f} ({stop_multiplier}x ATR)")
        logger.info(f"  Take Profit: {take_profit:.2f} ({tp_multiplier}x ATR)")
        logger.info(f"  Position Size: {position_size:.3f} lots")
        logger.info(f"  Risk: {risk_pct}%")
        
        return stop_loss, take_profit, position_size
        
    def calculate_atr_position_size(self, atr_value, stop_multiplier):
        """
        Calculate position size based on ATR and risk management.
        
        Args:
            atr_value: Current ATR value
            stop_multiplier: ATR multiplier for stop loss
            
        Returns:
            float: Position size in lots
        """
        if not self.atr_position_sizing:
            return self.min_lot_size
        
        # For backtesting, we use the dynamically updated account balance
        # This ensures position sizing is accurate after each trade
        logger.info(f"Using account balance: ${self.account_balance:.2f}")
            
        # Calculate risk amount
        risk_amount = self.account_balance * (self.risk_per_trade_pct / 100.0)
        
        # Calculate stop distance in points
        stop_distance_points = atr_value * stop_multiplier
        
        # Calculate position size based on risk
        pip_value_per_lot = self._pip_value_per_001_lot * 100  # Convert to per 1.0 lot
        position_size = risk_amount / (stop_distance_points * pip_value_per_lot)
        
        # Apply min/max constraints
        position_size = max(self.min_lot_size, min(self.max_lot_size, position_size))
        
        # Calculate actual dollar risk for verification
        actual_risk = position_size * stop_distance_points * pip_value_per_lot
        
        logger.info(f"ATR Position Sizing:")
        logger.info(f"  Risk Amount: ${risk_amount:.2f}")
        logger.info(f"  Stop Distance: {stop_distance_points:.2f} points")
        logger.info(f"  Position Size: {position_size:.3f} lots")
        logger.info(f"  Actual Risk: ${actual_risk:.2f}")
        
        return position_size

    # def display_entry_conditions(self, df):
    #     """
    #     Display and log all entry condition and indicator details for the latest bar in both LONG and SHORT directions,
    #     in a single table with columns for both statuses. Group all News Time Filter and Bank Holiday Filter variants into a single row each.
    #     """
    #     from tabulate import tabulate
    #     if df is None or df.empty:
    #         print("[display_entry_conditions] No data available.")
    #         logger.info("[display_entry_conditions] No data available.")
    #         return
    #     current_bar = df.iloc[-1]
    #     bar_time = getattr(current_bar, 'name', None)
    #     print("\n=== ENTRY CONDITION ANALYSIS ===")
    #     logger.info("=== ENTRY CONDITION ANALYSIS ===")
    #     # Get details for both directions
    #     details_long = self._get_entry_condition_details(current_bar, 'LONG')
    #     details_short = self._get_entry_condition_details(current_bar, 'SHORT')
    #     # Group conditions by category, but group all News/Bank Holiday filters into one row each
    #     categories = [
    #         ("Price/SMA", ["Price vs SMA Fast", "SMA Fast vs Slow"]),
    #         ("SuperTrend", ["SuperTrend Direction", "Price vs SuperTrend"]),
    #         ("Momentum", ["RSI Level", "ADX Strength", "DI Momentum"]),
    #         ("Volatility", ["ATR Volatility"]),
    #         ("Session/Time", ["Trading Hours", "Market Session", "News Time Filter", "Bank Holiday Filter"])
    #     ]
    #     # Helper to get the most relevant value for grouped filters
    #     def get_grouped_value(details, keys):
    #         for key in keys:
    #             val = details.get(key)
    #             if val and val not in ("-", "Disabled", "N/A"):
    #                 return val
    #         # If all are disabled or missing, return the first available or '-'
    #         for key in keys:
    #             val = details.get(key)
    #             if val:
    #                 return val
    #         return "-"
    #     table = []
    #     for cat, keys in categories:
    #         for key in keys:
    #             if key == "News Time Filter":
    #                 # Group all News Time Filter variants
    #                 news_keys = [
    #                     "News Time Filter", "News Time Filter (FF)", "News Time Filter (Basic)"
    #                 ]
    #                 long_val = get_grouped_value(details_long, news_keys)
    #                 short_val = get_grouped_value(details_short, news_keys)
    #                 table.append([cat, key, long_val, short_val])
    #                 ascii_long = long_val.replace('✅', '[OK]').replace('❌', '[X]').replace('⚪', '[ ]').replace('⚠️', '[!]').replace('🚪', '[EXIT]').replace('📊', '[STATS]').replace('📈', '[UP]').replace('📉', '[DOWN]').replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    #                 ascii_short = short_val.replace('✅', '[OK]').replace('❌', '[X]').replace('⚪', '[ ]').replace('⚠️', '[!]').replace('🚪', '[EXIT]').replace('📊', '[STATS]').replace('📈', '[UP]').replace('📉', '[DOWN]').replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    #                 logger.info(f"LONG | {key}: {ascii_long}")
    #                 logger.info(f"SHORT | {key}: {ascii_short}")
    #             elif key == "Bank Holiday Filter":
    #                 # Group all Bank Holiday Filter variants
    #                 bank_keys = [
    #                     "Bank Holiday Filter", "Bank Holiday Filter (FF)", "Bank Holiday Filter (Basic)"
    #                 ]
    #                 long_val = get_grouped_value(details_long, bank_keys)
    #                 short_val = get_grouped_value(details_short, bank_keys)
    #                 table.append([cat, key, long_val, short_val])
    #                 ascii_long = long_val.replace('✅', '[OK]').replace('❌', '[X]').replace('⚪', '[ ]').replace('⚠️', '[!]').replace('🚪', '[EXIT]').replace('📊', '[STATS]').replace('📈', '[UP]').replace('📉', '[DOWN]').replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    #                 ascii_short = short_val.replace('✅', '[OK]').replace('❌', '[X]').replace('⚪', '[ ]').replace('⚠️', '[!]').replace('🚪', '[EXIT]').replace('📊', '[STATS]').replace('📈', '[UP]').replace('📉', '[DOWN]').replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    #                 logger.info(f"LONG | {key}: {ascii_long}")
    #                 logger.info(f"SHORT | {key}: {ascii_short}")
    #             else:
    #                 long_val = details_long.get(key, "-")
    #                 short_val = details_short.get(key, "-")
    #                 table.append([cat, key, long_val, short_val])
    #                 ascii_long = long_val.replace('✅', '[OK]').replace('❌', '[X]').replace('⚪', '[ ]').replace('⚠️', '[!]').replace('🚪', '[EXIT]').replace('📊', '[STATS]').replace('📈', '[UP]').replace('📉', '[DOWN]').replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    #                 ascii_short = short_val.replace('✅', '[OK]').replace('❌', '[X]').replace('⚪', '[ ]').replace('⚠️', '[!]').replace('🚪', '[EXIT]').replace('📊', '[STATS]').replace('📈', '[UP]').replace('📉', '[DOWN]').replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    #                 logger.info(f"LONG | {key}: {ascii_long}")
    #                 logger.info(f"SHORT | {key}: {ascii_short}")
    #     # Add any extra keys not in categories
    #     all_keys = set(details_long.keys()).union(details_short.keys())
    #     categorized_keys = set(k for _, keys in categories for k in keys)
    #     # Exclude grouped variants from extra keys
    #     grouped_exclude = {"News Time Filter (FF)", "News Time Filter (Basic)", "Bank Holiday Filter (FF)", "Bank Holiday Filter (Basic)"}
    #     for key in all_keys - categorized_keys - grouped_exclude:
    #         long_val = details_long.get(key, "-")
    #         short_val = details_short.get(key, "-")
    #         table.append(["Other", key, long_val, short_val])
    #         ascii_long = long_val.replace('✅', '[OK]').replace('❌', '[X]').replace('⚪', '[ ]').replace('⚠️', '[!]').replace('🚪', '[EXIT]').replace('📊', '[STATS]').replace('📈', '[UP]').replace('📉', '[DOWN]').replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    #         ascii_short = short_val.replace('✅', '[OK]').replace('❌', '[X]').replace('⚪', '[ ]').replace('⚠️', '[!]').replace('🚪', '[EXIT]').replace('📊', '[STATS]').replace('📈', '[UP]').replace('📉', '[DOWN]').replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    #         logger.info(f"LONG | {key}: {ascii_long}")
    #         logger.info(f"SHORT | {key}: {ascii_short}")
    #     print(tabulate(table, headers=["Category", "Condition", "Long Status", "Short Status"], tablefmt="fancy_grid"))
    #     print("=== END ENTRY CONDITION ANALYSIS ===\n")
    #     logger.info("=== END ENTRY CONDITION ANALYSIS ===")

    def generate_html_report(self, df: pd.DataFrame, trades: List[TradeSignal], output_file: str = 'strategy_report.html', from_date=None, to_date=None) -> str:
        """
        Generate an HTML report with strategy details, indicator values, and trades.

        Args:
            df: DataFrame with price and indicator data
            trades: List of TradeSignal objects (legacy) - now uses self._completed_trades
            output_file: Path to save the HTML report
        """
        # Use the full DataFrame for the report but filter display to trading period only
        display_df = df.copy()
        
        # Filter to show only the trading period (not lookback)
        if from_date and to_date:
            from_dt = pd.to_datetime(from_date) if isinstance(from_date, str) else from_date
            to_dt = pd.to_datetime(to_date) if isinstance(to_date, str) else to_date
            display_df = display_df[(display_df.index >= from_dt) & (display_df.index <= to_dt)]
        
        start_date = str(display_df.index.min().date()) if not display_df.empty else "N/A"
        end_date = str(display_df.index.max().date()) if not display_df.empty else "N/A"

        # Create subplots with 3 rows: price, volume, and indicators
        fig = make_subplots(
            rows=3, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03,
            row_heights=[0.6, 0.1, 0.3],
            subplot_titles=(f'Price with Indicators ({start_date} to {end_date})', 'Volume', 'RSI & ADX')
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=display_df.index,
                open=display_df['Open'],
                high=display_df['High'],
                low=display_df['Low'],
                close=display_df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Calculate price range with 10% padding using filtered DataFrame
        price_range = display_df['High'].max() - display_df['Low'].min()
        yaxis_range = [
            display_df['Low'].min() - price_range * 0.1,  # 10% padding below
            display_df['High'].max() + price_range * 0.1   # 10% padding above
        ]
        
        # Update layout with proper price range and date constraints
        fig.update_layout(
            title=f'{self.symbol} Trading Strategy Report',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(family='Arial', size=12, color='#2c3e50'),
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation='h', y=1.02, yanchor='bottom', x=0.5, xanchor='center'),
            yaxis=dict(
                title='Price',
                range=yaxis_range,
                fixedrange=False,
                showgrid=True,
                gridcolor='#e1e5eb',
                zeroline=False
            ),
            # Force the chart to respect our date range
            xaxis=dict(
                autorange=False,
                constrain='domain',
                showgrid=True,
                gridcolor='#e1e5eb'
            )
        )
        
        # Generate HTML for the plot
        plot_html = fig.to_html(full_html=False)

        # Generate HTML for strategy configuration
        config_html = f"""
        <h2>Strategy Configuration</h2>
        <table border="1" class="dataframe">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Symbol</td><td>{self.symbol}</td></tr>
            <tr><td>Timeframe</td><td>{self.data_timeframe}</td></tr>
            <tr><td>Start Date</td><td>{start_date}</td></tr>
            <tr><td>End Date</td><td>{end_date}</td></tr>            
            <tr><td>Account Balance</td><td>${self.account_balance:,.2f}</td></tr>
            <tr><td>Risk per Trade</td><td>{self.risk_per_trade_pct}%</td></tr>            
            <tr><td>Take Profit (ATR x TP Multiplier)</td><td>{self.atr_tp_multiplier} x ATR</td></tr>
            <tr><td>Commission per Lot</td><td>${self.commission_per_lot:,.2f}</td></tr>
            <tr><td>Pip Value (0.01 lot)</td><td>${self._pip_value_per_001_lot:,.2f}</td></tr>
            <tr><td>SMA Fast Period</td><td>{self.sma_fast_period}</td></tr>
            <tr><td>SMA Slow Period</td><td>{self.sma_slow_period}</td></tr>
            <tr><td>ADX Period</td><td>{self.adx_period}</td></tr>
            <tr><td>ADXR Period</td><td>{self.adxr_period}</td></tr>
            <tr><td>RSI Period</td><td>{self.rsi_period}</td></tr>
            <tr><td>RSI MA Period</td><td>{self.rsi_ma_period}</td></tr>
            <tr><td>Use Trailing Stop</td><td>{'Yes' if self.use_trailing_stop else 'No'}</td></tr>
            <tr><td>Trailing Stop ATR Multiplier</td><td>{self.atr_trailing_multiplier}</td></tr>
            <tr><td>Trading Hours</td><td>{self.start_hour:02d}:00 - {self.end_hour:02d}:00</td></tr>
        </table>
        """
        
        # Generate HTML for latest indicator values
        latest = display_df.iloc[-1]
        
        # Helper function to safely get indicator values
        def safe_get(df, col, default='N/A'):
            try:
                if col in df.columns:
                    val = df.iloc[-1][col]
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        return val
                return default
            except Exception:
                return default
        
        # Get indicator values safely
        close_val = safe_get(df, 'Close', safe_get(df, 'close'))
        volume_val = safe_get(df, 'Volume', safe_get(df, 'volume'))
        sma_fast_val = safe_get(df, 'sma_fast')
        sma_slow_val = safe_get(df, 'sma_slow')
        adx_val = safe_get(df, 'adx')
        adxr_val = safe_get(df, 'adxr')
        di_plus_val = safe_get(df, 'di_plus')
        di_minus_val = safe_get(df, 'di_minus')
        rsi_val = safe_get(df, 'rsi')
        rsi_ma_val = safe_get(df, 'rsi_ma')
        atr_val = safe_get(df, 'atr')
        st_long_val = safe_get(df, 'supertrend_long')
        st_short_val = safe_get(df, 'supertrend_short')
        cpr_pivot_val = safe_get(df, 'cpr_pivot')
        cpr_tc_val = safe_get(df, 'cpr_tc')
        cpr_bc_val = safe_get(df, 'cpr_bc')
        st_long_direction = safe_get(df, 'st_long_direction', 0)
        st_short_direction = safe_get(df, 'st_short_direction', 0)
        
        # Format values for display
        def format_value(val, format_str='{:.2f}'):
            if isinstance(val, (int, float)) and not pd.isna(val):
                return format_str.format(val)
            return str(val)
        
        # Generate indicator HTML
        indicators_html = """
        <h2>Latest Indicator Values</h2>
        <table border="1" class="dataframe">
            <tr><th>Indicator</th><th>Value</th></tr>
            <tr><td>Price</td><td>{}</td></tr>
            <tr><td>Volume</td><td>{}</td></tr>
            <tr><td>SMA Fast ({})</td><td>{}</td></tr>
            <tr><td>SMA Slow ({})</td><td>{}</td></tr>
            <tr><td>ADX ({})</td><td>{}</td></tr>
            <tr><td>+DI</td><td>{}</td></tr>
            <tr><td>-DI</td><td>{}</td></tr>
            <tr><td>RSI ({})</td><td>{}</td></tr>
            <tr><td>RSI MA ({})</td><td>{}</td></tr>
            <tr><td>ATR ({})</td><td>{}</td></tr>
            <tr><td>SuperTrend Long</td><td>{}</td></tr>
            <tr><td>SuperTrend Short</td><td>{}</td></tr>
            <tr><td>CPR Pivot</td><td>{}</td></tr>
            <tr><td>CPR TC</td><td>{}</td></tr>
            <tr><td>CPR BC</td><td>{}</td></tr>
        """.format(
            format_value(close_val),
            format_value(volume_val, '{:,.0f}'),
            getattr(self, 'sma_fast_period', 'N/A'), format_value(sma_fast_val),
            getattr(self, 'sma_slow_period', 'N/A'), format_value(sma_slow_val),
            getattr(self, 'adx_period', 'N/A'), format_value(adx_val, '{:.2f}'),
            getattr(self, 'adxr_period', 'N/A'), format_value(adxr_val, '{:.2f}'),
            format_value(di_plus_val, '{:.2f}'),
            format_value(di_minus_val, '{:.2f}'),
            getattr(self, 'rsi_period', 'N/A'), format_value(rsi_val, '{:.2f}'),
            getattr(self, 'rsi_ma_period', 'N/A'), format_value(rsi_ma_val, '{:.2f}'),
            getattr(self, 'atr_period', 'N/A'), format_value(atr_val),
            format_value(st_long_val) + (' ↑' if isinstance(st_long_val, (int, float)) and not pd.isna(st_long_val) else ''),
            format_value(st_short_val) + (' ↓' if isinstance(st_short_val, (int, float)) and not pd.isna(st_short_val) else ''),
            format_value(cpr_pivot_val),
            format_value(cpr_tc_val),
            format_value(cpr_bc_val),
            st_long_arrow='↑' if st_long_direction > 0 else '↓',
            st_short_arrow='↑' if st_short_direction > 0 else '↓',
            sma_fast_period=self.sma_fast_period,
            sma_slow_period=self.sma_slow_period,
            adx_period=self.adx_period,
            adxr_period=self.adxr_period,
            rsi_period=self.rsi_period,
            rsi_ma_period=self.rsi_ma_period,
            atr_period=self.atr_period
        )
        
        # Generate HTML for trades using new TradePosition structure
        trades_html = """
        <h2>Trade History</h2>
        <table border="1" class="dataframe">
            <tr>
                <th>Magic Number</th>
                <th>Entry Time</th>
                <th>Direction</th>
                <th>Entry Price</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Position Size</th>
                <th>Lots</th>
                <th>Commission</th>
                <th>Exit Time</th>
                <th>Exit Price</th>
                <th>Exit Reason</th>
                <th>Bars Held</th>
                <th>P/L (Gross)</th>
                <th>P/L (Net)</th>
                <th>P/L % (Equity)</th>
                <th>P/L % (Price)</th>
                <th>Risk:Reward</th>
                <th>Account Balance</th>
            </tr>
        """
        
        # Use completed trades from the new magic number-based system
        #completed_trades = self._completed_trades + [pos for pos in self._open_positions.values() if pos.is_closed]
        # Add this before the loop that processes trades in generate_html_report
        running_balance = self._initial_account_balance
        completed_trades = trades
        for trade in completed_trades:
            if trade.exit_price is not None:
                # Calculate trade metrics in points first (MT5 style)
                point = self.point_size  # Get point size from config
                # Use actual pip value from strategy configuration
                pip_value = getattr(self, '_pip_value_per_001_lot', 1.0)  # Default to 1.0 if not set
                
                if trade.direction == TradeDirection.LONG:
                    points = (trade.exit_price - trade.entry_price) / point
                    pl_gross = points * pip_value * trade.position_size
                    # Calculate P/L% based on Account Equity (Risk Management Focus)
                    pl_pct_equity = (pl_gross / self.account_balance) * 100 if self.account_balance > 0 else 0
                    # Keep original price-based P/L% for reference
                    pl_pct_price = (trade.exit_price / trade.entry_price - 1) * 100
                    risk = (trade.entry_price - trade.stop_loss) / point if trade.stop_loss else 0
                    reward = (trade.exit_price - trade.entry_price) / point if trade.exit_price else 0
                else:  # SHORT
                    points = (trade.entry_price - trade.exit_price) / point
                    pl_gross = points * pip_value * trade.position_size
                    # Calculate P/L% based on Account Equity (Risk Management Focus)
                    pl_pct_equity = (pl_gross / self.account_balance) * 100 if self.account_balance > 0 else 0
                    # Keep original price-based P/L% for reference
                    pl_pct_price = (1 - trade.exit_price / trade.entry_price) * 100
                    risk = (trade.stop_loss - trade.entry_price) / point if trade.stop_loss else 0
                    reward = (trade.entry_price - trade.exit_price) / point if trade.exit_price else 0
                
                # Calculate commission (round turn)
                commission = self.commission_per_lot * (trade.position_size / 1.0)  # Two way
                pl_net = pl_gross - (commission)
                
                # Update running balance
                running_balance += pl_net

                # Format the account balance for display
                account_balance_str = '${:,.2f}'.format(running_balance)
        
                # Calculate P/L% based on Account Equity for net profit/loss
                pl_pct_equity_net = (pl_net / self.account_balance) * 100 if self.account_balance > 0 else 0
                
                # Calculate risk:reward ratio
                risk_reward = abs(reward / risk) if risk != 0 else 0
                
                # Calculate bars held
                bars_held = 0
                if hasattr(trade, 'exit_time'):
                    # Convert timestamps to numpy datetime64 for proper subtraction
                    np_index = df.index.values.astype('datetime64[ns]')
                    np_timestamp = np.datetime64(trade.timestamp)
                    np_exit_time = np.datetime64(trade.exit_time)
                    
                    # Find the nearest index for entry and exit times
                    entry_idx = np.abs(np_index - np_timestamp).argmin()
                    exit_idx = np.abs(np_index - np_exit_time).argmin()
                    bars_held = exit_idx - entry_idx
                
                # Format values
                pl_color = 'green' if pl_net >= 0 else 'red'
                pl_gross_text = f'<span style="color: {pl_color}">{pl_gross:+,.2f}</span>'
                pl_net_text = f'<span style="color: {pl_color}">{pl_net:+,.2f}</span>'
                # Display both equity-based and price-based P/L%
                pl_pct_equity_text = f'<span style="color: {pl_color}">{pl_pct_equity_net:+.3f}%</span>'
                pl_pct_price_text = f'<span style="color: {pl_color}">{pl_pct_price:+.2f}%</span>'
                
                # Format trade row with all details
                # Format values with proper handling of None
                magic_number = str(trade.magic_number)
                timestamp = trade.timestamp.strftime('%Y-%m-%d %H:%M')
                direction = trade.direction.value
                entry = '{0:.2f}'.format(trade.entry_price) if trade.entry_price is not None else 'N/A'
                stop_loss = '{0:.2f}'.format(trade.stop_loss) if trade.stop_loss is not None else 'N/A'
                take_profit = '{0:.2f}'.format(trade.take_profit) if trade.take_profit is not None else 'N/A'
                position_value = '${:,.2f}'.format(trade.position_size * 1000)
                position_size = '{0:.2f}'.format(trade.position_size)
                commission_str = '${:,.2f}'.format(commission)
                exit_time = trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else 'N/A'
                exit_price = '{0:.2f}'.format(trade.exit_price) if trade.exit_price is not None else 'N/A'
                # Get descriptive exit reason
                exit_reason = self.get_exit_reason_description(trade.exit_reason) if trade.exit_reason else 'N/A'
                risk_reward_str = '{0:.2f}:1'.format(risk_reward) if risk_reward is not None else 'N/A'
                
                trade_row = """
                <tr>
                    <td>{0}</td>
                    <td>{1}</td>
                    <td>{2}</td>
                    <td>{3}</td>
                    <td>{4}</td>
                    <td>{5}</td>
                    <td>{6}</td>
                    <td>{7}</td>
                    <td>{8}</td>
                    <td>{9}</td>
                    <td>{10}</td>
                    <td>{11}</td>
                    <td>{12}</td>
                    <td>{13}</td>
                    <td>{14}</td>
                    <td>{15}</td>
                    <td>{16}</td>
                    <td>{17}</td>
                    <td>{18}</td>
                </tr>
                """.format(
                    magic_number, timestamp, direction, entry, stop_loss, take_profit,
                    position_value, position_size, commission_str, exit_time,
                    exit_price, exit_reason, bars_held, pl_gross_text,
                    pl_net_text, pl_pct_equity_text, pl_pct_price_text, risk_reward_str, account_balance_str
                )
                trades_html += trade_row
        
        trades_html += "</table>"
        
        # Combine all HTML content
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure equity_chart_html is defined even if no trades
        if 'equity_chart_html' not in locals():
            equity_chart_html = '<p>No trades executed - Equity curve not available</p>'
        # Calculate total P/L using the same method as individual trades
        point = self.point_size  # Get point size from config
        pip_value = getattr(self, '_pip_value_per_001_lot', 1.0)  # Default to 1.0 if not set
        
        total_pl = sum(
            ((t.exit_price - t.entry_price) / point * pip_value * t.position_size if t.direction == TradeDirection.LONG
             else (t.entry_price - t.exit_price) / point * pip_value * t.position_size)
            for t in completed_trades if t.exit_price is not None
        )
        
        # Calculate total commission (round turn for all trades)
        total_commission = sum(
            self.commission_per_lot * t.position_size # * 2  # Round turn commission
            for t in completed_trades if t.exit_price is not None
        )
        
        # Calculate net P/L (gross P/L minus total commission)
        net_pl = total_pl - total_commission
        
        pl_color = 'green' if total_pl >= 0 else 'red'
        net_pl_color = 'green' if net_pl >= 0 else 'red'

        # Calculate summary metrics
        total_trades = len(completed_trades)
        winning_trades = sum(1 for t in completed_trades if t.exit_price is not None and
                           ((t.direction == TradeDirection.LONG and t.exit_price > t.entry_price) or
                            (t.direction == TradeDirection.SHORT and t.exit_price < t.entry_price)))
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit factor using the same method as individual trades
        gross_profit = sum(
            ((t.exit_price - t.entry_price) / point * pip_value * t.position_size if t.direction == TradeDirection.LONG
             else (t.entry_price - t.exit_price) / point * pip_value * t.position_size)
            for t in completed_trades if t.exit_price is not None and
            ((t.direction == TradeDirection.LONG and t.exit_price > t.entry_price) or
             (t.direction == TradeDirection.SHORT and t.exit_price < t.entry_price))
        )
        
        gross_loss = abs(sum(
            ((t.exit_price - t.entry_price) / point * pip_value * t.position_size if t.direction == TradeDirection.LONG
             else (t.entry_price - t.exit_price) / point * pip_value * t.position_size)
            for t in completed_trades if t.exit_price is not None and
            ((t.direction == TradeDirection.LONG and t.exit_price < t.entry_price) or
             (t.direction == TradeDirection.SHORT and t.exit_price > t.entry_price))
        ))
        
        profit_factor = gross_profit / max(1e-9, gross_loss) if gross_loss > 0 else float('inf')
        
        # Calculate equity curve by simulating the account balance at each bar
        equity_curve = []
        trade_times = []
        running_equity = self.account_balance
        
        # Create a copy of trades to track open positions
        open_trades = []
        trade_index = 0
        
        # Sort trades by entry time, handling both TradeSignal objects and dictionaries
        def get_trade_timestamp(trade):
            if hasattr(trade, 'timestamp'):  # TradeSignal object
                return trade.timestamp
            elif isinstance(trade, dict) and 'timestamp' in trade:  # Dictionary
                return trade['timestamp']
            return None
            
        sorted_trades = sorted(
            [t for t in trades if (
                (hasattr(t, 'entry') and t.entry is not None and hasattr(t, 'timestamp')) or
                (isinstance(t, dict) and 'entry' in t and t['entry'] is not None and 'timestamp' in t)
            )],
            key=get_trade_timestamp
        )
        
        # Process each bar
        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_price = row['Close']
            
            # Check for new trades that should be opened at this timestamp
            while trade_index < len(sorted_trades) and sorted_trades[trade_index].timestamp <= timestamp:
                open_trades.append(sorted_trades[trade_index])
                trade_index += 1
            
            # Calculate current equity based on open positions
            current_equity = running_equity
            for trade in open_trades:
                if hasattr(trade, 'exit_time') and trade.exit_time is not None and trade.exit_time <= timestamp:
                    # This trade is closed, skip it
                    continue
                    
                # Calculate current P&L for open positions using the same method as individual trades
                if trade.direction == TradeDirection.LONG:
                    pl = (current_price - trade.entry) / point * pip_value * trade.position_size
                else:  # SHORT
                    pl = (trade.entry - current_price) / point * pip_value * trade.position_size
                
                current_equity += pl
            
            # Remove closed trades
            open_trades = [t for t in open_trades if not (hasattr(t, 'exit_time') and t.exit_time is not None and t.exit_time <= timestamp)]
            
            # Update running equity if we have closed trades
            for trade in sorted_trades:
                if hasattr(trade, 'exit_time') and trade.exit_time is not None and trade.exit_time == timestamp:
                    if trade.direction == TradeDirection.LONG:
                        pl = (trade.exit_price - trade.entry) / point * pip_value * trade.position_size
                    else:  # SHORT
                        pl = (trade.entry - trade.exit_price) / point * pip_value * trade.position_size
                    running_equity += pl
            
            equity_curve.append(current_equity)
            trade_times.append(timestamp)
        
        # Ensure we have at least 2 points for the equity curve
        if len(equity_curve) < 2:
            equity_curve = [self.account_balance, self.account_balance]
            trade_times = [df.index[0], df.index[-1]]
        
        # Calculate drawdown
        running_max = []
        drawdown_pct = []
        peak = -float('inf')
        
        logger.debug(f"Calculating drawdown for {len(equity_curve)} equity points")
        logger.debug(f"Sample equity values: {equity_curve[:5]}...{equity_curve[-5:] if len(equity_curve) > 10 else ''}")
        
        for i, equity in enumerate(equity_curve):
            if not isinstance(equity, (int, float)) or math.isnan(equity) or math.isinf(equity):
                logger.warning(f"Invalid equity value at index {i}: {equity}, using previous valid peak")
                if running_max:
                    peak = running_max[-1]
                else:
                    peak = self.account_balance
            elif equity > peak:
                peak = equity
            
            running_max.append(peak)
            
            # Calculate drawdown percentage
            if peak > 1e-9:  # Avoid division by zero
                dd = ((peak - equity) / peak) * 100
                dd_clamped = max(0.0, min(100.0, dd))  # Clamp between 0% and 100%
                drawdown_pct.append(dd_clamped)
                
                # Log significant drawdowns
                if dd_clamped > 1.0:  # Log drawdowns > 1%
                    logger.debug(f"Significant drawdown at index {i}: {dd_clamped:.2f}% (Equity: {equity:.2f}, Peak: {peak:.2f})")
            else:
                drawdown_pct.append(0.0)
                logger.warning(f"Invalid peak value ({peak}) when calculating drawdown at index {i}")
        
        # Calculate max drawdown - ensure we have valid values
        try:
            if not drawdown_pct or len(drawdown_pct) == 0:
                max_dd = 0.0
                logger.warning("No drawdown data available, setting max_dd to 0.0")
            else:
                max_dd = max(drawdown_pct)
                logger.debug(f"Max drawdown calculation - Found {len(drawdown_pct)} data points, Max: {max_dd}")
                
            # Ensure max_dd is a valid number
            if not isinstance(max_dd, (int, float)) or math.isnan(max_dd) or math.isinf(max_dd):
                logger.warning(f"Invalid max_dd value: {max_dd}, defaulting to 0.0")
                max_dd = 0.0
                
            # Ensure max_dd is within reasonable bounds (0-100%)
            max_dd = max(0.0, min(100.0, max_dd))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}", exc_info=True)
            max_dd = 0.0
            
        # Calculate previous method of drawdown (using only closed trades)        
        if len(trades) > 0:
            initial_balance = self.account_balance
            balance = initial_balance
            peak_balance = balance
            max_dd_value = 0.0
            trade_count = 0
            
            # Log initial state
            logger.debug(f"Previous method - Initial balance: ${initial_balance:.2f}")
            
            # Process trades in chronological order, handling both TradeSignal objects and dictionaries
            def get_trade_timestamp(trade):
                if hasattr(trade, 'timestamp'):  # TradeSignal object
                    return trade.timestamp
                elif isinstance(trade, dict) and 'timestamp' in trade:  # Dictionary
                    return trade['timestamp']
                return datetime.min  # Default to minimum datetime if no timestamp found
                
            def get_trade_attr(trade, attr, default=None):
                if hasattr(trade, attr):  # TradeSignal object
                    return getattr(trade, attr)
                elif isinstance(trade, dict) and attr in trade:  # Dictionary
                    return trade[attr]
                return default
                
            for trade in sorted(trades, key=get_trade_timestamp):
                exit_price = get_trade_attr(trade, 'exit_price')
                entry = get_trade_attr(trade, 'entry')
                direction_val = get_trade_attr(trade, 'direction')
                position_size = get_trade_attr(trade, 'position_size', 0)
                
                if exit_price is not None and entry is not None:
                    trade_count += 1
                    # Calculate P&L for this trade
                    if direction_val == TradeDirection.LONG or (isinstance(direction_val, str) and direction_val.upper() == 'LONG'):
                        pl = (exit_price - entry) * position_size
                        direction = "LONG"
                    else:  # SHORT
                        pl = (entry - exit_price) * position_size
                        direction = "SHORT"
                    
                    # Subtract commission using config value
                    commission = self.commission_per_lot * trade.position_size
                    pl_after_commission = pl - (2 * commission)  # Entry and exit commission
                    
                    # Store previous balance for logging
                    prev_balance = balance
                    
                    # Update balance
                    balance += pl_after_commission
                    
                    # Update peak balance
                    peak_balance = max(peak_balance, balance)
                    
                    # Calculate drawdown from peak
                    if peak_balance > 0:
                        drawdown = ((peak_balance - balance) / peak_balance) * 100
                        max_dd_value = max(max_dd_value, drawdown)
                    
                    # Log trade details
                    logger.debug(f"Trade {trade_count} ({direction}): Entry={trade.entry:.2f}, Exit={trade.exit_price:.2f}, "
                                f"P/L=${pl:.2f}, Comm=${2*commission:.2f}, "
                                f"Balance: ${prev_balance:.2f} -> ${balance:.2f}, "
                                f"Peak: ${peak_balance:.2f}, DD: {drawdown:.2f}%")
            
            # Drawdown analysis
            if trade_count > 0:
                # Find the largest single trade loss
                losing_trades = []
                for t in trades:
                    if hasattr(t, 'exit_price') and hasattr(t, 'entry') and t.exit_price is not None and t.entry is not None:
                        if t.direction == TradeDirection.LONG and t.exit_price < t.entry:
                            loss = (t.entry - t.exit_price) / point * pip_value * t.position_size
                            losing_trades.append((t, loss))
                        elif t.direction == TradeDirection.SHORT and t.exit_price > t.entry:
                            loss = (t.exit_price - t.entry) / point * pip_value * t.position_size
                            losing_trades = []  # Clear losing trades if any winning trade is found

                if losing_trades:
                    # Sort losing trades by loss amount (descending)
                    losing_trades.sort(key=lambda x: x[1], reverse=True)
                    largest_loss = losing_trades[0][1]
                    largest_loss_pct = (largest_loss / initial_balance) * 100
                    logger.info(f"Largest single trade loss: ${largest_loss:.2f} ({largest_loss_pct:.2f}% of initial balance)")
                
                # Calculate maximum consecutive losses
                consecutive_losses = 0
                max_consecutive_losses = 0
                current_streak = 0
                
                for t in sorted(trades, key=get_trade_timestamp):
                    exit_price = get_trade_attr(t, 'exit_price')
                    entry = get_trade_attr(t, 'entry')
                    direction_val = get_trade_attr(t, 'direction')
                    
                    if exit_price is not None and entry is not None:
                        is_long = direction_val == TradeDirection.LONG or (isinstance(direction_val, str) and direction_val.upper() == 'LONG')
                        is_short = not is_long
                        
                        is_loss = (is_long and exit_price < entry) or (is_short and exit_price > entry)
                        if is_loss:
                            current_streak += 1
                            max_consecutive_losses = max(max_consecutive_losses, current_streak)
                        else:
                            current_streak = 0
                
                logger.info(f"Maximum consecutive losses: {max_consecutive_losses}")
                
                # Track balance and drawdown for every trade
                balance = initial_balance
                peak_balance = initial_balance
                max_drawdown = 0.0
                
                # Process each trade in chronological order
                for trade in sorted(trades, key=get_trade_timestamp):
                    exit_price = get_trade_attr(trade, 'exit_price')
                    entry = get_trade_attr(trade, 'entry')
                    direction_val = get_trade_attr(trade, 'direction')
                    position_size = get_trade_attr(trade, 'position_size', 0.0)
                    
                    if exit_price is not None and entry is not None and position_size is not None:
                        # Calculate P&L for this trade using the same method as individual trades
                        is_long = direction_val == TradeDirection.LONG or (isinstance(direction_val, str) and direction_val.upper() == 'LONG')
                        if is_long:
                            pl = (exit_price - entry) / point * pip_value * position_size
                        else:  # SHORT
                            pl = (entry - exit_price) / point * pip_value * position_size
                        
                        # Track balance before trade close (entry point)
                        if peak_balance > 0:
                            current_dd = ((peak_balance - balance) / peak_balance) * 100
                            max_drawdown = max(max_drawdown, current_dd)
                            if current_dd > 0.1:  # Only log drawdowns > 0.1%
                                logger.info(f"Pre-trade drawdown: {current_dd:.2f}% | Balance: ${balance:.2f} | Peak: ${peak_balance:.2f}")
                        
                        # Apply commission (entry and exit)
                        commission = self.commission_per_lot * trade.position_size * 2
                        balance += (pl - commission)
                        
                        # Update peak balance if we've reached a new high
                        if balance > peak_balance:
                            peak_balance = balance
                        
                        # Calculate drawdown after trade close
                        if peak_balance > 0:
                            current_dd = ((peak_balance - balance) / peak_balance) * 100
                            max_drawdown = max(max_drawdown, current_dd)
                            if current_dd > 0.1:  # Only log drawdowns > 0.1%
                                logger.info(f"Post-trade drawdown: {current_dd:.2f}% | Balance: ${balance:.2f} | Peak: ${peak_balance:.2f}")
                
                # Final verification of max drawdown
                if peak_balance > 0:
                    final_dd = ((peak_balance - balance) / peak_balance) * 100
                    max_drawdown = max(max_drawdown, final_dd)
                
                # Log summary
                logger.info(f"Initial balance: ${initial_balance:.2f}")
                logger.info(f"Peak balance: ${peak_balance:.2f}")
                logger.info(f"Final balance: ${balance:.2f}")
                logger.info(f"Maximum drawdown from peak: {max_drawdown:.2f}%")
                
                # Update the max_dd variable used in the rest of the code
                max_dd = max_drawdown
                
                logger.info(f"Peak-to-trough drawdown: {max_dd:.2f}%")
                logger.info(f"Final balance: ${balance:.2f} (Starting: ${initial_balance:.2f})")
        
        logger.info(f"Final Max Drawdown: {max_dd:.2f}%")            
        logger.debug(f"Equity curve values: {equity_curve}")
        logger.debug(f"Running max values: {running_max}")
        
        # Ensure max_dd is a valid number
        if not isinstance(max_dd, (int, float)) or math.isnan(max_dd) or math.isinf(max_dd):
            logger.warning(f"Invalid max_dd value: {max_dd}, defaulting to 0.0")
            max_dd = 0.0
        
        # Create equity and drawdown chart
        equity_fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('Equity Curve', 'Drawdown')
        )
        
        # Add equity curve trace
        equity_fig.add_trace(
            go.Scatter(
                x=trade_times,
                y=equity_curve,
                mode='lines+markers',
                name='Equity',
                line=dict(color='#2ecc71', width=2),
                marker=dict(size=6, color='#27ae60')
            ),
            row=1, col=1
        )
        
        # Add running max trace
        equity_fig.add_trace(
            go.Scatter(
                x=trade_times,
                y=running_max,
                mode='lines',
                name='Peak',
                line=dict(color='#e74c3c', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Add drawdown trace
        equity_fig.add_trace(
            go.Scatter(
                x=trade_times,
                y=drawdown_pct,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='#e74c3c', width=2),
                fillcolor='rgba(231, 76, 60, 0.2)'
            ),
            row=2, col=1
        )
        
        # Update layout
        equity_fig.update_layout(
            title_text='Equity Curve & Drawdown',
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(family='Arial', size=12, color='#2c3e50'),
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation='h', y=1.02, yanchor='bottom', x=0.5, xanchor='center'),
            height=600
        )
        
        # Update y-axes
        equity_fig.update_yaxes(title_text='Equity ($)', row=1, col=1, showgrid=True, gridcolor='#e1e5eb')
        equity_fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1, showgrid=True, gridcolor='#e1e5eb')
        equity_fig.update_xaxes(showgrid=True, gridcolor='#e1e5eb')
        
        # Convert to HTML
        equity_chart_html = equity_fig.to_html(full_html=False)
        
        # Set default values if no trades
        if not trades:
            total_pl = 0
            win_rate = 0
            profit_factor = 0
            max_dd = 0.0  # Ensure float type for consistency
            equity_chart_html = '<p>No trades executed - Equity curve not available</p>'
            
        # Build HTML template with cascading sections using f-strings
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pl_color = 'var(--success-color)' if total_pl >= 0 else 'var(--danger-color)'
        
        # Generate the HTML content with f-strings for all variables
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Report - {self.symbol} - {current_time}</title>
            <style>
                :root {{
                    --primary-color: #2c3e50;
                    --secondary-color: #3498db;
                    --success-color: #2ecc71;
                    --danger-color: #e74c3c;
                    --light-bg: #f8f9fa;
                    --border-color: #e1e5eb;
                    --text-muted: #6c757d;
                }}
                
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 0;
                    padding: 0;
                    color: var(--primary-color);
                    line-height: 1.6;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    background: linear-gradient(135deg, var(--primary-color), #34495e);
                    color: white;
                    padding: 20px 0;
                    margin-bottom: 30px;
                    border-radius: 0 0 10px 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 600;
                }}
                
                .header p {{
                    margin: 5px 0 0;
                    opacity: 0.9;
                    font-size: 14px;
                }}
                
                .section {{
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 25px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    border: 1px solid var(--border-color);
                }}
                
                .section h2 {{
                    color: var(--primary-color);
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 10px;
                    margin-top: 0;
                    font-size: 22px;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                
                .metric-card {{
                    background: var(--light-bg);
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                
                .metric-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 16px;
                    color: var(--text-muted);
                }}
                
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 0;
                }}
                
                .profit {{ color: var(--success-color); }}
                .loss {{ color: var(--danger-color); }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    font-size: 14px;
                }}
                
                th, td {{
                    border: 1px solid var(--border-color);
                    padding: 10px 12px;
                    text-align: left;
                }}
                
                th {{
                    background-color: var(--light-bg);
                    font-weight: 600;
                }}
                
                tr:nth-child(even) {{
                    background-color: #fcfcfc;
                }}
                
                .chart-container {{
                    margin: 20px 0;
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                
                .config-table {{
                    font-size: 13px;
                }}
                
                .config-table th, .config-table td {{
                    padding: 8px 12px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>Trading Strategy Report - {self.symbol}</h1>
                    <p>Date Range: {start_date} to {end_date}</p>
                    <p>Generated on: {current_time}</p>
                </div>
            </div>
            
            <div class="container">
                <!-- Strategy Summary Section -->
                <div class="section">
                    <h2>Strategy Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h3>Total Trades</h3>
                            <p class="metric-value">{total_trades}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Win Rate</h3>
                            <p class="metric-value" style="color: var(--secondary-color);">{win_rate:.1f}%</p>
                        </div>
                        <div class="metric-card">
                            <h3>Gross P/L</h3>
                            <p class="metric-value" style="color: {pl_color};">${total_pl:+,.2f}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Total Commission</h3>
                            <p class="metric-value" style="color: var(--danger-color);">${total_commission:+,.2f}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Net P/L</h3>
                            <p class="metric-value" style="color: {net_pl_color};">${net_pl:+,.2f}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Profit Factor</h3>
                            <p class="metric-value" style="color: var(--success-color);">{profit_factor:.2f}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Max Drawdown</h3>
                            <p class="metric-value" style="color: var(--danger-color);">
                                {max_dd:.2f}%
                            </p>
                        </div>
                        
                    </div>
                </div>
                
                <!-- Equity & Drawdown Section -->
                <div class="section">
                    <h2>Equity & Drawdown</h2>
                    <div class="chart-container">
                        {equity_chart_html}
                    </div>
                </div>
                
                <!-- Price Chart Section -->
                <div class="section">
                    <h2>Price Action</h2>
                    <div class="chart-container">
                        {plot_html}
                    </div>
                </div>
                
                <!-- Configuration Section -->
                <div class="section">
                    <h2>Strategy Configuration</h2>
                    {config_html}
                </div>
                
                <!-- Trades Section -->
                <div class="section">
                    <h2>Trade History</h2>
                    {trades_html}
                </div>
                
                <!-- Indicators Section -->
                <div class="section">
                    <h2>Indicators</h2>
                    {indicators_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_file
    
    def run_strategy(self, symbol: str = "XAUUSD", timeframe: str = None, backtest_data: pd.DataFrame = None, execution_timeframe: str = None) -> List[TradeSignal]:
            """
            Run the trading strategy in either backtest or live mode.
            
            In backtest mode (when backtest_data is provided), it processes each bar sequentially
            to generate signals. In live mode, it fetches the latest data and generates signals
            based on the most recent complete bar.
            
            Args:
                symbol: Trading symbol (e.g., 'XAUUSD')
                timeframe: Data timeframe (e.g., 'M1', 'H1')
                backtest_data: DataFrame with OHLCV data for backtesting
                execution_timeframe: Timeframe for execution (same as timeframe if None)
                
            Returns:        
                List of TradeSignal objects generated during the run
            """
            # Initialize signals list
            signals = []
            completed_trades = []
            open_trades = []
            # Determine if we're in backtest mode
            is_backtest = backtest_data is not None
            
            try:
                # Use provided backtest data or fetch live data
                if is_backtest:
                    df = backtest_data
                    df = self.calculate_indicators(df) 

                    # Filter DataFrame to only include bars within the specified date range
                    if self.from_date and self.to_date:
                        # Convert string dates to datetime if needed
                        from_dt = pd.to_datetime(self.from_date) if isinstance(self.from_date, str) else self.from_date
                        to_dt = pd.to_datetime(self.to_date) if isinstance(self.to_date, str) else self.to_date
                        
                        # Filter DataFrame
                        original_count = len(df)
                        df = df[(df.index >= from_dt) & (df.index <= to_dt)]
                        
                        logger.info(f"Backtest date filtering: {original_count} -> {len(df)} bars")
                        logger.info(f"Trading period: {from_dt} to {to_dt}")
                        
                        if len(df) == 0:
                            logger.warning("No data found in specified date range!")
                            return []
                    
                    # # To export indicators for comparison
                    # df_export = df.copy()
                    # df_export['Time'] = df_export.index.strftime('%Y.%m.%d %H:%M')  # Format to match MT5
                    # #df['Time'] = pd.to_datetime(df['your_datetime_col']).dt.strftime('%Y.%m.%d %H:%M')
                    # df_export.rename(columns={
                    #     'adx': 'ADX14',
                    #     'rsi': 'RSI14',
                    #     'sma_fast': 'MA9',
                    #     'sma_slow': 'MA21',
                    #     'adx_di_plus': '+DI14',
                    #     'adx_di_minus': '-DI14',
                    #     'st_long': 'SPRTRND',
                    #     'st_short': 'SPRTRND',
                    #     'st_bullish': 'SPRTRND',
                    #     'st_bearish': 'SPRTRND',
                    #     # Add more as needed
                    # }, inplace=True)
                    # df_export.to_csv("python_indicators.csv", index=False)
                    
                    # Add prev_close to dataframe for SuperTrend exit condition
                    df['prev_close'] = df['Close'].shift(1)
                    
                    # Note: update_open_positions is not needed in backtesting mode
                    # as the proper candle-by-candle exit checking is done in the loop below
                    # self.monitor_exit_conditions(df)  # Uncomment for debugging
                    
                    logger.info(f"\nRun Strategy: Running backtest on {len(df)} bars of {symbol} data")

                    for i in range(1, len(df)):
                        current_bar = df.iloc[i]
                        prev_bar = df.iloc[i-1]
                        
                        # 📊 DETAILED BAR LOGGING
                        bar_info = f"\n{'='*60}\n📊 BAR {i}/{len(df)}: {current_bar.name}\n📊 OHLC: O:{current_bar['Open']:.2f} H:{current_bar['High']:.2f} L:{current_bar['Low']:.2f} C:{current_bar['Close']:.2f}"
                        print(bar_info)
                        logger.info(bar_info)
                        
                        # Check entry permission after trade exits are processed
                        check_entries = True  # Default: allow entries

                        if self._tp_block_active:
                            price_touched_st_short = False
                            bars_after_tp = pd.DataFrame()
                            
                            if self._tp_block_start_time is not None:
                                # Calculate time since TP block started
                                time_since_tp = current_bar.name - self._tp_block_start_time
                                minutes_since_tp = time_since_tp.total_seconds() / 60
                                
                                # Minimum cooldown period from config
                                
                                if minutes_since_tp < self.tp_cooldown_minutes:
                                    logger.info(f"  ⏰ COOLDOWN PERIOD: {minutes_since_tp:.1f} minutes since TP (need {self.tp_cooldown_minutes} min) - blocking entries")
                                    check_entries = False
                                else:
                                    # After cooldown period, check if price has touched st_short
                                    bars_after_tp = df[df.index > self._tp_block_start_time]
                                    
                                    if not bars_after_tp.empty:
                                        if self.last_trade_exit['direction'] == TradeDirection.LONG:
                                            # For long exits, check if price has come down to touch st_short
                                            price_touched_st_short = any((row['Low'] <= row['st_short'] <= row['High']) for _, row in bars_after_tp.iterrows())
                                        else:  # SHORT
                                            # For short exits, check if price has come up to touch st_short
                                            price_touched_st_short = any((row['Low'] <= row['st_short'] <= row['High']) for _, row in bars_after_tp.iterrows())
                                    else:
                                        price_touched_st_short = False
                                    
                                    if price_touched_st_short:
                                        logger.info(f"  ✅ Cooldown passed + Price touched st_short - allowing entries")
                                    else:
                                        logger.info(f"  🚫 Cooldown passed but price has not touched st_short - blocking entries")

                            logger.info("🔒 ENTRY PERMISSION DEBUG:")
                            logger.info(f"  TP block active: {'✅ Yes' if self._tp_block_active else '❌ No'}")
                            logger.info(f"  TP block start time: {self._tp_block_start_time}")
                            if self._tp_block_start_time is not None:
                                time_since_tp = current_bar.name - self._tp_block_start_time
                                minutes_since_tp = time_since_tp.total_seconds() / 60
                                logger.info(f"  Time since TP: {minutes_since_tp:.1f} minutes")
                                logger.info(f"  Minimum cooldown: {self.tp_cooldown_minutes} minutes")
                            logger.info(f"  Last trade exit time: {self.last_trade_exit.get('time')}")
                            logger.info(f"  Last trade exit reason: {self.last_trade_exit.get('reason')}")
                            logger.info(f"  Price touched st_short since TP: {'✅ Yes' if price_touched_st_short else '❌ No'}")
                            if not bars_after_tp.empty:
                                current_price = current_bar['Close']
                                current_st_short = current_bar.get('st_short', 0)
                                logger.info(f"  📊 Current price: {current_price:.2f}, Current st_short: {current_st_short:.2f}")
                                logger.info(f"  📊 Distance to st_short: {abs(current_price - current_st_short):.2f} points")
                                logger.info(f"  Bars since TP block: {len(bars_after_tp)}")
                            
                            if self._tp_block_start_time is not None:
                                time_since_tp = current_bar.name - self._tp_block_start_time
                                minutes_since_tp = time_since_tp.total_seconds() / 60
                                
                                if minutes_since_tp >= self.tp_cooldown_minutes and price_touched_st_short:
                                    logger.info("  ✅ Cooldown passed + Price touched st_short - clearing TP block flag")
                                    self._tp_block_active = False
                                    self._tp_block_start_time = None
                                    check_entries = True
                                else:
                                    logger.info("  🚫 Blocking entries (cooldown or no st_short touch)")
                                    check_entries = False
                        else:
                            logger.info("🔒 ENTRY PERMISSION DEBUG:")
                            logger.info("  No TP block in effect - entries allowed")
                            check_entries = True

                        logger.info(f"  ✨ Final check_entries result: {'✅ ALLOWED' if check_entries else '❌ BLOCKED'}")
                        logger.info("  ⚠️ If check_entries is BLOCKED, no trades will be generated even with 100% signal strength")

                        # Only generate entry signals if check_entries is True
                        bar_signals = []
                        if check_entries:
                            bar_signals = self.generate_signal_for_bar(current_bar, prev_bar, len(open_trades))
                        
                        # Filter signals based on entry permission
                        if not check_entries:
                            logger.info("  🚫 ENTRY CONDITIONS BLOCKED - No trades will be generated due to TP cooldown")
                            bar_signals = []  # Clear all signals
                        else:
                            logger.info("  ✅ ENTRY CONDITIONS ALLOWED - Trades may be generated if conditions are met")
                        
                        # Show detailed condition analysis for this bar (only when signals are generated)
                        if bar_signals:
                            for signal in bar_signals:
                                direction = signal.direction.name
                                if self._trading_mode == 'live':
                                    self.debug_signal_conditions_mtf(direction, current_bar, prev_bar)
                                else:
                                    self.debug_signal_conditions_mtf_testing(direction, current_bar, prev_bar)

                        
                        else:
                            logger.info(f"❌ NO SIGNALS: No trading signals generated for this bar")
                            
                            # DO NOT reset TP block when no signals are generated
                            # The TP block should only be reset when price touches st_short
                            # This ensures we wait for the proper condition before allowing new entries
                        
                        for signal in bar_signals:
                            # Option 1: Only one trade at a time (flat-to-flat)
                            if len(open_trades) > 0:
                                logger.info(f"    ⚠️ SKIPPED: Already have {len(open_trades)} open trade(s)")
                                continue  # Skip opening new trade

                            # # Option 2: Only one trade per direction
                            # if has_open_trade_in_direction(open_trades, signal.direction):
                            #     continue  # Skip opening new trade in this direction

                            # # Option 3: No opposite direction trades
                            # if has_open_trade_opposite(open_trades, signal.direction):
                            #     continue  # Skip opening new trade if opposite is open

                            # Adjust entry price if TP block was just cleared and price touched st_short
                            adjusted_entry = signal.entry
                            
                            # Check if this is the first entry after TP block was cleared
                            if (hasattr(self, '_tp_block_active') and not self._tp_block_active and 
                                self._tp_block_start_time is not None and 
                                self.last_trade_exit.get('reason') == 'TP_HIT'):
                                
                                # Check if price touched st_short since TP hit
                                bars_after_tp = df[df.index > self._tp_block_start_time]
                                if not bars_after_tp.empty:
                                    price_touched_st_short = any((row['Low'] <= row['st_short'] <= row['High']) for _, row in bars_after_tp.iterrows())
                                    if price_touched_st_short:
                                        adjusted_entry = current_bar['st_short']  # Use st_short level as entry
                                        logger.info(f"    🎯 ADJUSTING ENTRY: Using st_short level {adjusted_entry:.2f} instead of close {signal.entry:.2f}")
                                
                            logger.info(f"    ✅ OPENING TRADE: {signal.direction.name} at {adjusted_entry:.2f}")
                            
                            # Use open_position method to ensure trade logging
                            position = self.open_position(
                                direction=signal.direction,
                                entry_price=adjusted_entry,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                position_size=signal.position_size,
                                timestamp=signal.timestamp,
                                symbol=signal.symbol,
                                trailing_stop=signal.trailing_stop,
                                atr_trailing_multiplier=signal.atr_trailing_multiplier,
                                hidden_stop_loss=True
                            )
                            open_trades.append(position)

                        # 2. Check for exits for all open trades
                        if open_trades:
                            logger.info(f"🔍 CHECKING {len(open_trades)} OPEN TRADE(S) FOR EXITS:")
                            
                        for trade in open_trades[:]:  # Copy to avoid modification during iteration
                            # Show trade details before exit check
                            logger.info(f"  🔍 Trade {trade.magic_number} ({trade.direction.name}): Entry:{trade.entry_price:.2f} SL:{trade.stop_loss:.2f} TP:{trade.take_profit:.2f}")
                            
                            # Show detailed exit condition analysis for this trade
                            exit_conditions = self._check_position_exit_conditions(trade, current_bar, prev_bar)
                            
                            # Log exit conditions using helper method
                            self._log_exit_conditions(exit_conditions)

                            if trade.should_close(current_bar):
                                logger.info(f"    🚪 EXIT TRIGGERED: {trade.exit_reason}")
                                logger.info(f"    📊 Exit OHLC: O:{current_bar['Open']:.2f} H:{current_bar['High']:.2f} L:{current_bar['Low']:.2f} C:{current_bar['Close']:.2f}")
                                logger.info(f"    💰 Exit Price: {trade.exit_price:.2f}")
                                
                                # Show exit condition details
                                if trade.direction == TradeDirection.LONG:
                                    if current_bar['Low'] <= trade.stop_loss:
                                        logger.info(f"    📉 SL HIT: Bar Low ({current_bar['Low']:.2f}) <= SL ({trade.stop_loss:.2f})")
                                    elif current_bar['Close'] >= trade.take_profit:
                                        logger.info(f"    📈 TP HIT: Bar Close ({current_bar['Close']:.2f}) >= TP ({trade.take_profit:.2f})")
                                else:  # SHORT
                                    if current_bar['High'] >= trade.stop_loss:
                                        logger.info(f"    📉 SL HIT: Bar High ({current_bar['High']:.2f}) >= SL ({trade.stop_loss:.2f})")
                                    elif current_bar['Close'] <= trade.take_profit:
                                        logger.info(f"    📈 TP HIT: Bar Close ({current_bar['Close']:.2f}) <= TP ({trade.take_profit:.2f})")
                                
                                trade.exit_time = current_bar.name
                                # trade.exit_price is set by should_close
                                trade.is_closed = True
                                
                                # Use close_position method to ensure trade logging
                                self.close_position(
                                    magic_number=trade.magic_number,
                                    exit_price=trade.exit_price,
                                    exit_reason=trade.exit_reason,
                                    exit_time=trade.exit_time
                                )
                                
                                # Update last trade exit information for entry permission logic
                                self.last_trade_exit = {
                                    'time': trade.exit_time,
                                    'reason': trade.exit_reason,
                                    'direction': trade.direction,
                                    'entry_price': trade.entry_price,
                                    'exit_price': trade.exit_price
                                }
                                
                                completed_trades.append(trade)
                                open_trades.remove(trade)
                                logger.info(f"    ✅ TRADE CLOSED: {trade.direction.name} {trade.entry_price:.2f} → {trade.exit_price:.2f}")
                                logger.info(f"    📝 Updated last_trade_exit: {trade.exit_reason} at {trade.exit_time}")
                                
                                # After trade exit, check if we should block new entries due to TP cooldown
                                if trade.exit_reason == 'TP_HIT':
                                    logger.info("    🔒 TP_HIT detected - entry permission will be checked for next bars...")
                                    # Set a flag to block entries on the next bar
                                    self._tp_block_active = True
                                    self._tp_block_start_time = trade.exit_time
                            else:
                                logger.info(f"    ⏳ NO EXIT: Trade continues...")

                    # After loop, you may want to close any trades still open at the last bar
                    if open_trades:
                        logger.info(f"\n🔚 CLOSING {len(open_trades)} TRADE(S) AT END OF BACKTEST:")
                        for trade in open_trades:
                            logger.info(f"  📊 Final OHLC: O:{df.iloc[-1]['Open']:.2f} H:{df.iloc[-1]['High']:.2f} L:{df.iloc[-1]['Low']:.2f} C:{df.iloc[-1]['Close']:.2f}")
                            trade.exit_time = df.index[-1]
                            trade.exit_price = df.iloc[-1]['Close']
                            trade.exit_reason = "END_OF_BACKTEST"
                            trade.is_closed = True
                            
                            # Use close_position method to ensure trade logging
                            self.close_position(
                                magic_number=trade.magic_number,
                                exit_price=trade.exit_price,
                                exit_reason=trade.exit_reason,
                                exit_time=trade.exit_time
                            )
                            
                            completed_trades.append(trade)
                            logger.info(f"    ✅ CLOSED: {trade.direction.name} {trade.entry_price:.2f} → {trade.exit_price:.2f} (End of backtest)")

                    logger.info(f"\n{'='*60}")
                    print(f"📊 BACKTEST SUMMARY: {len(completed_trades)} completed trades")
                    for i, trade in enumerate(completed_trades, 1):
                        pnl = (trade.exit_price - trade.entry_price) if trade.direction == TradeDirection.LONG else (trade.entry_price - trade.exit_price)
                        pnl_pips = pnl / 0.01  # Convert to pips for XAUUSD
                        print(f"  {i}. {trade.direction.name} {trade.entry_price:.2f} → {trade.exit_price:.2f} ({pnl_pips:+.1f} pips) [{trade.exit_reason}]")

                    return completed_trades  # Or signals, as needed
                else:
                    # LIVE TRADING MODE - Updated to handle position management                    
                    # Initialize timezone detection for live trading                    
                    if not is_backtest:
                        logger.info("RunStrategy: Live trading mode is active")
                        
                        # # Log timezone configuration for debugging
                        # time_info = self.get_current_time_info()
                        # logger.info(f"Timezone Configuration:")
                        # logger.info(f"  Server Timezone: {self._server_timezone} (Auto-detect: {self._auto_detect_server_timezone})")
                        # logger.info(f"  Execution Timezone: {self._execution_timezone}")
                        # logger.info(f"  Default Timezone: {self._default_timezone}")
                        # logger.info(f"Current Times:")
                        # logger.info(f"  UTC Time: {time_info['utc']}")
                        # logger.info(f"  Server Time: {time_info['server']}")
                        # logger.info(f"  Execution Time: {time_info['execution']}")
                        # logger.info(f"  Default Time: {time_info['default']}")
                        
                        # If server timezone is still None and auto-detect is enabled, try one more time
                        # This is a fallback in case the detection in __init__ failed
                        if self._auto_detect_server_timezone and self._server_timezone is None:
                            logger.warning("Server timezone is still None. Attempting detection again...")
                            self.detect_server_timezone()
                            if self._server_timezone is not None:
                                logger.info(f"Server timezone successfully detected: {self._server_timezone}")
                            else:
                                logger.warning("Server timezone detection failed again. Using default timezone for calculations.")
                        
                        # Log final timezone configuration
                        time_info = self.get_current_time_info()
                        # logger.info(f"\n🌍 LIVE TRADING START:")
                        # logger.info(f"  Default TZ: {time_info['default'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        # logger.info(f"  Execution TZ: {time_info['execution'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        # logger.info(f"  Server TZ: {time_info['server'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        # logger.info(f"  UTC: {time_info['utc'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    
                    # Fetch live data from MT5
                    df = self.get_mt5_data(symbol, timeframe)
                    if df is None:
                        logger.error("\nRun Strategy: No data received from MT5")
                        return []
                        
                    if len(df) < 2:
                        logger.warning(f"\nRun Strategy: Not enough data received from MT5: only {len(df)} bars")
                        return []
                                        
                    # Calculate indicators
                    df = self.calculate_indicators(df)
                    
                    # Add prev_close to dataframe for SuperTrend exit condition
                    df['prev_close'] = df['Close'].shift(1)
                    
                    # Get current open trades from MT5 or your position manager
                    open_positions_result = self.get_open_positions()

                    # Handle both dict and list returns
                    if isinstance(open_positions_result, dict):
                        open_trades = list(open_positions_result.values())
                    elif isinstance(open_positions_result, list):
                        open_trades = open_positions_result
                    else:
                        open_trades = []  # Default to empty list
                    
                    # For live trading, process the most recent complete bar
                    if len(df) >= 2:
                        current_bar = df.iloc[-1]  # Latest complete bar
                        prev_bar = df.iloc[-2]     # Previous bar
                        
                        # Get current time in execution timezone
                        current_time_info = self.get_current_time_info()
                        
                        # 📊 DETAILED BAR LOGGING
                        bar_info = f"\n{'='*125}\n📊 LIVE BAR: {current_bar.name} (Server)"
                        bar_info += f" | 🌍 Current Time: {current_time_info['execution'].strftime('%Y-%m-%d %H:%M:%S %Z')}"
                        bar_info += f" | 🎯 Trading Style: {self._determine_trading_style(current_bar)}"
                        bar_info += f"\n{'='*125}\n"
                        bar_info += f"🔸 Symbol: {strategy.symbol}"
                        bar_info += f" | 📈 Timeframe: {strategy.execution_timeframe}"
                        bar_info += f" | 🔢 Max Open Trades: {getattr(strategy, '_max_open_trades', 'N/A')}"
                        
                        # Display different risk values based on trading style
                        trading_style = self._determine_trading_style(current_bar)
                        if trading_style == 'swing':
                            risk_value = getattr(strategy, 'swing_risk_per_trade_pct', 'N/A')
                        else:  # scalp or any other style
                            risk_value = getattr(strategy, '_risk_per_trade_pct', 'N/A')
                            
                        bar_info += f" | 💰 Risk per Trade: {risk_value}%"
                        bar_info += f" | ⏰ Strart Hour: {strategy.start_hour}"
                        bar_info += f" | ⏰ End Hour: {strategy.end_hour}"
                        bar_info += f"\n{'='*125}\n"
                        bar_info += f"📰 Avoid News: {'✅' if strategy._avoid_news_times else '❌'}"
                        bar_info += f" | 🏦 Avoid Bank Holidays: {'✅' if strategy._avoid_bank_holidays else '❌'}"
                        bar_info += f" | 📉 Daily Drawdown Limit: {'✅' if strategy._daily_drawdown_limit else '❌'}"
                        bar_info += f" | 📊 ATR Based Trading: {'✅' if strategy.atr_based_trading else '❌'}"
                        #bar_info += f" | 📊 Soft TSL: {'✅ Enabled' if strategy.soft_tsl_enabled else '❌ Disabled'}"
                        print(bar_info)
                        #logger.info(bar_info)
                                           
                        # 🔍 DEBUG: Show all signal conditions
                        if current_bar['sma_fast'] > current_bar['sma_slow']:
                            self.debug_signal_conditions_mtf("LONG", current_bar, prev_bar)
                        elif current_bar['sma_fast'] < current_bar['sma_slow']:
                            self.debug_signal_conditions_mtf("SHORT", current_bar, prev_bar)
                        else:
                            # Optionally, handle the case where they are equal
                            print("No clear trend: SMA Fast equals SMA Slow")
                            
                        # 1. CHECK EXIT CONDITIONS FOR OPEN TRADES FIRST
                        if open_trades:
                            logger.info(f"🔍 CHECKING {len(open_trades)} OPEN TRADE(S) FOR EXITS:")
                            
                            for trade in open_trades[:]:  # Copy to avoid modification during iteration
                                # Show trade details before exit check
                                logger.info(f"  🔍 Trade {trade.magic_number} ({trade.direction.name}): Entry:{trade.entry_price:.2f} SL:{trade.stop_loss:.2f} TP:{trade.take_profit:.2f}")
                                
                                # Show detailed exit condition analysis for this trade
                                exit_conditions = self._check_position_exit_conditions(trade, current_bar, prev_bar)
                                
                                logger.info(f"\n{'='*125}")
                                logger.info(f"    🚪 EXIT CONDITIONS ANALYSIS:")
                                for condition in exit_conditions:
                                    status_icon = "🔴" if condition['status'] == 'ACTIVE' else "🟢"
                                    logger.info(f"      {status_icon} {condition['type']}: {condition['condition']}")
                                    if condition['exit_price']:
                                        logger.info(f"         Exit Price: {condition['exit_price']:.2f}")                            
                                logger.info(f"\n{'='*125}")

                                if trade.should_close(current_bar):

                                    # Convert exit time to default timezone
                                    exit_time_execution = current_bar.name
                                        
                                    logger.info(f"    🚪 EXIT TRIGGERED: {trade.exit_reason}")
                                    logger.info(f"    📊 Exit OHLC: O:{current_bar['Open']:.2f} H:{current_bar['High']:.2f} L:{current_bar['Low']:.2f} C:{current_bar['Close']:.2f}")
                                    logger.info(f"    💰 Exit Price: {trade.exit_price:.2f}")
                                    logger.info(f"    📅 Exit Time: {exit_time_execution.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                                    
                                    # Show exit condition details
                                    if trade.direction == TradeDirection.LONG:
                                        if current_bar['Low'] <= trade.stop_loss:
                                            logger.info(f"    📉 SL HIT: Bar Low ({current_bar['Low']:.2f}) <= SL ({trade.stop_loss:.2f})")
                                        elif current_bar['Close'] >= trade.take_profit:
                                            logger.info(f"    📈 TP HIT: Bar Close ({current_bar['Close']:.2f}) >= TP ({trade.take_profit:.2f})")
                                    else:  # SHORT
                                        if current_bar['High'] >= trade.stop_loss:
                                            logger.info(f"    📉 SL HIT: Bar High ({current_bar['High']:.2f}) >= SL ({trade.stop_loss:.2f})")
                                        elif current_bar['Close'] <= trade.take_profit:
                                            logger.info(f"    📈 TP HIT: Bar Close ({current_bar['Close']:.2f}) <= TP ({trade.take_profit:.2f})")
                                    
                                    trade.exit_time = current_bar.name
                                    # trade.exit_price is set by should_close
                                    trade.is_closed = True
                                    
                                    # Use close_position method to ensure trade logging
                                    self.close_position(
                                        magic_number=trade.magic_number,
                                        exit_price=trade.exit_price,
                                        exit_reason=trade.exit_reason,
                                        exit_time=trade.exit_time
                                    )
                                    
                                    # Update last trade exit information for entry permission logic
                                    self.last_trade_exit = {
                                        'time': trade.exit_time,
                                        'reason': trade.exit_reason,
                                        'direction': trade.direction,
                                        'entry_price': trade.entry_price,
                                        'exit_price': trade.exit_price
                                    }
                                    
                                    completed_trades.append(trade)
                                    open_trades.remove(trade)
                                    logger.info(f"    ✅ TRADE CLOSED: {trade.direction.name} {trade.entry_price:.2f} → {trade.exit_price:.2f}")
                                    logger.info(f"    📝 Updated last_trade_exit: {trade.exit_reason} at {trade.exit_time}")
                                    
                                    # After trade exit, check if we should block new entries due to TP cooldown
                                    if trade.exit_reason == 'TP_HIT':
                                        logger.info("    🔒 TP_HIT detected - entry permission will be checked for next bars...")
                                        # Set a flag to block entries on the next bar
                                        self._tp_block_active = True
                                        self._tp_block_start_time = trade.exit_time
                                else:
                                    logger.info(f"    ⏳ NO EXIT: Trade continues...")

                        # 2. CHECK ENTRY PERMISSION (Same logic as backtesting)
                        check_entries = True  # Default: allow entries

                        if self._tp_block_active:
                            price_touched_st_short = False
                            
                            if self._tp_block_start_time is not None:
                                # Convert both times to UTC for consistent comparison
                                current_time_utc = self._ensure_utc_datetime(current_bar.name)
                                
                                # Ensure tp_block_start_time is timezone-aware and in UTC
                                tp_block_time_utc = self._ensure_utc_datetime(self._tp_block_start_time)
                                
                                # Store original tp_block_start_time for dataframe filtering later
                                tp_block_start_time = self._tp_block_start_time
                                 
                                # Calculate time since TP block started
                                time_since_tp = current_time_utc - tp_block_time_utc
                                minutes_since_tp = time_since_tp.total_seconds() / 60
                                
                                if minutes_since_tp < self.tp_cooldown_minutes:
                                    logger.info(f"  ⏰ COOLDOWN PERIOD: {minutes_since_tp:.1f} minutes since TP (need {self.tp_cooldown_minutes} min) - blocking entries")
                                    check_entries = False
                                else:
                                    # After cooldown period, check if price has touched st_short
                                    bars_after_tp = df[df.index > tp_block_start_time]
                                    
                                    if not bars_after_tp.empty:
                                        if self.last_trade_exit['direction'] == TradeDirection.LONG:
                                            # For long exits, check if price has come down to touch st_short
                                            price_touched_st_short = any((row['Low'] <= row['st_short'] <= row['High']) for _, row in bars_after_tp.iterrows())
                                        else:  # SHORT
                                            # For short exits, check if price has come up to touch st_short
                                            price_touched_st_short = any((row['Low'] <= row['st_short'] <= row['High']) for _, row in bars_after_tp.iterrows())
                                    else:
                                        price_touched_st_short = False
                                    
                                    if price_touched_st_short:
                                        logger.info(f"  ✅ Cooldown passed + Price touched st_short - allowing entries")
                                    else:
                                        logger.info(f"  🚫 Cooldown passed but price has not touched st_short - blocking entries")

                            logger.info("🔒 ENTRY PERMISSION DEBUG:")
                            logger.info(f"  TP block active: {'✅ Yes' if self._tp_block_active else '❌ No'}")
                            logger.info(f"  TP block start time: {tp_block_start_time}")
                            if tp_block_start_time is not None:
                                time_since_tp = current_bar.name - tp_block_start_time
                            #     if hasattr(time_since_tp, 'tz_convert'):
                            #         time_since_tp = time_since_tp.tz_convert(self.execution_timezone)
                                minutes_since_tp = time_since_tp.total_seconds() / 60
                                logger.info(f"  Time since TP: {minutes_since_tp:.1f} minutes")
                                logger.info(f"  Minimum cooldown: {self.tp_cooldown_minutes} minutes")
                            logger.info(f"  Last trade exit time: {self.last_trade_exit.get('time')}")
                            logger.info(f"  Last trade exit reason: {self.last_trade_exit.get('reason')}")
                            logger.info(f"  Price touched st_short since TP: {'✅ Yes' if price_touched_st_short else '❌ No'}")
                            
                            if tp_block_start_time is not None:
                                # Use the same UTC conversion approach as above
                                current_time_utc = self._ensure_utc_datetime(current_bar.name)
                                tp_block_time_utc = self._ensure_utc_datetime(self._tp_block_start_time)
                                    
                                # Calculate time difference in UTC
                                time_since_tp = current_time_utc - tp_block_time_utc
                                minutes_since_tp = time_since_tp.total_seconds() / 60
                                
                                if minutes_since_tp >= self.tp_cooldown_minutes and price_touched_st_short:
                                    logger.info("  ✅ Cooldown passed + Price touched st_short - clearing TP block flag")
                                    self._tp_block_active = False
                                    self._tp_block_start_time = None
                                    check_entries = True
                                else:
                                    logger.info("  🚫 Blocking entries (cooldown or no st_short touch)")
                                    check_entries = False
                        else:
                            logger.info("🔒 ENTRY PERMISSION DEBUG:")
                            logger.info("  No TP block in effect - entries allowed")
                            check_entries = True

                        logger.info(f"  ✨ Final check_entries result: {'✅ ALLOWED' if check_entries else '❌ BLOCKED'}")

                        # 3. GENERATE ENTRY SIGNALS (Only if check_entries is True)
                        bar_signals = []
                        if check_entries:
                            bar_signals = self.generate_signal_for_bar(current_bar, prev_bar, len(open_trades))
                        
                        # Filter signals based on entry permission
                        if not check_entries:
                            logger.info("  🚫 ENTRY CONDITIONS BLOCKED - No trades will be generated due to TP cooldown")
                            bar_signals = []  # Clear all signals
                        else:
                            logger.info("  ✅ ENTRY CONDITIONS ALLOWED - Trades may be generated if conditions are met")
                        
                        # Show detailed condition analysis for this bar (only when signals are generated)
                        if bar_signals:                            
                            logger.info(f"✅ SIGNAL: Trading signal generated for this bar")
                        else:
                            logger.info(f"❌ NO SIGNAL: No trading signal generated for this bar")
                        
                        # 4. PROCESS ENTRY SIGNALS
                        for signal in bar_signals:
                            # Option 1: Only one trade at a time (flat-to-flat)
                            if len(open_trades) > 0:
                                logger.info(f"    ⚠️ SKIPPED: Already have {len(open_trades)} open trade(s)")
                                continue  # Skip opening new trade

                            # Adjust entry price if TP block was just cleared and price touched st_short
                            adjusted_entry = signal.entry
                            
                            # Check if this is the first entry after TP block was cleared
                            if (hasattr(self, '_tp_block_active') and not self._tp_block_active and 
                                self._tp_block_start_time is not None and 
                                self.last_trade_exit.get('reason') == 'TP_HIT'):
                                
                                # Check if price touched st_short since TP hit
                                bars_after_tp = df[df.index > self._tp_block_start_time]
                                if not bars_after_tp.empty:
                                    price_touched_st_short = any((row['Low'] <= row['st_short'] <= row['High']) for _, row in bars_after_tp.iterrows())
                                    if price_touched_st_short:
                                        adjusted_entry = current_bar['st_short']  # Use st_short level as entry
                                        logger.info(f"    🎯 ADJUSTING ENTRY: Using st_short level {adjusted_entry:.2f} instead of close {signal.entry:.2f}")
                            
                            # Log entry with timezone info
                            entry_time_info = self.get_current_time_info()
                            logger.info(f"    ✅ OPENING TRADE: {signal.direction.name} at {adjusted_entry:.2f}")
                            logger.info(f"    ⏰ Entry Time: {entry_time_info['execution'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                            
                            # Use open_position method to ensure trade logging
                            position = self.open_position(
                                direction=signal.direction,
                                entry_price=adjusted_entry,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                position_size=signal.position_size,
                                timestamp=signal.timestamp,
                                symbol=signal.symbol,
                                trailing_stop=signal.trailing_stop,
                                atr_trailing_multiplier=signal.atr_trailing_multiplier,
                                hidden_stop_loss=True
                            )
                            
                            # Add to signals list for return
                            signals.append(signal)
                            
                            # Update open_trades list (for next iteration if running continuously)
                            open_trades.append(position)

                    # Display current market conditions
                    latest_bar = df.iloc[-1]
                    current_time_info = self.get_current_time_info()

                    #print(f"\nRun Strategy: Latest bar: {latest_bar.name}")
                    #print(f"Current Time (Execution): {current_time_info['execution'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    #print(f"Current Time (Server): {current_time_info['server'].strftime('%Y-%m-%d %H:%M:%S UTC%z')}")
                    
                    # Handle different column name formats (lowercase or capitalized)
                    open_col = 'Open' if 'Open' in df.columns else 'open'
                    high_col = 'High' if 'High' in df.columns else 'high'
                    low_col = 'Low' if 'Low' in df.columns else 'low'
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    
                    #print(f"\nOHLC: {latest_bar[open_col]:.2f}/{latest_bar[high_col]:.2f}/{latest_bar[low_col]:.2f}/{latest_bar[close_col]:.2f}")
                    
                    # Display key indicator values
                    if hasattr(self, '_df') and not self._df.empty:
                        last_row = self._df.iloc[-1]
                        print("\nRun Strategy: Indicator Analysis:")
                        for col in self._df.columns:
                            if col not in ['open', 'high', 'low', 'close', 'volume', 'time']:
                                try:
                                    value = last_row[col]
                                    if isinstance(value, (int, float)):
                                        print(f"  {col}: {value:.4f}")
                                    else:
                                        print(f"  {col}: {value}")
                                except Exception as e:
                                    pass
                                    
                    # Log any generated signals
                    if signals:
                        print("\nRun Strategy: Signal Analysis:")
                        for i, signal in enumerate(signals):
                            direction = "+" if signal.direction == TradeDirection.LONG else "-"
                            print(f"  {direction} {signal.direction.name} signal at {signal.timestamp}")
                            print(f"    Entry: {signal.entry:.2f}, SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}")
                            logger.info(f"Generated {signal.direction.name} signal at {signal.timestamp}")
                            logger.info(f"  Entry: {signal.entry:.2f}, SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}")
                    else:
                        print("\nRun Strategy: No signals generated for this bar")
                        
                    # Log completed trades (if any trades were closed during this run)
                    if completed_trades:
                        print(f"\n📊 LIVE TRADING SUMMARY: {len(completed_trades)} trades closed this run")
                        for i, trade in enumerate(completed_trades, 1):
                            pnl = (trade.exit_price - trade.entry_price) if trade.direction == TradeDirection.LONG else (trade.entry_price - trade.exit_price)
                            pnl_pips = pnl / 0.01  # Convert to pips for XAUUSD
                            print(f"  {i}. {trade.direction.name} {trade.entry_price:.2f} → {trade.exit_price:.2f} ({pnl_pips:+.1f} pips) [{trade.exit_reason}]")
                            
            except Exception as e:
                logger.error(f"\nRun Strategy: Error in run_strategy: {str(e)}", exc_info=True)
                if not is_backtest:  # Only raise in live mode to prevent bot from crashing
                    raise
                
            return signals
    
if __name__ == "__main__":
    import MetaTrader5 as mt5
    import os
    import sys
    import time
    from datetime import datetime, timedelta

    try:
        print("=" * 80)
        print(">> XAUUSD STRATEGY RUNNER")
        print("=" * 80)

        # Create strategy instance
        print(">> Initializing strategy...")
        strategy = XAUUSDStrategy()
        
        # Print absolute file locations
        print("\n>> File Locations:")
        # Log file path is defined at the module level
        print(f"   Strategy Log: {os.path.abspath(log_file)}")
        # Config file path from strategy's config object
        config_file = strategy.config.filename if hasattr(strategy.config, 'filename') else 'config.ini'
        print(f"   Config File: {os.path.abspath(config_file)}")
        # Database file path from config or default location
        try:
            db_path = strategy.config.get('Database', 'database_path')
            if db_path is None:
                # Use current working directory for default path
                db_path = os.path.join(os.getcwd(), 'instance/mt5_data.db')  # Default path
        except (KeyError, AttributeError):
            # Use current working directory for default path
            db_path = os.path.join(os.getcwd(), 'instance/mt5_data.db')  # Default path
        print(f"   Database: {os.path.abspath(db_path)}")

        # Check trading mode (fallback to 'testing' if not set)
        trading_mode = getattr(strategy, '_trading_mode').lower()
        print(f"\n>> Trading Mode: {trading_mode.upper()}")

        if trading_mode == 'live':
            print("\n>> RUNNING IN LIVE TRADING MODE")
            # print("=" * 50)
            # print(f"   Symbol: {strategy.symbol}")
            # print(f"   Data Timeframe: {strategy.data_timeframe}")
            # print(f"   Execution Timeframe: {strategy.execution_timeframe}")
            # print(f"   Max Open Trades: {getattr(strategy, '_max_open_trades', 'N/A')}")
            # print(f"   Risk per Trade: {getattr(strategy, '_risk_per_trade_pct', 'N/A')}%\n")

            # Get LiveTrader config values
            live_cfg = strategy.config
            # max_retries = int(live_cfg.get('LiveTrader', 'max_retries', fallback=5))
            # retry_delay = int(live_cfg.get('LiveTrader', 'retry_delay', fallback=3))
            check_interval = int(live_cfg.get('LiveTrader', 'check_interval'))

            # Restore open positions from MT5
            strategy.restore_open_positions()
            #print(f"[DEBUG] Open positions after restore: {strategy._open_positions}")
            
            if len(strategy._open_positions) > strategy._max_open_trades:
                print("\n>> Warning: More open trades than allowed!")
                
            # Main live trading loop
            try:
                while True:
                    print("\n>> Fetching latest market data...")
                    df = strategy.get_mt5_data(
                        symbol=strategy.symbol,
                        timeframe=strategy.data_timeframe,
                        from_date=datetime.now() - timedelta(days=1),
                        to_date=datetime.now()
                    )
                    if df is None or df.empty:
                        print("\n>> Error: Could not fetch market data")
                        time.sleep(check_interval)
                        continue
                    print(f"\n>> Data fetched successfully: {len(df)} bars")
                    
                    # Run live trading logic: check for trade signals and execute trades if pass percent is met                    
                    signals = strategy.run_strategy(symbol=strategy.symbol, timeframe=strategy.data_timeframe)
                    if signals:
                        print(f"\n>> Found {len(signals)} signals in recent data")
                        for signal in signals:
                            print(f"    Signal: {signal.direction} at {signal.timestamp}")
                            print(f"    Entry: {signal.entry} SL: {signal.stop_loss} TP: {signal.take_profit}")
                            print(f"    Size: {signal.position_size}")                            
                    time.sleep(check_interval)                    
            except KeyboardInterrupt:
                print("\nLive trading stopped by user")
            except Exception as e:
                print(f"\nError in live trading mode: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            # Testing/backtest mode
            print("\n>> RUNNING IN TESTING/BACKTEST MODE")
            print("=" * 50)
            start_date = strategy.from_date
            end_date = strategy.to_date
            if isinstance(start_date, str):
                try:
                    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                try:
                    end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    end_date = datetime.strptime(end_date, '%Y-%m-%d')
            print(f"\n>> Backtest Date Range: {start_date} to {end_date}")
            print(f"\n>> Fetching historical data for {strategy.symbol} {strategy.data_timeframe}...")            
            df = strategy.load_historical_data(
                symbol=strategy.symbol,
                timeframe=strategy.data_timeframe,
                from_date=start_date,
                to_date=end_date
            )
            if df is None or df.empty:
                print("\n>> [ERROR] No data returned for backtesting!")
                exit(1)
            print(f"\n>> [SUCCESS] Fetched {len(df)} bars of historical data")
            
            print("\n>> Running backtest...")
            signals = strategy.run_strategy(backtest_data=df)
            if not signals:
                print("\n>> [WARNING] No trading signals generated during backtest")
            else:
                print(f"\n>> Backtest Results:")
                print(f"   Total Signals: {len(signals)}")
                # Print a few sample trades if available
                print(f"\n>> Sample Signals (first 5):")
                for i, signal in enumerate(signals[:5], 1):
                    direction = getattr(signal, 'direction', None)
                    direction_str = direction.name if hasattr(direction, 'name') else str(direction)
                    #print(f"  {i}. {direction_str} @ {getattr(signal, 'timestamp', 'N/A')} | Entry: {getattr(signal, 'entry', 'N/A')} | SL: {getattr(signal, 'stop_loss', 'N/A')} | TP: {getattr(signal, 'take_profit', 'N/A')} | Size: {getattr(signal, 'position_size', 'N/A')}")
                    print(f"  {i}. {direction_str} @ {getattr(signal, 'timestamp', 'N/A')} | Entry: {getattr(signal, 'entry_price', 'N/A')} | SL: {getattr(signal, 'stop_loss', 'N/A')} | TP: {getattr(signal, 'take_profit', 'N/A')} | Size: {getattr(signal, 'position_size', 'N/A')}")
                    if hasattr(signal, 'exit_time') and signal.exit_time:
                        print(f"     Exit: {signal.exit_time} | Price: {getattr(signal, 'exit_price', 'N/A')} | Reason: {getattr(signal, 'exit_reason', '')}")
            print("\n>> Generating enhanced HTML report...")
            try:
                report_path = strategy.generate_html_report(df, signals) if df is not None else None
                if report_path:
                    print(f"\n>> Enhanced HTML report saved to: {os.path.abspath(report_path)}")
                    try:
                        import webbrowser
                        webbrowser.open(f'file://{os.path.abspath(report_path)}')
                        print(f"\n>> Report opened in default web browser")
                    except Exception as e:
                        print(f"\n>> Could not open browser automatically: {e}")
                        print(f"\n>>   Please manually open: {os.path.abspath(report_path)}")
                else:
                    print("\n>> Could not generate HTML report: No data available")
            except Exception as e:
                print(f"\n>> Could not generate HTML report: {e}")
    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
    finally:
        if 'mt5' in globals() and mt5.terminal_info() is not None:
            mt5.shutdown()
        