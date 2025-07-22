import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import pytz
import os

# Import config manager
from config_manager import get_config

# Get config instance
config = get_config()

class MT5DataLoader:
    def __init__(self, db_path: str = None):
        """
        Initialize the MT5 data loader with a SQLite database.
        
        Args:
            db_path: Optional path to the SQLite database file. 
                    If not provided, uses 'instance/mt5_data.db' relative to the config file.
        """
        self.db_path = db_path or config.get('Database', 'path', 'instance/mt5_data.db')
        self._initialize_database()
        
        # Print configuration for debugging
        print(f"MT5 Data Loader initialized with:")
        print(f"- Database path: {self.db_path}")
        mt5_path = config.get('MT5', 'executable_path', 'Not configured')
        print(f"- MT5 executable path: {mt5_path if mt5_path else 'Default (from PATH)'}")
        
    def _initialize_database(self):
        """Initialize the SQLite database with the required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlc_mt5_xauusd_1min (
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            ''')
            conn.commit()
    
    def connect_to_mt5(self):
        """
        Connect to MetaTrader 5 terminal using the configured executable path.
        If no path is configured, MT5's default initialization is used.
        """
        mt5_path = config.get('MT5', 'executable_path', '').strip()
        
        try:
            if mt5_path and os.path.exists(mt5_path):
                print(f"Initializing MT5 with path: {mt5_path}")
                if not mt5.initialize(path=mt5_path):
                    print(f"MT5 initialization with path failed, trying default initialization...")
                    if not mt5.initialize():
                        print("MT5 default initialization failed")
                        return False
            else:
                if not mt5.initialize():
                    print("MT5 default initialization failed")
                    return False
                    
            print(f"MT5 initialized successfully. Terminal info: {mt5.terminal_info()}")
            return True
            
        except Exception as e:
            print(f"Error initializing MT5: {e}")
            return False
    
    def disconnect_mt5(self):
        """Disconnect from MetaTrader 5 terminal."""
        mt5.shutdown()
    
    def download_historical_data(self, symbol: str, timeframe: str, days: int = 7):
        """
        Download historical data from MT5 and save to the database.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Timeframe ('M1', 'M5', 'H1', etc.)
            days: Number of days of historical data to download
        """
        if not self.connect_to_mt5():
            return None
        
        # Map timeframe string to MT5 constant
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        if timeframe not in tf_map:
            print(f"Unsupported timeframe: {timeframe}")
            return None
        
        # Calculate date range
        end_date = datetime.now(pytz.utc)
        start_date = end_date - timedelta(days=days)
        
        print(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}")
        
        # Request historical data
        rates = mt5.copy_rates_range(
            symbol,
            tf_map[timeframe],
            start_date,
            end_date
        )
        
        if rates is None or len(rates) == 0:
            print(f"No data received for {symbol} {timeframe}")
            return None
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })
        
        # Add symbol and timeframe columns
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        # Reorder columns
        df = df[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Save to database
        self._save_to_database(df, 'ohlc_mt5_xauusd_1min')
        
        self.disconnect_mt5()
        return df
    
    def _save_to_database(self, df: pd.DataFrame, table_name: str):
        """Save DataFrame to SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            conn.commit()
    
    def load_from_database(self, from_date=None, to_date=None, limit: int = None) -> pd.DataFrame:
        """
        Load data from the ohlc_mt5_xauusd_1min table for a specific date range.
        Args:
            from_date: Start date (inclusive, as string or datetime)
            to_date: End date (inclusive, as string or datetime)
            limit: Maximum number of rows to return (optional)
        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlc_mt5_xauusd_1min
            WHERE 1=1
        """
        params = []
        if from_date:
            query += " AND timestamp >= ?"
            params.append(str(from_date))
        if to_date:
            query += " AND timestamp <= ?"
            params.append(str(to_date))
        query += " ORDER BY timestamp"
        if limit is not None:
            query += f" LIMIT {limit}"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=tuple(params))
        # Standardize all column names to title case and fix common typos
        column_map = {
           'open': 'open',
           'high': 'high',
           'low': 'low',
           'close': 'close',
           'tick_volume': 'tick_volume',
           'volume': 'volume'
            # Add more as needed
        }
        df.rename(columns=column_map, inplace=True)
        df.columns = [col.lower() for col in df.columns]
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        return df

def main():
    # Initialize the data loader
    data_loader = MT5DataLoader()
    
    # Download 7 days of 1-minute XAUUSD data
    symbol = 'XAUUSD'
    timeframe = 'M1'
    days = 60
    
    print(f"Downloading {days} days of {symbol} {timeframe} data...")
    df = data_loader.download_historical_data(symbol, timeframe, days)
    
    if df is not None:
        print(f"Downloaded {len(df)} bars of {symbol} {timeframe} data")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Verify data was saved by loading it back
        loaded_df = data_loader.load_from_database(symbol, timeframe)
        print(f"Loaded {len(loaded_df)} bars from database")
        print("\nSample data:")
        print(loaded_df.head())
    else:
        print("Failed to download data. Please check your MT5 connection and symbol name.")

if __name__ == "__main__":
    main()