#!/usr/bin/env python3
"""
MT5 Data Updater

This script updates the SQLite database with the latest XAUUSD 1-minute OHLC data from MetaTrader 5.
It checks the last date in the database and only downloads new data to avoid duplicates.
"""

import os
import sqlite3
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
from config_manager import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('update_mt5_data.log')
    ]
)
logger = logging.getLogger(__name__)

class MT5DataUpdater:
    def __init__(self):
        self.config = get_config()
        self.db_path = self._get_database_path()
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.table_name = "ohlc_mt5_xauusd_1min"
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database if it doesn't exist
        self._init_database()
    
    def _get_database_path(self) -> str:
        """Get the database path from config.ini or use default."""
        db_path = self.config.get('Database', 'path', fallback='instance/mt5_data.db')
        
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
        
        return db_path
    
    def _init_database(self):
        """Initialize the database with required tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    timestamp DATETIME PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER
                )
            """)
            conn.commit()
    
    def get_last_timestamp(self) -> Optional[datetime]:
        """Get the most recent timestamp from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT MAX(timestamp) FROM {self.table_name}
            """)
            result = cursor.fetchone()[0]
            
            if result is None:
                return None
                
            # Convert string to datetime object
            return datetime.strptime(result, '%Y-%m-%d %H:%M:%S')
    
    def fetch_mt5_data(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from MetaTrader 5."""
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return None
        
        try:
            # Set timezone to UTC for consistency
            utc_tz = pytz.UTC
            start_date = start_date.astimezone(utc_tz)
            end_date = end_date.astimezone(utc_tz)
            
            logger.info(f"Fetching {self.symbol} data from {start_date} to {end_date}")
            
            # Fetch data from MT5
            rates = mt5.copy_rates_range(
                self.symbol,
                self.timeframe,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                logger.warning("No data received from MT5")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to match database schema
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            # Select only the columns we need
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from MT5: {str(e)}")
            return None
        finally:
            mt5.shutdown()
    
    def save_to_database(self, df: pd.DataFrame) -> int:
        """Save DataFrame to the database, ignoring duplicates."""
        if df.empty:
            return 0
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert timestamp to string for SQLite
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Insert data, ignoring duplicates
                rows = df.to_dict('records')
                cursor = conn.cursor()
                
                # Use INSERT OR IGNORE to skip duplicates
                cursor.executemany(f"""
                    INSERT OR IGNORE INTO {self.table_name} 
                    (timestamp, open, high, low, close, volume)
                    VALUES (:timestamp, :open, :high, :low, :close, :volume)
                """, rows)
                
                conn.commit()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            return 0
    
    def update_data(self):
        """Update the database with the latest data from MT5."""
        logger.info("Starting data update process")
        
        # Get the last timestamp from the database
        last_timestamp = self.get_last_timestamp()
        
        # If no data exists, start from 7 days ago
        if last_timestamp is None:
            start_date = datetime.now() - timedelta(days=7)
            logger.warning(f"No existing data found. Starting from {start_date}")
        else:
            # Start from the next minute after the last timestamp
            start_date = last_timestamp + timedelta(minutes=1)
            logger.info(f"Last timestamp in database: {last_timestamp}")
        
        # End at the start of today (00:00:00)
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # If start_date is after end_date, no update needed
        if start_date >= end_date:
            logger.info("Database is already up to date")
            return
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        # Fetch data from MT5
        df = self.fetch_mt5_data(start_date, end_date)
        
        if df is None or df.empty:
            logger.warning("No new data to update")
            return
        
        # Save to database
        rows_updated = self.save_to_database(df)
        logger.info(f"Successfully updated {rows_updated} rows in the database")

def main():
    try:
        updater = MT5DataUpdater()
        updater.update_data()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
