#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MT5 OHLC Data Fix Script
This script identifies and fixes issues in the MT5 OHLC database:
1. Removes duplicate timestamps
2. Fixes price anomalies (high < low, close outside high-low range, etc.)
3. Fills small gaps in time series (optional)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pytz
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mt5_data_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'mt5_data.db')
TABLE_NAME = 'ohlc_mt5_xauusd_1min'
BACKUP_TABLE = f"{TABLE_NAME}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def display_mt5_account_info():
    """Connect to MT5 and display enhanced account information."""
    try:
        # Import MT5ConnectionManager here to avoid circular imports
        from bot_lite.mt5_manager import MT5ConnectionManager
        from bot_lite.config_manager import get_config
        
        # Get config and initialize MT5ConnectionManager
        config = get_config()
        mt5_manager = MT5ConnectionManager(config)
        
        # Initialize MT5 connection
        if not mt5_manager.initialize():
            logger.error("MT5 initialization failed using MT5ConnectionManager")
            return False
        
        # Get account info using the manager
        account_info = mt5_manager.get_account_info()
        if account_info is None:
            logger.error("Failed to get account info using MT5ConnectionManager")
            mt5_manager.shutdown()
            return False
        
        # Log enhanced account information
        logger.info("="*50)
        logger.info("MT5 Account Information:")
        logger.info("-"*50)
        logger.info(f"Account Number: {mt5.account_info().login}")
        logger.info(f"Account Name: {account_info['name']}")
        logger.info(f"Server: {account_info['server']}")
        logger.info(f"Company: {account_info['company']}")
        logger.info(f"Currency: {account_info['currency']}")
        logger.info("-"*50)
        logger.info(f"Balance: {account_info['balance']:.2f}")
        logger.info(f"Equity: {account_info['equity']:.2f}")
        logger.info(f"Margin: {account_info['margin']:.2f}")
        logger.info(f"Free Margin: {account_info['free_margin']:.2f}")
        logger.info(f"Leverage: {account_info['leverage']}:1")
        logger.info("="*50)
        
        # Shutdown MT5 connection
        mt5_manager.shutdown()
        return True
    except Exception as e:
        logger.error(f"Error getting MT5 account info: {e}")
        # Try direct shutdown in case of error
        if mt5.terminal_info() is not None:
            mt5.shutdown()
        return False

def connect_to_db():
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        logger.info(f"Successfully connected to database: {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def backup_table(conn):
    """Create a backup of the original table."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE {BACKUP_TABLE} AS SELECT * FROM {TABLE_NAME}")
        conn.commit()
        logger.info(f"Created backup table: {BACKUP_TABLE}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error creating backup table: {e}")
        return False

def load_ohlc_data(conn):
    """Load OHLC data from the database into a pandas DataFrame."""
    try:
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql_query(query, conn)
        logger.info(f"Loaded {len(df)} rows from {TABLE_NAME}")
        return df
    except Exception as e:
        logger.error(f"Error loading OHLC data: {e}")
        return None

def identify_issues(df):
    """Identify issues in the OHLC data."""
    issues = {
        'duplicates': 0,
        'price_anomalies': 0,
        'gaps': 0
    }
    
    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Check for duplicate timestamps
    duplicates = df['timestamp'].duplicated()
    issues['duplicates'] = duplicates.sum()
    if issues['duplicates'] > 0:
        logger.info(f"Found {issues['duplicates']} duplicate timestamps")
    
    # Check for price anomalies
    high_lt_low = (df['high'] < df['low']) & (~df['high'].isnull()) & (~df['low'].isnull())
    close_gt_high = (df['close'] > df['high']) & (~df['close'].isnull()) & (~df['high'].isnull())
    close_lt_low = (df['close'] < df['low']) & (~df['close'].isnull()) & (~df['low'].isnull())
    open_gt_high = (df['open'] > df['high']) & (~df['open'].isnull()) & (~df['high'].isnull())
    open_lt_low = (df['open'] < df['low']) & (~df['open'].isnull()) & (~df['low'].isnull())
    
    price_anomalies = high_lt_low | close_gt_high | close_lt_low | open_gt_high | open_lt_low
    issues['price_anomalies'] = price_anomalies.sum()
    if issues['price_anomalies'] > 0:
        logger.info(f"Found {issues['price_anomalies']} price anomalies")
    
    # Check for gaps in time series
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff().dropna()
    expected_diff = pd.Timedelta(minutes=1)
    gaps = time_diffs[time_diffs > expected_diff]
    issues['gaps'] = len(gaps)
    if issues['gaps'] > 0:
        logger.info(f"Found {issues['gaps']} time gaps")
        
        # Log the largest gaps
        largest_gaps = gaps.nlargest(5)
        for i in largest_gaps.index:
            gap_start = df_sorted.loc[i-1, 'timestamp']
            gap_end = df_sorted.loc[i, 'timestamp']
            gap_duration = (gap_end - gap_start).total_seconds() / 60  # in minutes
            logger.info(f"Gap from {gap_start} to {gap_end} ({gap_duration:.1f} minutes)")
    
    return issues, df, duplicates, price_anomalies

def fix_duplicates(df, duplicates):
    """Fix duplicate timestamps by keeping the first occurrence."""
    if duplicates.sum() > 0:
        logger.info("Removing duplicate timestamps...")
        df_fixed = df[~df['timestamp'].duplicated()]
        logger.info(f"Removed {len(df) - len(df_fixed)} duplicate rows")
        return df_fixed
    return df

def fix_price_anomalies(df, price_anomalies):
    """Fix price anomalies by adjusting high, low, open, close values."""
    if price_anomalies.sum() > 0:
        logger.info("Fixing price anomalies...")
        df_fixed = df.copy()
        
        # Get indices of rows with price anomalies
        anomaly_indices = df[price_anomalies].index
        
        for idx in anomaly_indices:
            row = df_fixed.loc[idx]
            
            # Fix high < low
            if row['high'] < row['low']:
                # Swap high and low
                df_fixed.loc[idx, 'high'], df_fixed.loc[idx, 'low'] = row['low'], row['high']
                logger.debug(f"Fixed high < low at {row['timestamp']}")
            
            # Fix close outside high-low range
            if row['close'] > df_fixed.loc[idx, 'high']:
                df_fixed.loc[idx, 'high'] = row['close']
                logger.debug(f"Fixed close > high at {row['timestamp']}")
            elif row['close'] < df_fixed.loc[idx, 'low']:
                df_fixed.loc[idx, 'low'] = row['close']
                logger.debug(f"Fixed close < low at {row['timestamp']}")
            
            # Fix open outside high-low range
            if row['open'] > df_fixed.loc[idx, 'high']:
                df_fixed.loc[idx, 'high'] = row['open']
                logger.debug(f"Fixed open > high at {row['timestamp']}")
            elif row['open'] < df_fixed.loc[idx, 'low']:
                df_fixed.loc[idx, 'low'] = row['open']
                logger.debug(f"Fixed open < low at {row['timestamp']}")
        
        logger.info(f"Fixed {len(anomaly_indices)} price anomalies")
        return df_fixed
    return df

def fill_small_gaps(df, max_gap_minutes=5, fill_weekend_gaps=False):
    """Fill small gaps in the time series with interpolated values.
    
    Args:
        df: DataFrame with timestamp column
        max_gap_minutes: Maximum gap size to fill in minutes
        fill_weekend_gaps: If True, will also fill gaps during weekends and holidays
                          using the last known values
    """
    logger.info(f"Filling gaps smaller than {max_gap_minutes} minutes...")
    
    # Ensure timestamp column is datetime and sorted
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Find gaps
    time_diffs = df_sorted['timestamp'].diff().dropna()
    expected_diff = pd.Timedelta(minutes=1)
    gaps = time_diffs[time_diffs > expected_diff]
    
    if len(gaps) == 0:
        logger.info("No gaps to fill")
        return df
    
    # Process gaps in batches to avoid memory issues
    filled_count = 0
    new_rows = []
    skipped_gaps = 0
    weekend_gaps_filled = 0
    
    # Helper function to check if a gap is during trading hours
    def is_trading_hours_gap(start_time, end_time):
        # Check if the gap crosses a weekend
        start_day = start_time.weekday()
        end_day = end_time.weekday()
        
        # Friday evening to Monday morning is a weekend gap
        if start_day == 4 and end_day == 0:  # Friday to Monday
            return False
        
        # Check for typical forex market hours (24/5)
        # Forex market typically closes Friday ~22:00 and reopens Sunday ~22:00
        if start_day == 4 and start_time.hour >= 22:  # Friday after 22:00
            return False
        if end_day == 0 and end_time.hour <= 22:  # Monday before 22:00
            return False
            
        return True
    
    # Process each gap
    for i in gaps.index:
        prev_time = df_sorted.loc[i-1, 'timestamp']
        curr_time = df_sorted.loc[i, 'timestamp']
        gap_minutes = (curr_time - prev_time).total_seconds() / 60
        
        # Determine if this is a trading hours gap or weekend/holiday gap
        is_trading_gap = is_trading_hours_gap(prev_time, curr_time)
        
        # Skip large gaps during trading hours or weekend gaps if not filling them
        if (is_trading_gap and gap_minutes > max_gap_minutes) or \
           (not is_trading_gap and not fill_weekend_gaps):
            logger.debug(f"Skipping gap of {gap_minutes:.1f} minutes from {prev_time} to {curr_time}")
            skipped_gaps += 1
            continue
        
        # Generate timestamps for the gap
        gap_times = pd.date_range(start=prev_time + pd.Timedelta(minutes=1), 
                                 end=curr_time - pd.Timedelta(minutes=1),
                                 freq='1min')
        
        if len(gap_times) == 0:
            continue
        
        # Get values to interpolate between
        prev_row = df_sorted.loc[i-1]
        curr_row = df_sorted.loc[i]
        
        # Different filling strategies for trading hours vs weekend gaps
        if is_trading_gap:
            # Linear interpolation for trading hour gaps
            for j, ts in enumerate(gap_times):
                # Calculate interpolation factor (0 to 1)
                total_steps = len(gap_times) + 1
                factor = (j + 1) / total_steps
                
                new_row = {'timestamp': ts}
                
                # Interpolate numeric columns
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_sorted.columns:
                        new_row[col] = prev_row[col] + (curr_row[col] - prev_row[col]) * factor
                        # Round to appropriate precision
                        new_row[col] = round(new_row[col], 2)
                
                # Handle volume (use average or nearest)
                if 'volume' in df_sorted.columns:
                    new_row['volume'] = int((prev_row['volume'] + curr_row['volume']) / 2)
                
                new_rows.append(new_row)
            
            filled_count += len(gap_times)
        elif fill_weekend_gaps:
            # For weekend/holiday gaps, use last known values (flat filling)
            for ts in gap_times:
                new_row = {'timestamp': ts}
                
                # Use closing values for all weekend candles
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_sorted.columns:
                        new_row[col] = prev_row['close']
                
                # Set volume to 0 for weekend/holiday
                if 'volume' in df_sorted.columns:
                    new_row['volume'] = 0
                
                new_rows.append(new_row)
            
            weekend_gaps_filled += 1
            filled_count += len(gap_times)
    
    # Add the new rows to the dataframe
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df_sorted, df_new], ignore_index=True)
        df_result = df_combined.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Filled {filled_count} missing values in {len(new_rows)} gaps")
        if weekend_gaps_filled > 0:
            logger.info(f"Filled {weekend_gaps_filled} weekend/holiday gaps")
        return df_result
    else:
        logger.info(f"No gaps to fill. Skipped {skipped_gaps} gaps.")
        return df_sorted

def save_fixed_data(conn, df):
    """Save the fixed data back to the database."""
    try:
        # Drop the original table
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE {TABLE_NAME}")
        conn.commit()
        
        # Create a new table with the fixed data
        df.to_sql(TABLE_NAME, conn, index=False)
        conn.commit()
        
        logger.info(f"Successfully saved {len(df)} rows to {TABLE_NAME}")
        return True
    except Exception as e:
        logger.error(f"Error saving fixed data: {e}")
        return False

def main():
    """Main function to fix the OHLC data."""
    parser = argparse.ArgumentParser(description='Fix issues in MT5 OHLC data')
    parser.add_argument('--fill-gaps', action='store_true', help='Fill small gaps in time series')
    parser.add_argument('--max-gap', type=int, default=5, help='Maximum gap size to fill (in minutes)')
    parser.add_argument('--fill-weekends', action='store_true', help='Fill weekend gaps with flat values')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating a backup table')
    parser.add_argument('--account-info', action='store_true', help='Include MT5 account information in logs')
    args = parser.parse_args()
    
    logger.info("Starting MT5 OHLC data fix...")
    
    # Display MT5 account information if requested
    if args.account_info:
        display_mt5_account_info()
    
    # Connect to the database
    conn = connect_to_db()
    if not conn:
        logger.error("Failed to connect to the database. Exiting.")
        return
    
    # Create a backup of the original table
    if not args.no_backup:
        if not backup_table(conn):
            logger.error("Failed to create backup table. Exiting.")
            conn.close()
            return
    
    # Load OHLC data
    df = load_ohlc_data(conn)
    if df is None or df.empty:
        logger.error("Failed to load OHLC data. Exiting.")
        conn.close()
        return
    
    # Identify issues
    issues, df, duplicates, price_anomalies = identify_issues(df)
    
    # Fix issues
    if sum(issues.values()) > 0:
        # Fix duplicates
        if issues['duplicates'] > 0:
            df = fix_duplicates(df, duplicates)
        
        # Fix price anomalies
        if issues['price_anomalies'] > 0:
            df = fix_price_anomalies(df, price_anomalies)
        
        # Fill gaps
        if issues['gaps'] > 0 and args.fill_gaps:
            df = fill_small_gaps(df, args.max_gap, args.fill_weekends)
        
        # Save fixed data
        if save_fixed_data(conn, df):
            logger.info("Successfully fixed and saved OHLC data")
        else:
            logger.error("Failed to save fixed data")
    else:
        logger.info("No issues found in the OHLC data")
    
    # Close the database connection
    conn.close()
    logger.info("OHLC data fix completed")

if __name__ == "__main__":
    main()
