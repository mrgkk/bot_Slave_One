"""
Script to download and store 1-minute OHLC data from MT5 to SQLite database.
"""
import argparse
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from bot_lite.mt5_data_loader import MT5DataLoader
from bot_lite.config_manager import get_config

def download_data(symbol: str, days: int, update_existing: bool = False):
    """
    Download and store 1-minute OHLC data for the specified symbol and number of days.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        days: Number of days of historical data to download
        update_existing: If True, update existing data in the database
    """
    print(f"Starting data download for {symbol} (last {days} days)...")
    
    # Initialize data loader
    data_loader = MT5DataLoader()
    
    # Connect to MT5
    if not data_loader.connect_to_mt5():
        print("Failed to connect to MT5. Please check your configuration.")
        return
    
    try:
        # Download the data
        print(f"Downloading {days} days of 1-minute data for {symbol}...")
        df = data_loader.download_historical_data(
            symbol=symbol,
            timeframe='M1',
            days=days
        )
        
        if df is not None and not df.empty:
            print(f"Successfully downloaded {len(df)} bars of data")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Verify data was saved
            loaded_df = data_loader.load_from_database(symbol, 'M1')
            if not loaded_df.empty:
                print(f"Data successfully stored in database. Total records: {len(loaded_df)}")
                print("\nSample of stored data:")
                print(loaded_df.head())
            else:
                print("Warning: Data was downloaded but could not be verified in the database.")
        else:
            print("No data was downloaded. Please check your MT5 connection and symbol name.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Disconnect from MT5
        data_loader.disconnect_mt5()
        print("MT5 connection closed.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download 1-minute OHLC data from MT5')
    parser.add_argument('--symbol', type=str, default='XAUUSD',
                       help='Trading symbol (default: XAUUSD)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days of historical data to download (default: 7)')
    parser.add_argument('--update', action='store_true',
                       help='Update existing data in the database')
    
    args = parser.parse_args()
    
    # Download the data
    download_data(args.symbol, args.days, args.update)

if __name__ == "__main__":
    main()
