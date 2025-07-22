#!/usr/bin/env python3
"""
Script to create trade logging tables in the database.
Creates two tables:
1. testing_trades - for trades executed in testing mode
2. live_trades - for trades executed in live mode
"""

import sqlite3
import os
from config_manager import get_config

def create_trade_tables():
    """Create the trade logging tables in the database."""
    
    # Get database path from config
    config = get_config()
    db_path = config.get('Database', 'path', 'instance/mt5_data.db')
    
    # Ensure the instance directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create testing_trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS testing_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL NOT NULL,
                profit_loss REAL,
                commission REAL,
                exit_reason TEXT,
                exit_time TEXT,
                magic_number INTEGER,
                strategy_version TEXT NOT NULL,
                trading_mode TEXT DEFAULT 'testing',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create live_trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL NOT NULL,
                profit_loss REAL,
                commission REAL,
                exit_reason TEXT,
                exit_time TEXT,
                magic_number INTEGER,
                strategy_version TEXT NOT NULL,
                trading_mode TEXT DEFAULT 'live',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_testing_trades_timestamp ON testing_trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_testing_trades_symbol ON testing_trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_testing_trades_direction ON testing_trades(direction)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_testing_trades_magic_number ON testing_trades(magic_number)')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trades_timestamp ON live_trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trades_symbol ON live_trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trades_direction ON live_trades(direction)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_live_trades_magic_number ON live_trades(magic_number)')
        
        conn.commit()
        print(f"✅ Successfully created trade logging tables in {db_path}")
        print("📋 Tables created:")
        print("   - testing_trades (for testing mode trades)")
        print("   - live_trades (for live mode trades)")
        print("📊 Indexes created for better query performance")
        
        # Show table structure
        print("\n📋 Table Structure:")
        for table_name in ['testing_trades', 'live_trades']:
            print(f"\n{table_name}:")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"   - {col[1]} ({col[2]})")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()
    
    return True

def verify_tables():
    """Verify that the tables were created successfully."""
    
    config = get_config()
    db_path = config.get('Database', 'path', 'instance/mt5_data.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('testing_trades', 'live_trades')")
        tables = cursor.fetchall()
        
        if len(tables) == 2:
            print("✅ Both trade tables exist:")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"   - {table[0]}: {count} rows")
        else:
            print("❌ Some trade tables are missing")
            print(f"Found tables: {[t[0] for t in tables]}")
            
    except Exception as e:
        print(f"❌ Error verifying tables: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🚀 Creating trade logging tables...")
    success = create_trade_tables()
    
    if success:
        print("\n🔍 Verifying table creation...")
        verify_tables()
        print("\n✅ Trade logging tables setup complete!")
    else:
        print("\n❌ Failed to create trade logging tables") 