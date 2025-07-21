#!/usr/bin/env python3
"""
Database Viewer for Trading Bot

This script provides a detailed view of the SQLite database used by the trading bot.
It shows table information, schemas, and sample data.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Import config manager
from config_manager import get_config

def get_database_path() -> str:
    """Get the database path from config.ini."""
    config = get_config()
    db_path = config.get('Database', 'path', fallback='instance/mt5_data.db')
    
    # Convert to absolute path if it's a relative path
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
    
    return db_path

def get_table_info(conn) -> List[Dict[str, Any]]:
    """Get information about all tables in the database."""
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("""
        SELECT name 
        FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name;
    """)
    
    tables = []
    for (table_name,) in cursor.fetchall():
        # Get row count
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = cursor.fetchone()[0]
        
        # Get column info
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get date range if there's a timestamp column
        date_range = None
        if 'timestamp' in [col.lower() for col in columns]:
            try:
                cursor.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM "{table_name}"')
                min_ts, max_ts = cursor.fetchone()
                if min_ts and max_ts:
                    date_range = (min_ts, max_ts)
            except:
                pass
        
        tables.append({
            'name': table_name,
            'row_count': row_count,
            'columns': columns,
            'date_range': date_range
        })
    
    return tables

def get_sample_data(conn, table_name: str, limit: int = 5) -> Tuple[List[Dict], List[str]]:
    """Get sample data from a table."""
    cursor = conn.cursor()
    
    # Get column names
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    columns = [col[1] for col in cursor.fetchall()]
    
    # Get sample data
    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}')
    rows = cursor.fetchall()
    
    # Convert rows to list of dicts
    data = []
    for row in rows:
        data.append(dict(zip(columns, row)))
    
    return data, columns

def format_table_info(tables: List[Dict]) -> str:
    """Format table information as a string."""
    output = []
    output.append("\n=== Database Summary ===")
    output.append(f"Total Tables: {len(tables)}")
    
    for table in tables:
        output.append("\n" + "="*50)
        output.append(f"Table: {table['name']}")
        output.append("-" * 50)
        output.append(f"Rows: {table['row_count']:,}")
        
        if table['date_range']:
            min_ts, max_ts = table['date_range']
            output.append(f"Date Range: {min_ts}  to  {max_ts}")
        
        output.append("\nColumns:")
        for i, col in enumerate(table['columns'], 1):
            output.append(f"  {i}. {col}")
    
    return "\n".join(output)

def format_sample_data(table_name: str, data: List[Dict], columns: List[str]) -> str:
    """Format sample data as a string."""
    if not data:
        return f"\nNo data found in table '{table_name}'"
    
    output = []
    output.append(f"\nSample Data from '{table_name}':")
    
    # Format header
    header = " | ".join(f"{col:<15}" for col in columns)
    separator = "-" * len(header)
    output.append(separator)
    output.append(header)
    output.append(separator)
    
    # Format rows
    for row in data:
        row_str = []
        for col in columns:
            value = row.get(col, '')
            # Truncate long values
            if value is not None and len(str(value)) > 15:
                value = str(value)[:12] + "..."
            row_str.append(f"{str(value or ''):<15}")
        output.append(" | ".join(row_str))
    
    output.append(separator)
    return "\n".join(output)

def main():
    """Main function to display database information."""
    try:
        # Get database path
        db_path = get_database_path()
        print(f"\nConnecting to database: {db_path}")
        
        if not os.path.exists(db_path):
            print(f"\nError: Database file not found at: {db_path}")
            return
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        
        try:
            # Get and display table information
            tables = get_table_info(conn)
            print(format_table_info(tables))
            
            # Show sample data for each table
            for table in tables:
                sample_data, columns = get_sample_data(conn, table['name'])
                print(format_sample_data(table['name'], sample_data, columns))
                
        finally:
            conn.close()
            
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    print("\nDatabase inspection complete.")

if __name__ == "__main__":
    main()
