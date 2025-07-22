#!/usr/bin/env python3
"""
Trade logging utility for the bot_lite system.
Handles logging trades to the appropriate table based on trading mode.
"""

import sqlite3
import os
from datetime import datetime, timezone
import pytz
from typing import Optional, Dict, Any
from config_manager import get_config

class TradeLogger:
    """Utility class for logging trades to the database."""
    
    def __init__(self):
        """Initialize the trade logger."""
        self.config = get_config()
        self.db_path = self.config.get('Database', 'path', 'instance/mt5_data.db')
        
        # Clean config values to remove comments
        strategy_version = self.config.get('Strategy', 'strategy_version', '1.0.0')
        if isinstance(strategy_version, str):
            strategy_version = strategy_version.split('#')[0].strip()
        self.strategy_version = strategy_version
        
        trading_mode = self.config.get('Strategy', 'trading_mode', 'testing')
        if isinstance(trading_mode, str):
            trading_mode = trading_mode.split('#')[0].strip()
        self.trading_mode = trading_mode
        
        # Get execution timezone from config
        execution_tz_str = self.config.get('Strategy', 'execution_timezone', 'UTC')
        try:
            self.execution_timezone = pytz.timezone(execution_tz_str)
            print(f"Trade logger using execution timezone: {execution_tz_str}")
        except Exception as e:
            print(f"Invalid execution timezone: {execution_tz_str}, error: {e}. Using UTC.")
            self.execution_timezone = pytz.UTC
        
    def log_trade_entry(self, trade_data: Dict[str, Any]) -> bool:
        """
        Log a trade entry to the appropriate table.
        
        Args:
            trade_data: Dictionary containing trade information
                Required keys:
                - timestamp: Trade entry timestamp
                - symbol: Trading symbol
                - direction: Trade direction (LONG/SHORT)
                - entry_price: Entry price
                - stop_loss: Stop loss price
                - take_profit: Take profit price
                - position_size: Position size in lots
                - magic_number: Magic number for the trade
                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine which table to use based on trading mode
            table_name = f"{self.trading_mode}_trades"
            
            # Ensure timestamp is in execution timezone
            timestamp = trade_data['timestamp']
            if isinstance(timestamp, str):
                try:
                    # Try to parse the timestamp string
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        # Try alternative format with milliseconds
                        timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        print(f"Warning: Could not parse timestamp {timestamp}, using current time")
                        timestamp = datetime.now()
            
            # If timestamp is naive (no timezone), assume it's in server timezone and convert to execution timezone
            if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
                # First make it timezone-aware by assuming it's in UTC
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                
            # Now convert to execution timezone
            if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(self.execution_timezone)
                formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                print(f"✅ Converted timestamp to execution timezone: {formatted_timestamp} {self.execution_timezone}")
            else:
                formatted_timestamp = timestamp
                print(f"⚠️ Using timestamp as-is (could not convert): {formatted_timestamp}")
            
            # Prepare the data for insertion
            insert_data = {
                'timestamp': formatted_timestamp,
                'symbol': trade_data['symbol'],
                'direction': trade_data['direction'],
                'entry_price': trade_data['entry_price'],
                'exit_price': None,  # Will be updated when trade closes
                'stop_loss': trade_data['stop_loss'],
                'take_profit': trade_data['take_profit'],
                'position_size': trade_data['position_size'],
                'profit_loss': None,  # Will be calculated when trade closes
                'commission': None,  # Will be calculated when trade closes
                'exit_reason': None,  # Will be set when trade closes
                'exit_time': None,  # Will be set when trade closes
                'magic_number': trade_data['magic_number'],
                'strategy_version': self.strategy_version,
                'trading_mode': self.trading_mode
            }
            
            # Connect to database and insert
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert the trade entry
            cursor.execute(f'''
                INSERT INTO {table_name} (
                    timestamp, symbol, direction, entry_price, exit_price,
                    stop_loss, take_profit, position_size, profit_loss,
                    commission, exit_reason, exit_time, magic_number,
                    strategy_version, trading_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                insert_data['timestamp'], insert_data['symbol'], insert_data['direction'],
                insert_data['entry_price'], insert_data['exit_price'], insert_data['stop_loss'],
                insert_data['take_profit'], insert_data['position_size'], insert_data['profit_loss'],
                insert_data['commission'], insert_data['exit_reason'], insert_data['exit_time'],
                insert_data['magic_number'], insert_data['strategy_version'], insert_data['trading_mode']
            ))
            
            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"✅ Logged trade entry: {trade_data['symbol']} {trade_data['direction']} at {trade_data['entry_price']} (ID: {trade_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error logging trade entry: {e}")
            return False
    
    def log_trade_exit(self, magic_number: int, exit_data: Dict[str, Any]) -> bool:
        """
        Log a trade exit by updating the existing trade record.
        
        Args:
            magic_number: Magic number of the trade to update
            exit_data: Dictionary containing exit information
                Required keys:
                - exit_price: Exit price
                - exit_time: Exit timestamp
                - exit_reason: Reason for exit
                - profit_loss: Profit/loss amount
                - commission: Commission amount
                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            table_name = f"{self.trading_mode}_trades"
            
            # Ensure exit_time is in execution timezone
            exit_time = exit_data['exit_time']
            if isinstance(exit_time, str):
                try:
                    # Try to parse the timestamp string
                    exit_time = datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        # Try alternative format with milliseconds
                        exit_time = datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        print(f"Warning: Could not parse exit_time {exit_time}, using current time")
                        exit_time = datetime.now()
            
            # If exit_time is naive (no timezone), assume it's in server timezone and convert to execution timezone
            if isinstance(exit_time, datetime) and exit_time.tzinfo is None:
                # First make it timezone-aware by assuming it's in UTC
                exit_time = exit_time.replace(tzinfo=timezone.utc)
                
            # Now convert to execution timezone
            if isinstance(exit_time, datetime) and exit_time.tzinfo is not None:
                exit_time = exit_time.astimezone(self.execution_timezone)
                formatted_exit_time = exit_time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"✅ Converted exit_time to execution timezone: {formatted_exit_time} {self.execution_timezone}")
            else:
                formatted_exit_time = exit_time
                print(f"⚠️ Using exit_time as-is (could not convert): {formatted_exit_time}")
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update the trade record
            cursor.execute(f"""
                UPDATE {table_name} SET
                    exit_price = ?,
                    exit_time = ?,
                    exit_reason = ?,
                    profit_loss = ?,
                    commission = ?
                WHERE magic_number = ? AND trading_mode = ?
            """, (
                exit_data['exit_price'],
                formatted_exit_time,
                exit_data['exit_reason'],
                exit_data['profit_loss'],
                exit_data['commission'],
                magic_number,
                self.trading_mode
            ))
            
            rows_updated = cursor.rowcount
            conn.commit()
            conn.close()
            
            if rows_updated > 0:
                print(f"✅ Updated trade exit: Magic {magic_number} - {exit_data['exit_reason']} at {exit_data['exit_price']}")
                return True
            else:
                print(f"⚠️ No trade found with magic number {magic_number} in {self.trading_mode} mode")
                return False
                
        except Exception as e:
            print(f"❌ Error logging trade exit: {e}")
            return False
    
    def get_trade_by_magic(self, magic_number: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a trade by its magic number.
        
        Args:
            magic_number: Magic number of the trade
            
        Returns:
            Dict containing trade data or None if not found
        """
        try:
            table_name = f"{self.trading_mode}_trades"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
                SELECT * FROM {table_name} 
                WHERE magic_number = ? AND trading_mode = ?
            ''', (magic_number, self.trading_mode))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Create dictionary from row data
                trade_data = dict(zip(columns, row))
                return trade_data
            
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving trade: {e}")
            return None
    
    def get_trades_by_date_range(self, from_date: str, to_date: str, symbol: str = None) -> list:
        """
        Get trades within a date range.
        
        Args:
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            symbol: Optional symbol filter
            
        Returns:
            List of trade dictionaries
        """
        try:
            table_name = f"{self.trading_mode}_trades"
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()
            
            # Convert date range to execution timezone format
            # Note: The database already stores dates in execution timezone format due to our modifications
            # But we need to ensure the query parameters are also in the correct format
            
            # Build query with parameters
            query = f"SELECT * FROM {table_name} WHERE timestamp BETWEEN ? AND ? AND trading_mode = ?"
            params = [f"{from_date} 00:00:00", f"{to_date} 23:59:59", self.trading_mode]
            
            # Add symbol filter if provided
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
                
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            trades = [dict(row) for row in rows]
            conn.close()
            
            return trades
            
        except Exception as e:
            print(f"❌ Error retrieving trades: {e}")
            return []
    
    def get_trade_statistics(self, from_date: str = None, to_date: str = None) -> Dict[str, Any]:
        """
        Get trade statistics for the current trading mode.
        
        Args:
            from_date: Optional start date filter
            to_date: Optional end date filter
            
        Returns:
            Dictionary containing trade statistics
        """
        try:
            table_name = f"{self.trading_mode}_trades"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build base query
            base_query = f"FROM {table_name} WHERE trading_mode = ?"
            params = [self.trading_mode]
            
            if from_date and to_date:
                base_query += " AND timestamp BETWEEN ? AND ?"
                params.extend([f"{from_date} 00:00:00", f"{to_date} 23:59:59"])
            
            # Get total trades
            cursor.execute(f"SELECT COUNT(*) {base_query}", params)
            total_trades = cursor.fetchone()[0]
            
            # Get completed trades (with exit_price)
            cursor.execute(f"SELECT COUNT(*) {base_query} AND exit_price IS NOT NULL", params)
            completed_trades = cursor.fetchone()[0]
            
            # Get winning trades
            cursor.execute(f"SELECT COUNT(*) {base_query} AND exit_price IS NOT NULL AND profit_loss > 0", params)
            winning_trades = cursor.fetchone()[0]
            
            # Get total profit/loss
            cursor.execute(f"SELECT COALESCE(SUM(profit_loss), 0) {base_query} AND exit_price IS NOT NULL", params)
            total_pl = cursor.fetchone()[0] or 0
            
            # Get total commission
            cursor.execute(f"SELECT COALESCE(SUM(commission), 0) {base_query} AND exit_price IS NOT NULL", params)
            total_commission = cursor.fetchone()[0] or 0
            
            # Calculate win rate
            win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
            
            conn.close()
            
            return {
                'total_trades': total_trades,
                'completed_trades': completed_trades,
                'open_trades': total_trades - completed_trades,
                'winning_trades': winning_trades,
                'losing_trades': completed_trades - winning_trades,
                'win_rate': round(win_rate, 2),
                'total_profit_loss': round(total_pl, 2),
                'total_commission': round(total_commission, 2),
                'net_profit_loss': round(total_pl - total_commission, 2),
                'trading_mode': self.trading_mode,
                'strategy_version': self.strategy_version
            }
            
        except Exception as e:
            print(f"❌ Error getting trade statistics: {e}")
            return {}

    def delete_trade_by_magic(self, magic_number: int) -> bool:
        """
        Delete a trade by its magic number in the current trading mode.
        """
        try:
            table_name = f"{self.trading_mode}_trades"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name} WHERE magic_number = ? AND trading_mode = ?", (magic_number, self.trading_mode))
            rows_deleted = cursor.rowcount
            conn.commit()
            conn.close()
            print(f"Deleted {rows_deleted} trade(s) with magic number {magic_number} in {self.trading_mode} mode.")
            return rows_deleted > 0
        except Exception as e:
            print(f"❌ Error deleting trade: {e}")
            return False

    def delete_all_trades(self) -> int:
        """
        Delete all trades in the current trading mode table.
        Returns the number of rows deleted.
        """
        try:
            table_name = f"{self.trading_mode}_trades"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name} WHERE trading_mode = ?", (self.trading_mode,))
            rows_deleted = cursor.rowcount
            conn.commit()
            conn.close()
            print(f"Deleted {rows_deleted} trade(s) in {self.trading_mode} mode.")
            return rows_deleted
        except Exception as e:
            print(f"❌ Error deleting all trades: {e}")
            return 0

# Global instance for easy access
trade_logger = TradeLogger() 