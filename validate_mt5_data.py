#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MT5 OHLC Data Validation Script
This script validates the OHLC data in the MT5 database and generates an HTML report.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import jinja2
import base64
from io import BytesIO
import argparse
import webbrowser

# Configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'mt5_data.db')
TABLE_NAME = 'ohlc_mt5_xauusd_1min'
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mt5_db_validation_report.html')

def connect_to_db():
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        print(f"Successfully connected to database: {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def get_table_schema(conn):
    """Get the schema of the OHLC table."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
        schema = cursor.fetchall()
        return schema
    except sqlite3.Error as e:
        print(f"Error getting table schema: {e}")
        return None

def load_ohlc_data(conn):
    """Load OHLC data from the database into a pandas DataFrame."""
    try:
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql_query(query, conn)
        print(f"Loaded {len(df)} rows from {TABLE_NAME}")
        return df
    except Exception as e:
        print(f"Error loading OHLC data: {e}")
        return None

def validate_data(df):
    """Validate the OHLC data and return validation results."""
    results = {}
    
    # Check for basic data presence
    results['total_rows'] = len(df)
    
    # Check for null values
    null_counts = df.isnull().sum()
    results['null_values'] = null_counts.to_dict()
    
    # Check for duplicate timestamps
    if 'timestamp' in df.columns:
        duplicates = df['timestamp'].duplicated().sum()
        results['duplicate_timestamps'] = duplicates
    
    # Check for price anomalies (if high < low, etc.)
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        anomalies = {
            'high_lt_low': ((df['high'] < df['low']) & (~df['high'].isnull()) & (~df['low'].isnull())).sum(),
            'close_gt_high': ((df['close'] > df['high']) & (~df['close'].isnull()) & (~df['high'].isnull())).sum(),
            'close_lt_low': ((df['close'] < df['low']) & (~df['close'].isnull()) & (~df['low'].isnull())).sum(),
            'open_gt_high': ((df['open'] > df['high']) & (~df['open'].isnull()) & (~df['high'].isnull())).sum(),
            'open_lt_low': ((df['open'] < df['low']) & (~df['open'].isnull()) & (~df['low'].isnull())).sum(),
        }
        results['price_anomalies'] = anomalies
    
    # Check for gaps in time series (assuming 1-minute data)
    if 'timestamp' in df.columns:
        try:
            # Convert timestamp column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Calculate time differences
            time_diffs = df['timestamp'].diff().dropna()
            
            # Expected difference for 1-minute data
            expected_diff = pd.Timedelta(minutes=1)
            
            # Find gaps (time differences > expected)
            gaps = time_diffs[time_diffs > expected_diff]
            
            results['time_gaps'] = {
                'total_gaps': len(gaps),
                'max_gap': gaps.max().total_seconds() / 60 if len(gaps) > 0 else 0,  # in minutes
                'avg_gap': gaps.mean().total_seconds() / 60 if len(gaps) > 0 else 0,  # in minutes
            }
            
            # Store the top 10 largest gaps for the report
            if len(gaps) > 0:
                largest_gaps = []
                for i in gaps.nlargest(10).index:
                    gap_start = df.loc[i-1, 'timestamp']
                    gap_end = df.loc[i, 'timestamp']
                    gap_duration = (gap_end - gap_start).total_seconds() / 60  # in minutes
                    largest_gaps.append({
                        'start': gap_start.strftime('%Y-%m-%d %H:%M:%S'),
                        'end': gap_end.strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_minutes': gap_duration
                    })
                results['largest_gaps'] = largest_gaps
        except Exception as e:
            print(f"Error analyzing time gaps: {e}")
            results['time_gaps'] = {'error': str(e)}
    
    # Check for extreme price movements
    if all(col in df.columns for col in ['high', 'low']):
        # Calculate price range as percentage of price
        df['price_range_pct'] = (df['high'] - df['low']) / df['low'] * 100
        
        # Identify extreme movements (e.g., > 1% for XAUUSD in 1 minute is unusual)
        extreme_threshold = 1.0  # 1% threshold for 1-minute gold data
        extreme_moves = df[df['price_range_pct'] > extreme_threshold]
        
        results['extreme_price_movements'] = {
            'total_extreme_moves': len(extreme_moves),
            'max_range_pct': df['price_range_pct'].max(),
            'avg_range_pct': df['price_range_pct'].mean(),
        }
        
        # Store the top 10 most extreme movements for the report
        if len(extreme_moves) > 0:
            top_extreme = df.nlargest(10, 'price_range_pct')
            extreme_list = []
            for _, row in top_extreme.iterrows():
                extreme_list.append({
                    'time': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.api.types.is_datetime64_any_dtype(row['timestamp']) else str(row['timestamp']),
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'range_pct': row['price_range_pct']
                })
            results['top_extreme_movements'] = extreme_list
    
    return results

def generate_plots(df):
    """Generate plots for the report."""
    plots = {}
    
    # Ensure timestamp column is datetime
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Price history plot
    if all(col in df.columns for col in ['timestamp', 'close']):
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(df['timestamp'], df['close'])
            plt.title('XAUUSD Close Price History')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.grid(True)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots['price_history'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            print(f"Error generating price history plot: {e}")
    
    # 2. Daily data count
    if 'timestamp' in df.columns:
        try:
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby('date').size()
            
            plt.figure(figsize=(10, 6))
            daily_counts.plot(kind='bar')
            plt.title('Daily Data Count')
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.grid(True)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots['daily_count'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            print(f"Error generating daily count plot: {e}")
    
    # 3. Price range distribution
    if 'price_range_pct' in df.columns:
        try:
            plt.figure(figsize=(10, 6))
            df['price_range_pct'].hist(bins=50)
            plt.title('Price Range Distribution (% of price)')
            plt.xlabel('Price Range (%)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots['price_range_dist'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            print(f"Error generating price range distribution plot: {e}")
    
    return plots

def generate_html_report(schema, validation_results, plots, no_browser=False):
    """Generate an HTML report with the validation results."""
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MT5 Database Validation Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f8f8f8;
            }
            .success {
                color: green;
            }
            .warning {
                color: orange;
            }
            .error {
                color: red;
            }
            .plot-container {
                margin: 20px 0;
                text-align: center;
            }
            .plot-container img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .summary {
                font-weight: bold;
                font-size: 1.1em;
                margin-top: 20px;
                padding: 15px;
                background-color: #e8f4f8;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MT5 Database Validation Report</h1>
            <p>Generated on: {{ generation_time }}</p>
            <p>Database: {{ db_path }}</p>
            <p>Table: {{ table_name }}</p>
            
            <div class="section">
                <h2>Table Schema</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Column Name</th>
                        <th>Data Type</th>
                        <th>Not Null</th>
                        <th>Default Value</th>
                        <th>Primary Key</th>
                    </tr>
                    {% for col in schema %}
                    <tr>
                        <td>{{ col[0] }}</td>
                        <td>{{ col[1] }}</td>
                        <td>{{ col[2] }}</td>
                        <td>{{ col[3] }}</td>
                        <td>{{ col[4] }}</td>
                        <td>{{ col[5] }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>Data Overview</h2>
                <p>Total rows: {{ validation_results.total_rows }}</p>
                
                <h3>Null Values</h3>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Null Count</th>
                        <th>Percentage</th>
                    </tr>
                    {% for col, count in validation_results.null_values.items() %}
                    <tr>
                        <td>{{ col }}</td>
                        <td>{{ count }}</td>
                        <td>{{ "%.2f"|format(count / validation_results.total_rows * 100) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
                
                {% if validation_results.duplicate_timestamps is defined %}
                <h3>Duplicate Timestamps</h3>
                <p>
                    {% if validation_results.duplicate_timestamps > 0 %}
                    <span class="warning">Found {{ validation_results.duplicate_timestamps }} duplicate timestamps</span>
                    {% else %}
                    <span class="success">No duplicate timestamps found</span>
                    {% endif %}
                </p>
                {% endif %}
            </div>
            
            {% if validation_results.price_anomalies is defined %}
            <div class="section">
                <h2>Price Anomalies</h2>
                <table>
                    <tr>
                        <th>Anomaly Type</th>
                        <th>Count</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>High < Low</td>
                        <td>{{ validation_results.price_anomalies.high_lt_low }}</td>
                        <td>
                            {% if validation_results.price_anomalies.high_lt_low > 0 %}
                            <span class="error">ERROR</span>
                            {% else %}
                            <span class="success">OK</span>
                            {% endif %}
                        </td>
                    </tr>
                    <tr>
                        <td>Close > High</td>
                        <td>{{ validation_results.price_anomalies.close_gt_high }}</td>
                        <td>
                            {% if validation_results.price_anomalies.close_gt_high > 0 %}
                            <span class="error">ERROR</span>
                            {% else %}
                            <span class="success">OK</span>
                            {% endif %}
                        </td>
                    </tr>
                    <tr>
                        <td>Close < Low</td>
                        <td>{{ validation_results.price_anomalies.close_lt_low }}</td>
                        <td>
                            {% if validation_results.price_anomalies.close_lt_low > 0 %}
                            <span class="error">ERROR</span>
                            {% else %}
                            <span class="success">OK</span>
                            {% endif %}
                        </td>
                    </tr>
                    <tr>
                        <td>Open > High</td>
                        <td>{{ validation_results.price_anomalies.open_gt_high }}</td>
                        <td>
                            {% if validation_results.price_anomalies.open_gt_high > 0 %}
                            <span class="error">ERROR</span>
                            {% else %}
                            <span class="success">OK</span>
                            {% endif %}
                        </td>
                    </tr>
                    <tr>
                        <td>Open < Low</td>
                        <td>{{ validation_results.price_anomalies.open_lt_low }}</td>
                        <td>
                            {% if validation_results.price_anomalies.open_lt_low > 0 %}
                            <span class="error">ERROR</span>
                            {% else %}
                            <span class="success">OK</span>
                            {% endif %}
                        </td>
                    </tr>
                </table>
            </div>
            {% endif %}
            
            {% if validation_results.time_gaps is defined and validation_results.time_gaps.total_gaps is defined %}
            <div class="section">
                <h2>Time Series Gaps</h2>
                <p>Total gaps: {{ validation_results.time_gaps.total_gaps }}</p>
                <p>Maximum gap: {{ "%.2f"|format(validation_results.time_gaps.max_gap) }} minutes</p>
                <p>Average gap: {{ "%.2f"|format(validation_results.time_gaps.avg_gap) }} minutes</p>
                
                {% if validation_results.largest_gaps is defined and validation_results.largest_gaps|length > 0 %}
                <h3>Largest Gaps</h3>
                <table>
                    <tr>
                        <th>Start Time</th>
                        <th>End Time</th>
                        <th>Duration (minutes)</th>
                    </tr>
                    {% for gap in validation_results.largest_gaps %}
                    <tr>
                        <td>{{ gap.start }}</td>
                        <td>{{ gap.end }}</td>
                        <td>{{ "%.2f"|format(gap.duration_minutes) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endif %}
            
            {% if validation_results.extreme_price_movements is defined %}
            <div class="section">
                <h2>Extreme Price Movements</h2>
                <p>Total extreme movements: {{ validation_results.extreme_price_movements.total_extreme_moves }}</p>
                <p>Maximum price range: {{ "%.2f"|format(validation_results.extreme_price_movements.max_range_pct) }}%</p>
                <p>Average price range: {{ "%.2f"|format(validation_results.extreme_price_movements.avg_range_pct) }}%</p>
                
                {% if validation_results.top_extreme_movements is defined and validation_results.top_extreme_movements|length > 0 %}
                <h3>Top Extreme Movements</h3>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Open</th>
                        <th>High</th>
                        <th>Low</th>
                        <th>Close</th>
                        <th>Range (%)</th>
                    </tr>
                    {% for move in validation_results.top_extreme_movements %}
                    <tr>
                        <td>{{ move.time }}</td>
                        <td>{{ "%.2f"|format(move.open) }}</td>
                        <td>{{ "%.2f"|format(move.high) }}</td>
                        <td>{{ "%.2f"|format(move.low) }}</td>
                        <td>{{ "%.2f"|format(move.close) }}</td>
                        <td>{{ "%.2f"|format(move.range_pct) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endif %}
            
            <div class="section">
                <h2>Data Visualization</h2>
                
                {% if plots.price_history is defined %}
                <div class="plot-container">
                    <h3>Price History</h3>
                    <img src="data:image/png;base64,{{ plots.price_history }}" alt="Price History">
                </div>
                {% endif %}
                
                {% if plots.daily_count is defined %}
                <div class="plot-container">
                    <h3>Daily Data Count</h3>
                    <img src="data:image/png;base64,{{ plots.daily_count }}" alt="Daily Data Count">
                </div>
                {% endif %}
                
                {% if plots.price_range_dist is defined %}
                <div class="plot-container">
                    <h3>Price Range Distribution</h3>
                    <img src="data:image/png;base64,{{ plots.price_range_dist }}" alt="Price Range Distribution">
                </div>
                {% endif %}
            </div>
            
            <div class="section summary">
                <h2>Summary</h2>
                <p>
                    {% set total_errors = 0 %}
                    {% if validation_results.price_anomalies is defined %}
                        {% set total_errors = validation_results.price_anomalies.high_lt_low + 
                                             validation_results.price_anomalies.close_gt_high + 
                                             validation_results.price_anomalies.close_lt_low + 
                                             validation_results.price_anomalies.open_gt_high + 
                                             validation_results.price_anomalies.open_lt_low %}
                    {% endif %}
                    
                    {% if total_errors > 0 %}
                    <span class="error">Found {{ total_errors }} price anomalies that need attention.</span>
                    {% else %}
                    <span class="success">No price anomalies detected.</span>
                    {% endif %}
                </p>
                
                <p>
                    {% if validation_results.time_gaps is defined and validation_results.time_gaps.total_gaps > 0 %}
                    <span class="warning">Found {{ validation_results.time_gaps.total_gaps }} time gaps in the data.</span>
                    {% else %}
                    <span class="success">No time gaps detected.</span>
                    {% endif %}
                </p>
                
                <p>
                    {% if validation_results.duplicate_timestamps is defined and validation_results.duplicate_timestamps > 0 %}
                    <span class="warning">Found {{ validation_results.duplicate_timestamps }} duplicate timestamps.</span>
                    {% else %}
                    <span class="success">No duplicate timestamps detected.</span>
                    {% endif %}
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Render the template
    template_obj = jinja2.Template(template)
    html_content = template_obj.render(
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        schema=schema,
        validation_results=validation_results,
        plots=plots
    )
    
    # Write to file
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report generated: {REPORT_PATH}")
    
    # Automatically open the report in the default web browser if not disabled
    if not no_browser:
        try:
            report_path = os.path.abspath(REPORT_PATH)
            print(f"Opening report in web browser: {report_path}")
            webbrowser.open('file://' + report_path)
        except Exception as e:
            print(f"Could not open report in browser: {e}")

def main():
    """Main function to run the validation."""
    parser = argparse.ArgumentParser(description='Validate MT5 OHLC data and generate a report')
    parser.add_argument('--no-browser', action='store_true', help='Do not automatically open the report in a browser')
    args = parser.parse_args()
    
    print("Starting MT5 OHLC data validation...")
    
    # Connect to the database
    conn = connect_to_db()
    if not conn:
        print("Failed to connect to the database. Exiting.")
        return
    
    # Get table schema
    schema = get_table_schema(conn)
    if not schema:
        print("Failed to get table schema. Exiting.")
        conn.close()
        return
    
    # Load OHLC data
    df = load_ohlc_data(conn)
    if df is None or df.empty:
        print("Failed to load OHLC data. Exiting.")
        conn.close()
        return
    
    # Close the database connection
    conn.close()
    
    # Validate the data
    validation_results = validate_data(df)
    
    # Generate plots
    plots = generate_plots(df)
    
    # Generate HTML report
    generate_html_report(schema, validation_results, plots, args.no_browser)
    
    print("Validation completed successfully.")

if __name__ == "__main__":
    main()