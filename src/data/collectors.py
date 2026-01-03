"""
Stock Data Collection Module
Handles fetching stock symbols and historical data from Yahoo Finance
"""

import logging
import sqlite3
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import numpy as np


class StockSymbolManager:
    """Manages stock symbol lists for NSE and BSE"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def get_nse_symbols(self) -> List[str]:
        """Fetch NSE stock symbols"""
        try:
            # Get Nifty 500 symbols (comprehensive list)
            nifty_500_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
            
            # Alternative approach using direct symbol list
            nse_symbols = []
            
            # Major Nifty indices symbols
            major_symbols = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
                'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS',
                'SBIN.NS', 'ASIANPAINT.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'HCLTECH.NS',
                'AXISBANK.NS', 'LT.NS', 'DMART.NS', 'SUNPHARMA.NS', 'TITAN.NS',
                'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'POWERGRID.NS', 'NTPC.NS',
                'TECHM.NS', 'M&M.NS', 'TATAMOTORS.NS', 'INDUSINDBK.NS', 'BAJAJFINSV.NS',
                'ONGC.NS', 'DIVISLAB.NS', 'COALINDIA.NS', 'GRASIM.NS', 'HDFCLIFE.NS',
                'DRREDDY.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS', 'BPCL.NS', 'EICHERMOT.NS',
                'CIPLA.NS', 'TATASTEEL.NS', 'ADANIENTS.NS', 'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS',
                'HEROMOTOCO.NS', 'UPL.NS', 'SBILIFE.NS', 'TATACONSUM.NS', 'GODREJCP.NS'
            ]
            
            # Try to get comprehensive list from NSE website
            try:
                response = requests.get(nifty_500_url, timeout=30)
                if response.status_code == 200:
                    df = pd.read_csv(nifty_500_url)
                    symbols = [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
                    nse_symbols.extend(symbols)
                    self.logger.info(f"Fetched {len(symbols)} NSE symbols from official source")
                else:
                    # Fallback to major symbols
                    nse_symbols = major_symbols
                    self.logger.warning("Using fallback NSE symbol list")
            except Exception as e:
                nse_symbols = major_symbols
                self.logger.warning(f"Failed to fetch NSE symbols, using fallback list: {e}")
            
            # Remove duplicates and validate
            nse_symbols = list(set(nse_symbols))
            self.logger.info(f"Total NSE symbols: {len(nse_symbols)}")
            
            return nse_symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching NSE symbols: {e}")
            return []
    
    def get_bse_symbols(self) -> List[str]:
        """Fetch BSE stock symbols"""
        try:
            # BSE Top 100 companies (major liquid stocks)
            bse_symbols = [
                'RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'INFY.BO', 'HINDUNILVR.BO',
                'HDFC.BO', 'ICICIBANK.BO', 'KOTAKBANK.BO', 'BHARTIARTL.BO', 'ITC.BO',
                'SBIN.BO', 'ASIANPAINT.BO', 'BAJFINANCE.BO', 'MARUTI.BO', 'HCLTECH.BO',
                'AXISBANK.BO', 'LT.BO', 'DMART.BO', 'SUNPHARMA.BO', 'TITAN.BO',
                'ULTRACEMCO.BO', 'WIPRO.BO', 'NESTLEIND.BO', 'POWERGRID.BO', 'NTPC.BO',
                'TECHM.BO', 'M&M.BO', 'TATAMOTORS.BO', 'INDUSINDBK.BO', 'BAJAJFINSV.BO'
            ]
            
            self.logger.info(f"BSE symbols: {len(bse_symbols)}")
            return bse_symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching BSE symbols: {e}")
            return []
    
    def get_all_symbols(self) -> List[str]:
        """Get all NSE and BSE symbols"""
        nse_symbols = self.get_nse_symbols()
        bse_symbols = self.get_bse_symbols()
        
        all_symbols = nse_symbols + bse_symbols
        self.logger.info(f"Total symbols (NSE + BSE): {len(all_symbols)}")
        
        return all_symbols
    
    def get_market_indices(self) -> List[str]:
        """Get market index symbols"""
        indices = [
            '^NSEI',    # Nifty 50
            '^BSESN',   # Sensex
            '^INDIAVIX' # India VIX
        ]
        return indices


class StockDataCollector:
    """Main data collector for stock historical data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.symbol_manager = StockSymbolManager(config)
        
        # Database setup
        self.db_path = config['data']['database']['path']
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with proper schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Stock data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_data (
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        adj_close REAL,
                        volume INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, date)
                    )
                """)
                
                # Stock metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_metadata (
                        symbol TEXT PRIMARY KEY,
                        name TEXT,
                        sector TEXT,
                        industry TEXT,
                        market_cap REAL,
                        exchange TEXT,
                        currency TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Market indices table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_indices (
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        close REAL,
                        volume INTEGER,
                        PRIMARY KEY (symbol, date)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_data_symbol ON stock_data(symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_indices_symbol ON market_indices(symbol)")
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def download_single_stock(self, symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """Download historical data for a single stock"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Download historical data
            hist = ticker.history(
                period=period,
                interval=self.config['data']['yahoo']['interval'],
                auto_adjust=False,
                prepost=False
            )
            
            if hist.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None
            
            # Clean and validate data
            hist = hist.dropna()
            
            # Check data quality
            min_days = self.config['data']['quality']['min_trading_days']
            if len(hist) < min_days:
                self.logger.warning(f"Insufficient data for {symbol}: {len(hist)} days")
                return None
            
            # Add symbol column
            hist['symbol'] = symbol
            hist.reset_index(inplace=True)
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get stock metadata information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', '')),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'exchange': info.get('exchange', ''),
                'currency': info.get('currency', 'INR')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {e}")
            return {'symbol': symbol}
    
    def save_stock_data(self, df: pd.DataFrame):
        """Save stock data to database"""
        if df is None or df.empty:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare data for insertion
                data_to_insert = []
                for _, row in df.iterrows():
                    data_to_insert.append((
                        row['symbol'],
                        row['Date'],
                        row['Open'],
                        row['High'],
                        row['Low'],
                        row['Close'],
                        row['Adj Close'],
                        int(row['Volume']) if pd.notna(row['Volume']) else 0
                    ))
                
                # Insert with conflict resolution
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO stock_data 
                    (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, data_to_insert)
                
                conn.commit()
                self.logger.debug(f"Saved {len(data_to_insert)} records for {df['symbol'].iloc[0]}")
                
        except Exception as e:
            self.logger.error(f"Error saving stock data: {e}")
    
    def save_stock_metadata(self, metadata: Dict):
        """Save stock metadata to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO stock_metadata
                    (symbol, name, sector, industry, market_cap, exchange, currency)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.get('symbol', ''),
                    metadata.get('name', ''),
                    metadata.get('sector', ''),
                    metadata.get('industry', ''),
                    metadata.get('market_cap', 0),
                    metadata.get('exchange', ''),
                    metadata.get('currency', 'INR')
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def collect_single_stock_data(self, symbol: str) -> bool:
        """Collect data for a single stock (data + metadata)"""
        try:
            self.logger.info(f"Collecting data for {symbol}")
            
            # Download historical data
            df = self.download_single_stock(symbol)
            if df is not None:
                self.save_stock_data(df)
                
                # Get and save metadata
                metadata = self.get_stock_info(symbol)
                self.save_stock_metadata(metadata)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")
            return False
    
    def collect_market_indices(self):
        """Collect market index data"""
        indices = self.symbol_manager.get_market_indices()
        
        for index in indices:
            try:
                self.logger.info(f"Collecting index data for {index}")
                
                ticker = yf.Ticker(index)
                hist = ticker.history(period="5y")
                
                if not hist.empty:
                    data_to_insert = []
                    for date, row in hist.iterrows():
                        data_to_insert.append((
                            index,
                            date.strftime('%Y-%m-%d'),
                            row['Close'],
                            int(row['Volume']) if pd.notna(row['Volume']) else 0
                        ))
                    
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.executemany("""
                            INSERT OR REPLACE INTO market_indices
                            (symbol, date, close, volume)
                            VALUES (?, ?, ?, ?)
                        """, data_to_insert)
                        conn.commit()
                        
                    self.logger.info(f"Saved {len(data_to_insert)} records for {index}")
                    
            except Exception as e:
                self.logger.error(f"Error collecting index data for {index}: {e}")
    
    def collect_all_data(self, symbols: Optional[List[str]] = None, force_update: bool = False):
        """Collect data for all stocks using parallel processing"""
        if symbols is None:
            symbols = self.symbol_manager.get_all_symbols()
        
        self.logger.info(f"Starting data collection for {len(symbols)} stocks")
        
        # Collect market indices first
        self.collect_market_indices()
        
        # Parallel collection of stock data
        max_workers = self.config['data']['yahoo']['threads']
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.collect_single_stock_data, symbol): symbol
                for symbol in symbols
            }
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        
                    # Rate limiting
                    time.sleep(0.1)  # Small delay to be respectful to Yahoo Finance
                    
                    # Progress reporting
                    if (successful + failed) % 50 == 0:
                        self.logger.info(f"Progress: {successful + failed}/{len(symbols)} stocks processed")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    failed += 1
        
        self.logger.info(f"Data collection completed: {successful} successful, {failed} failed")
        
        # Database optimization
        self.optimize_database()
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("VACUUM")
                cursor.execute("ANALYZE")
                conn.commit()
                
            self.logger.info("Database optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
    
    def get_stock_data(self, symbol: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve stock data from database"""
        try:
            query = "SELECT * FROM stock_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY date"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM stock_data")
                symbols = [row[0] for row in cursor.fetchall()]
                
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
