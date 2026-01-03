"""
Data Processing Module
Handles data cleaning, preprocessing, and validation for stock data
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')


class DataCleaner:
    """Data cleaning and validation utilities"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds from config
        self.outlier_threshold = config['data']['quality']['outlier_threshold']
        self.max_missing_pct = config['data']['quality']['max_missing_pct']
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Detect and handle outliers using statistical methods"""
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(df_clean[col], nan_policy='omit'))
            
            # Identify outliers
            outliers = z_scores > self.outlier_threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                self.logger.warning(f"Found {outlier_count} outliers in {col}")
                
                # Replace outliers with interpolated values
                df_clean.loc[outliers, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='time')
        
        return df_clean
    
    def handle_splits_and_dividends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle stock splits and dividend adjustments"""
        df_clean = df.copy()
        
        # Calculate price ratios to detect splits
        if 'close' in df_clean.columns and 'adj_close' in df_clean.columns:
            df_clean['price_ratio'] = df_clean['close'] / df_clean['adj_close']
            
            # Detect significant price adjustments (splits/dividends)
            ratio_mean = df_clean['price_ratio'].mean()
            ratio_changes = np.abs(df_clean['price_ratio'].pct_change()) > 0.1
            
            if ratio_changes.any():
                self.logger.info("Detected price adjustments (splits/dividends)")
                # Use adjusted close for consistency
                df_clean['close'] = df_clean['adj_close']
        
        # Remove temporary column
        if 'price_ratio' in df_clean.columns:
            df_clean.drop('price_ratio', axis=1, inplace=True)
        
        return df_clean
    
    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """Validate data integrity and consistency"""
        if df.empty:
            return False
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).any()
        
        if invalid_ohlc:
            self.logger.warning("Found invalid OHLC relationships")
        
        # Check for negative prices or volumes
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any().any()
        negative_volumes = (df['volume'] < 0).any()
        
        if negative_prices or negative_volumes:
            self.logger.warning("Found negative prices or volumes")
        
        # Check missing data percentage
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        if missing_pct > self.max_missing_pct:
            self.logger.warning(f"High missing data percentage: {missing_pct:.2%}")
            return False
        
        return True
    
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete data cleaning pipeline for stock data"""
        if df.empty:
            return df
        
        # Make copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing dates (forward fill for short gaps)
        df_clean = df_clean.asfreq('D')
        df_clean = df_clean.fillna(method='ffill', limit=5)
        
        # Handle splits and dividends
        df_clean = self.handle_splits_and_dividends(df_clean)
        
        # Remove outliers
        df_clean = self.detect_outliers(df_clean)
        
        # Handle remaining missing values
        df_clean = df_clean.dropna()
        
        # Validate final data
        if not self.validate_data_integrity(df_clean):
            self.logger.error("Data validation failed after cleaning")
            return pd.DataFrame()
        
        return df_clean


class DataProcessor:
    """Main data processing class for preparing training data"""
    
    def __init__(self, config: Dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.cleaner = DataCleaner(config)
        
        # Processing parameters
        self.lookback_window = config['model']['tft']['lookback_window']
        self.prediction_horizons = config['model']['tft']['prediction_horizons']
    
    def get_stock_data_from_db(self, symbol: str, start_date: str = None, 
                              end_date: str = None) -> pd.DataFrame:
        """Load stock data from database"""
        try:
            query = """
                SELECT date, open, high, low, close, adj_close, volume
                FROM stock_data
                WHERE symbol = ?
            """
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
                
                # Clean the data
                df = self.cleaner.clean_stock_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Load market index data"""
        indices = ['^NSEI', '^BSESN', '^INDIAVIX']
        market_data = {}
        
        for index in indices:
            try:
                query = """
                    SELECT date, close, volume
                    FROM market_indices
                    WHERE symbol = ?
                """
                params = [index]
                
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
                    market_data[index] = df
                    
            except Exception as e:
                self.logger.error(f"Error loading market data for {index}: {e}")
        
        return market_data
    
    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Calculate returns for different periods"""
        if periods is None:
            periods = [1, 5, 22]  # Daily, weekly, monthly
        
        df_returns = df.copy()
        
        # Price returns
        for period in periods:
            df_returns[f'return_{period}d'] = df_returns['close'].pct_change(periods=period)
            # Protect log calculation from zero/negative values
            prev_close = df_returns['close'].shift(period)
            df_returns[f'log_return_{period}d'] = np.log((df_returns['close'] + 1e-10) / (prev_close + 1e-10))
        
        # Volume returns
        # Calculate volume return with protection against zero/small volumes
        prev_volume = df_returns['volume'].shift(1)
        df_returns['volume_return_1d'] = (df_returns['volume'] - prev_volume) / (prev_volume + 1e-10)
        
        return df_returns
    
    def calculate_volatility(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Calculate rolling volatility measures"""
        if windows is None:
            windows = [5, 10, 22]
        
        df_vol = df.copy()
        
        # Calculate daily returns if not present
        if 'return_1d' not in df_vol.columns:
            df_vol['return_1d'] = df_vol['close'].pct_change().fillna(0)
        
        # Rolling volatility
        for window in windows:
            df_vol[f'volatility_{window}d'] = df_vol['return_1d'].rolling(window=window).std() * np.sqrt(252)
            
            # High-Low volatility (Parkinson estimator)
            df_vol[f'hl_volatility_{window}d'] = np.sqrt(
                (np.log(df_vol['high'] / df_vol['low']) ** 2).rolling(window=window).mean() * 252 / (4 * np.log(2))
            )
        
        return df_vol
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features (day of week, month, etc.)"""
        df_temp = df.copy()
        
        # Basic temporal features
        df_temp['day_of_week'] = df_temp.index.dayofweek
        df_temp['month'] = df_temp.index.month
        df_temp['quarter'] = df_temp.index.quarter
        df_temp['day_of_month'] = df_temp.index.day
        df_temp['week_of_year'] = df_temp.index.isocalendar().week
        
        # Cyclical encoding for temporal features
        df_temp['day_of_week_sin'] = np.sin(2 * np.pi * df_temp['day_of_week'] / 7)
        df_temp['day_of_week_cos'] = np.cos(2 * np.pi * df_temp['day_of_week'] / 7)
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        
        # Business day indicator
        df_temp['is_business_day'] = df_temp.index.to_series().apply(
            lambda x: x.weekday() < 5
        ).astype(int)
        
        # Earnings seasons (approximate)
        earnings_months = self.config['features']['temporal']['earnings_season_months']
        df_temp['is_earnings_season'] = df_temp['month'].isin(earnings_months).astype(int)
        
        return df_temp
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'robust') -> Tuple[pd.DataFrame, Dict]:
        """Normalize features using specified method"""
        df_norm = df.copy()
        scalers = {}
        
        # Identify numeric columns (excluding temporal features)
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        temporal_cols = [col for col in numeric_cols if any(temp in col for temp in 
                        ['day_of_week', 'month', 'quarter', 'is_', 'sin', 'cos'])]
        
        # Columns to normalize (exclude temporal features)
        cols_to_normalize = [col for col in numeric_cols if col not in temporal_cols]
        
        if method == 'robust':
            scaler_class = RobustScaler
        else:
            scaler_class = StandardScaler
        
        for col in cols_to_normalize:
            if col in df_norm.columns and not df_norm[col].isna().all():
                # Check for infinity values before scaling
                if np.isinf(df_norm[col]).any():
                    self.logger.warning(f"Found infinity values in column {col}, replacing with NaN")
                    df_norm[col] = df_norm[col].replace([np.inf, -np.inf], np.nan)
                    df_norm[col] = df_norm[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Additional check for extreme values that might cause issues
                if df_norm[col].abs().max() > 1e10:
                    self.logger.warning(f"Found extreme values in column {col}, clipping")
                    df_norm[col] = df_norm[col].clip(-1e6, 1e6)
                
                scaler = scaler_class()
                try:
                    df_norm[[col]] = scaler.fit_transform(df_norm[[col]])
                    scalers[col] = scaler
                except Exception as e:
                    self.logger.warning(f"Failed to scale column {col}: {e}, using RobustScaler instead")
                    # Fallback to RobustScaler which is more robust to outliers
                    robust_scaler = RobustScaler()
                    df_norm[[col]] = robust_scaler.fit_transform(df_norm[[col]])
                    scalers[col] = robust_scaler
        
        return df_norm, scalers
    
    def create_sequences(self, df: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create sequences for time series modeling"""
        if len(df) < self.lookback_window + max(self.prediction_horizons):
            self.logger.warning(f"Insufficient data for sequence creation: {len(df)} rows")
            return np.array([]), np.array([]), []
        
        # Prepare feature columns (exclude target columns)
        feature_cols = [col for col in df.columns if col not in target_cols]
        
        X_sequences = []
        y_sequences = []
        sequence_dates = []
        
        # Create sequences
        for i in range(self.lookback_window, len(df) - max(self.prediction_horizons) + 1):
            # Input sequence (lookback window)
            X_seq = df.iloc[i-self.lookback_window:i][feature_cols].values
            
            # Target sequences for different horizons
            y_seq = []
            for horizon in self.prediction_horizons:
                if i + horizon - 1 < len(df):
                    target_values = df.iloc[i + horizon - 1][target_cols].values
                    y_seq.extend(target_values)
                else:
                    # Pad with last available values if horizon extends beyond data
                    target_values = df.iloc[-1][target_cols].values
                    y_seq.extend(target_values)
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
            sequence_dates.append(df.index[i])
        
        return np.array(X_sequences), np.array(y_sequences), sequence_dates
    
    def process_single_stock(self, symbol: str, start_date: str = None, 
                            end_date: str = None) -> Dict:
        """Complete processing pipeline for a single stock"""
        try:
            self.logger.info(f"Processing stock: {symbol}")
            
            # Load stock data
            df = self.get_stock_data_from_db(symbol, start_date, end_date)
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return {}
            
            # Add returns and volatility
            df = self.calculate_returns(df)
            df = self.calculate_volatility(df)
            
            # Add temporal features
            df = self.add_temporal_features(df)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            if len(df) < self.lookback_window + max(self.prediction_horizons):
                self.logger.warning(f"Insufficient data for {symbol} after processing")
                return {}
            
            # Normalize features
            df_norm, scalers = self.normalize_features(df)
            
            # Define target columns for prediction
            target_cols = ['close']  # Can extend to include other targets
            
            # Create sequences
            X, y, dates = self.create_sequences(df_norm, target_cols)
            
            if len(X) == 0:
                self.logger.warning(f"No sequences created for {symbol}")
                return {}
            
            return {
                'symbol': symbol,
                'X': X,
                'y': y,
                'dates': dates,
                'scalers': scalers,
                'feature_cols': [col for col in df_norm.columns if col not in target_cols],
                'target_cols': target_cols,
                'raw_data': df,
                'processed_data': df_norm
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return {}
    
    def process_multiple_stocks(self, symbols: List[str], start_date: str = None, 
                               end_date: str = None) -> Dict[str, Dict]:
        """Process multiple stocks"""
        results = {}
        
        self.logger.info(f"Processing {len(symbols)} stocks")
        
        for i, symbol in enumerate(symbols):
            result = self.process_single_stock(symbol, start_date, end_date)
            
            if result:
                results[symbol] = result
                
            # Progress reporting
            if (i + 1) % 50 == 0:
                self.logger.info(f"Processed {i + 1}/{len(symbols)} stocks")
        
        self.logger.info(f"Successfully processed {len(results)}/{len(symbols)} stocks")
        return results
    
    def get_dataset_statistics(self, processed_data: Dict[str, Dict]) -> Dict:
        """Calculate dataset statistics"""
        if not processed_data:
            return {}
        
        stats = {
            'num_stocks': len(processed_data),
            'total_sequences': sum(len(data['X']) for data in processed_data.values()),
            'feature_dimension': processed_data[list(processed_data.keys())[0]]['X'].shape[-1],
            'lookback_window': self.lookback_window,
            'prediction_horizons': self.prediction_horizons,
            'date_range': {
                'start': min(min(data['dates']) for data in processed_data.values()),
                'end': max(max(data['dates']) for data in processed_data.values())
            }
        }
        
        return stats
