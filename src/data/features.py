"""
Feature Engineering Module
Technical indicators, market features, and cross-sectional features for stock prediction
"""

import logging
import sqlite3
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings

# Technical analysis libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available, using pandas implementation")

warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Technical analysis indicators implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ta_config = config['features']['technical']
    
    def sma(self, series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window).mean()
    
    def ema(self, series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=window).mean()
    
    def rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(series.values, timeperiod=window), index=series.index)
        else:
            # Manual RSI calculation
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            return 100 - (100 / (1 + rs))
    
    def macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        if TALIB_AVAILABLE:
            macd_line, macd_signal, macd_hist = talib.MACD(
                series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return {
                'macd': pd.Series(macd_line, index=series.index),
                'macd_signal': pd.Series(macd_signal, index=series.index),
                'macd_histogram': pd.Series(macd_hist, index=series.index)
            }
        else:
            # Manual MACD calculation
            ema_fast = self.ema(series, fast)
            ema_slow = self.ema(series, slow)
            macd_line = ema_fast - ema_slow
            macd_signal = self.ema(macd_line, signal)
            macd_hist = macd_line - macd_signal
            
            return {
                'macd': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_hist
            }
    
    def bollinger_bands(self, series: pd.Series, window: int = 20, std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        if TALIB_AVAILABLE:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                series.values, timeperiod=window, nbdevup=std, nbdevdn=std
            )
            return {
                'bb_upper': pd.Series(bb_upper, index=series.index),
                'bb_middle': pd.Series(bb_middle, index=series.index),
                'bb_lower': pd.Series(bb_lower, index=series.index)
            }
        else:
            # Manual Bollinger Bands calculation
            sma = self.sma(series, window)
            rolling_std = series.rolling(window=window).std()
            
            return {
                'bb_upper': sma + (rolling_std * std),
                'bb_middle': sma,
                'bb_lower': sma - (rolling_std * std)
            }
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values,
                                     fastk_period=k_window, slowk_period=d_window, slowd_period=d_window)
            return {
                'stoch_k': pd.Series(slowk, index=close.index),
                'stoch_d': pd.Series(slowd, index=close.index)
            }
        else:
            # Manual calculation
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
            d_percent = k_percent.rolling(window=d_window).mean()
            
            return {
                'stoch_k': k_percent,
                'stoch_d': d_percent
            }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=window),
                           index=close.index)
        else:
            # Manual ATR calculation
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return true_range.rolling(window=window).mean()
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=window),
                           index=close.index)
        else:
            # Manual calculation
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            return -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        else:
            # Manual calculation
            obv = np.zeros(len(close))
            obv[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv[i] = obv[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv[i] = obv[i-1] - volume.iloc[i]
                else:
                    obv[i] = obv[i-1]
            
            return pd.Series(obv, index=close.index)


class VolumeIndicators:
    """Volume-based technical indicators"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def volume_sma(self, volume: pd.Series, window: int) -> pd.Series:
        """Volume Simple Moving Average"""
        return volume.rolling(window=window).mean()
    
    def volume_ratio(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume Ratio (current volume / average volume)"""
        avg_volume = self.volume_sma(volume, window)
        return volume / (avg_volume + 1e-10)
    
    def price_volume_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price Volume Trend"""
        price_change = close.pct_change()
        pvt = (price_change * volume).cumsum()
        return pvt
    
    def volume_weighted_average_price(self, high: pd.Series, low: pd.Series, 
                                    close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=window).sum() / (volume.rolling(window=window).sum() + 1e-10)
        return vwap


class MarketFeatures:
    """Market-wide and cross-sectional features"""
    
    def __init__(self, config: Dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.market_config = config['features']['market']
    
    def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market index data"""
        indices = ['^NSEI', '^BSESN', '^INDIAVIX']
        market_data = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for index in indices:
                    query = "SELECT date, close FROM market_indices WHERE symbol = ? ORDER BY date"
                    df = pd.read_sql_query(query, conn, params=[index])
                    
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        market_data[index] = df
                        
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
        
        return market_data
    
    def calculate_market_correlation(self, stock_data: pd.DataFrame, 
                                   market_data: pd.DataFrame, window: int = 252) -> pd.Series:
        """Calculate rolling correlation with market"""
        if stock_data.empty or market_data.empty:
            return pd.Series(dtype=float)
        
        # Align data
        aligned_stock, aligned_market = stock_data['close'].align(market_data['close'], join='inner')
        
        if len(aligned_stock) < window:
            return pd.Series(dtype=float)
        
        # Calculate rolling correlation
        correlation = aligned_stock.rolling(window=window).corr(aligned_market)
        return correlation
    
    def calculate_beta(self, stock_data: pd.DataFrame, market_data: pd.DataFrame, 
                      window: int = 252) -> pd.Series:
        """Calculate stock beta (systematic risk)"""
        if stock_data.empty or market_data.empty:
            return pd.Series(dtype=float)
        
        # Calculate returns
        stock_returns = stock_data['close'].pct_change()
        market_returns = market_data['close'].pct_change()
        
        # Align data
        aligned_stock, aligned_market = stock_returns.align(market_returns, join='inner')
        
        if len(aligned_stock) < window:
            return pd.Series(dtype=float)
        
        # Calculate rolling beta
        def rolling_beta(x, y):
            if len(x) < 2 or len(y) < 2:
                return np.nan
            return np.cov(x, y)[0, 1] / np.var(y)
        
        beta = pd.Series(index=aligned_stock.index, dtype=float)
        for i in range(window, len(aligned_stock)):
            stock_window = aligned_stock.iloc[i-window:i]
            market_window = aligned_market.iloc[i-window:i]
            beta.iloc[i] = rolling_beta(stock_window, market_window)
        
        return beta
    
    def calculate_relative_strength(self, stock_data: pd.DataFrame, 
                                  market_data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Calculate relative strength vs market"""
        if periods is None:
            periods = self.market_config['relative_strength_periods']
        
        rs_features = pd.DataFrame(index=stock_data.index)
        
        # Align data
        stock_close = stock_data['close']
        market_close = market_data['close']
        aligned_stock, aligned_market = stock_close.align(market_close, join='inner')
        
        for period in periods:
            if len(aligned_stock) >= period:
                stock_return = aligned_stock.pct_change(periods=period)
                market_return = aligned_market.pct_change(periods=period)
                rs_features[f'relative_strength_{period}d'] = stock_return - market_return
        
        return rs_features


class FeatureEngineer:
    """Main feature engineering class combining all feature types"""
    
    def __init__(self, config: Dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature generators
        self.technical = TechnicalIndicators(config)
        self.volume_indicators = VolumeIndicators(config)
        self.market_features = MarketFeatures(config, db_path)
        
        # Configuration parameters
        self.ta_config = config['features']['technical']
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        df_features = df.copy()
        
        try:
            # Moving averages
            for period in self.ta_config['sma_periods']:
                df_features[f'sma_{period}'] = self.technical.sma(df['close'], period)
                
            for period in self.ta_config['ema_periods']:
                df_features[f'ema_{period}'] = self.technical.ema(df['close'], period)
            
            # RSI
            rsi_period = self.ta_config['rsi_period']
            df_features['rsi'] = self.technical.rsi(df['close'], rsi_period)
            
            # MACD
            macd_params = self.ta_config['macd_params']
            macd_data = self.technical.macd(df['close'], *macd_params)
            for key, series in macd_data.items():
                df_features[key] = series
            
            # Bollinger Bands
            bb_period = self.ta_config['bollinger_period']
            bb_std = self.ta_config['bollinger_std']
            bb_data = self.technical.bollinger_bands(df['close'], bb_period, bb_std)
            for key, series in bb_data.items():
                df_features[key] = series
            
            # Additional indicators
            if all(col in df.columns for col in ['high', 'low', 'close']):
                stoch_data = self.technical.stochastic(df['high'], df['low'], df['close'])
                for key, series in stoch_data.items():
                    df_features[key] = series
                
                df_features['atr'] = self.technical.atr(df['high'], df['low'], df['close'])
                df_features['williams_r'] = self.technical.williams_r(df['high'], df['low'], df['close'])
            
            # Volume indicators
            if 'volume' in df.columns:
                df_features['obv'] = self.technical.obv(df['close'], df['volume'])
                df_features['volume_sma_20'] = self.volume_indicators.volume_sma(df['volume'], 20)
                df_features['volume_ratio'] = self.volume_indicators.volume_ratio(df['volume'])
                df_features['pvt'] = self.volume_indicators.price_volume_trend(df['close'], df['volume'])
                
                if all(col in df.columns for col in ['high', 'low']):
                    df_features['vwap'] = self.volume_indicators.volume_weighted_average_price(
                        df['high'], df['low'], df['close'], df['volume']
                    )
            
            # Price-based features
            df_features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
            df_features['daily_return'] = df['close'].pct_change()
            df_features['log_return'] = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
            
            # Momentum features
            for period in [5, 10, 20]:
                df_features[f'momentum_{period}'] = df['close'] / (df['close'].shift(period) + 1e-10) - 1
                df_features[f'roc_{period}'] = df['close'].pct_change(periods=period)
            
            # Volatility features
            for window in [5, 10, 20]:
                returns = df['close'].pct_change()
                df_features[f'volatility_{window}'] = returns.rolling(window=window).std()
                df_features[f'volatility_ratio_{window}'] = (
                    df_features[f'volatility_{window}'] / 
                    (df_features[f'volatility_{window}'].rolling(window=50).mean() + 1e-10)
                )
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
        
        return df_features
    
    def add_market_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add market-relative features"""
        df_features = df.copy()
        
        try:
            # For now, skip market features to avoid data dependency issues
            # Add simple market-proxy features instead
            
            # Simple trend features
            for window in [5, 20]:
                df_features[f'price_trend_{window}'] = (
                    df_features['close'] / df_features['close'].shift(window) - 1
                ).fillna(0)
                
            # Volume trend 
            df_features['volume_trend'] = (
                df_features['volume'] / df_features['volume'].shift(20) - 1
            ).fillna(0)
            
            # Market correlation placeholder (use price correlation with its own MA)
            df_features['market_correlation'] = (
                df_features['close'].rolling(50).corr(df_features['close'].rolling(200).mean())
            ).fillna(0.5)
            
            # Beta placeholder
            df_features['beta'] = 1.0
            
            self.logger.info(f"Added simplified market features for {symbol}")
            
            # VIX features (if available)
            # VIX features would go here if market_data was available
            # For now, add placeholder VIX features
            df_features['vix'] = 20.0  # Average VIX level
            df_features['vix_sma_20'] = 20.0
            df_features['vix_change'] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error adding market features: {e}")
            
            # Fallback: add default market features
            df_features['market_correlation'] = 0.5
            df_features['beta'] = 1.0
            df_features['price_trend_5'] = 0.0
            df_features['price_trend_20'] = 0.0 
            df_features['volume_trend'] = 0.0
            
        return df_features
    
    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern recognition features"""
        df_features = df.copy()
        
        try:
            # Candlestick patterns (simplified)
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Doji pattern
                body_size = np.abs(df['close'] - df['open'])
                range_size = df['high'] - df['low']
                df_features['is_doji'] = (body_size / (range_size + 1e-10) < 0.1).astype(int)
                
                # Hammer pattern
                lower_shadow = np.minimum(df['open'], df['close']) - df['low']
                upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
                df_features['is_hammer'] = (
                    (lower_shadow > 2 * body_size) & 
                    (upper_shadow < body_size)
                ).astype(int)
                
                # Engulfing patterns
                prev_body_size = body_size.shift(1)
                is_green = (df['close'] > df['open']).astype(int)
                prev_is_green = is_green.shift(1)
                
                df_features['bullish_engulfing'] = (
                    (is_green == 1) & 
                    (prev_is_green == 0) & 
                    (body_size > prev_body_size)
                ).astype(int)
                
                df_features['bearish_engulfing'] = (
                    (is_green == 0) & 
                    (prev_is_green == 1) & 
                    (body_size > prev_body_size)
                ).astype(int)
            
            # Gap patterns
            df_features['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
            df_features['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
            
            # Support and resistance levels (simplified)
            for window in [20, 50]:
                df_features[f'resistance_{window}'] = df['high'].rolling(window=window).max()
                df_features[f'support_{window}'] = df['low'].rolling(window=window).min()
                df_features[f'distance_to_resistance_{window}'] = (
                    df_features[f'resistance_{window}'] - df['close']
                ) / (df['close'] + 1e-10)
                df_features[f'distance_to_support_{window}'] = (
                    df['close'] - df_features[f'support_{window}']
                ) / (df['close'] + 1e-10)
                
        except Exception as e:
            self.logger.error(f"Error adding price patterns: {e}")
        
        return df_features
    
    def engineer_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        if df.empty:
            return df
        
        self.logger.info(f"Engineering features for {symbol}")
        
        # Add technical indicators
        df_features = self.add_technical_indicators(df)
        
        # Add market-relative features
        df_features = self.add_market_features(df_features, symbol)
        
        # Add price patterns
        df_features = self.add_price_patterns(df_features)
        
        # Remove infinite values and handle NaN more aggressively
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Check for any remaining infinite values
        if np.isinf(df_features.values).any():
            self.logger.warning(f"Still found infinite values in {symbol}, replacing with NaN")
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values using multiple strategies
        # 1. Forward fill first
        df_features = df_features.fillna(method='ffill')
        # 2. Backward fill for any remaining NaN at the start
        df_features = df_features.fillna(method='bfill')
        # 3. Fill any remaining with 0
        df_features = df_features.fillna(0)
        
        # Final check and clip extreme values
        df_features = df_features.clip(-1e6, 1e6)  # Clip to reasonable range
        
        # Log feature statistics
        feature_count = len(df_features.columns) - len(df.columns)
        self.logger.info(f"Added {feature_count} features for {symbol}")
        
        return df_features
