"""
Trading Signal Generation Module
Generates buy/sell/hold signals based on TFT model predictions
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from enum import Enum

from ..models.tft import TemporalFusionTransformer
from ..data.processors import DataProcessor
from ..data.features import FeatureEngineer


class SignalType(Enum):
    """Enumeration for signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


class SignalStrength(Enum):
    """Enumeration for signal strength levels"""
    WEAK = 1
    MEDIUM = 2
    STRONG = 3


class TradingSignal:
    """Class to represent a trading signal"""
    
    def __init__(self, symbol: str, signal_type: SignalType, strength: SignalStrength,
                 confidence: float, price_target: float, current_price: float,
                 timestamp: datetime, horizon: str = "1d"):
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.confidence = confidence
        self.price_target = price_target
        self.current_price = current_price
        self.timestamp = timestamp
        self.horizon = horizon
        
        # Calculated fields
        self.expected_return = (price_target / current_price - 1) if current_price > 0 else 0
        self.risk_reward_ratio = abs(self.expected_return) / max(0.01, 0.02)  # Assume 2% risk
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary representation"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.name,
            'strength': self.strength.name,
            'confidence': self.confidence,
            'price_target': self.price_target,
            'current_price': self.current_price,
            'expected_return': self.expected_return,
            'risk_reward_ratio': self.risk_reward_ratio,
            'timestamp': self.timestamp.isoformat(),
            'horizon': self.horizon
        }


class SignalGenerator:
    """Main class for generating trading signals from TFT predictions"""
    
    def __init__(self, config: Dict, model_path: str):
        self.config = config
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # Signal generation parameters
        self.signal_config = config['prediction']['signals']
        self.buy_threshold = self.signal_config['buy_threshold']
        self.sell_threshold = self.signal_config['sell_threshold']
        self.confidence_threshold = self.signal_config['confidence_threshold']
        
        # Initialize components
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.db_path = config['data']['database']['path']
        
        # Load model
        self._load_model()
        
        # Initialize data processors
        self.data_processor = DataProcessor(config, self.db_path)
        self.feature_engineer = FeatureEngineer(config, self.db_path)
    
    def _load_model(self):
        """Load the trained TFT model"""
        try:
            self.model = TemporalFusionTransformer(self.config).to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _prepare_prediction_data(self, symbol: str, end_date: str = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Prepare data for prediction"""
        try:
            # Get stock data
            stock_data = self.data_processor.get_stock_data_from_db(symbol, end_date=end_date)
            
            if stock_data.empty or len(stock_data) < self.config['model']['tft']['lookback_window']:
                return None
            
            # Add features
            stock_data_with_features = self.feature_engineer.engineer_features(stock_data, symbol)
            
            # Process data
            processed_data = self.data_processor.process_single_stock(symbol, end_date=end_date)
            
            if not processed_data or len(processed_data['X']) == 0:
                return None
            
            # Get the most recent sequence
            X = torch.FloatTensor(processed_data['X'][-1:]).to(self.device)  # Last sequence
            
            # Static features (simplified - using symbol characteristics)
            static_features = torch.FloatTensor([[hash(symbol) % 1000 / 1000.0] * 10]).to(self.device)
            
            return static_features, X
            
        except Exception as e:
            self.logger.error(f"Error preparing data for {symbol}: {e}")
            return None
    
    def _calculate_confidence(self, predictions: Dict[str, torch.Tensor]) -> float:
        """Calculate prediction confidence based on attention weights and prediction consistency"""
        try:
            # Use attention weights as proxy for confidence
            attention_weights = predictions.get('attention_weights', None)
            if attention_weights is not None:
                # Higher attention concentration = higher confidence
                attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
                confidence = 1.0 - (attention_entropy / torch.log(torch.tensor(float(attention_weights.size(-1)))))
                return float(confidence.mean())
            
            # Fallback: use consistency across different prediction horizons
            price_predictions = []
            for task in ['daily_price', 'weekly_price', 'monthly_price']:
                if task in predictions:
                    price_predictions.append(predictions[task].cpu().numpy())
            
            if len(price_predictions) > 1:
                # Calculate coefficient of variation as consistency measure
                pred_array = np.array(price_predictions).flatten()
                cv = np.std(pred_array) / (np.mean(np.abs(pred_array)) + 1e-8)
                confidence = 1.0 / (1.0 + cv)  # Higher consistency = higher confidence
                return min(confidence, 1.0)
            
            return 0.5  # Default confidence
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def _determine_signal_strength(self, expected_return: float, confidence: float) -> SignalStrength:
        """Determine signal strength based on expected return and confidence"""
        strength_score = abs(expected_return) * confidence
        
        if strength_score > 0.03:  # 3%+ return with high confidence
            return SignalStrength.STRONG
        elif strength_score > 0.015:  # 1.5%+ return with medium confidence
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK
    
    def generate_signal(self, symbol: str, current_price: float = None) -> Optional[TradingSignal]:
        """Generate trading signal for a single stock"""
        try:
            # Prepare prediction data
            data = self._prepare_prediction_data(symbol)
            if data is None:
                return None
            
            static_features, dynamic_features = data
            
            # Get current price if not provided
            if current_price is None:
                recent_data = self.data_processor.get_stock_data_from_db(symbol)
                if recent_data.empty:
                    return None
                current_price = float(recent_data['close'].iloc[-1])
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model(static_features, dynamic_features)
            
            # Extract daily price prediction (primary signal)
            daily_pred = float(predictions['daily_price'][0].cpu())
            predicted_price = current_price * (1 + daily_pred)
            
            # Calculate confidence
            confidence = self._calculate_confidence(predictions)
            
            # Skip if confidence is too low
            if confidence < self.confidence_threshold:
                return None
            
            # Determine signal type
            expected_return = daily_pred
            
            if expected_return > self.buy_threshold:
                signal_type = SignalType.BUY
            elif expected_return < self.sell_threshold:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Skip HOLD signals for active trading
            if signal_type == SignalType.HOLD:
                return None
            
            # Determine signal strength
            strength = self._determine_signal_strength(expected_return, confidence)
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                price_target=predicted_price,
                current_price=current_price,
                timestamp=datetime.now(),
                horizon="1d"
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def generate_signals_batch(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate signals for multiple stocks"""
        signals = []
        
        self.logger.info(f"Generating signals for {len(symbols)} symbols")
        
        for i, symbol in enumerate(symbols):
            try:
                signal = self.generate_signal(symbol)
                if signal:
                    signals.append(signal)
                
                # Progress logging
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(symbols)} symbols, generated {len(signals)} signals")
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        self.logger.info(f"Generated {len(signals)} signals from {len(symbols)} symbols")
        return signals
    
    def filter_signals(self, signals: List[TradingSignal], 
                      min_confidence: float = None,
                      min_expected_return: float = None,
                      max_signals: int = None) -> List[TradingSignal]:
        """Filter and rank signals based on various criteria"""
        filtered_signals = signals.copy()
        
        # Filter by minimum confidence
        if min_confidence:
            filtered_signals = [s for s in filtered_signals if s.confidence >= min_confidence]
        
        # Filter by minimum expected return
        if min_expected_return:
            filtered_signals = [s for s in filtered_signals if abs(s.expected_return) >= min_expected_return]
        
        # Sort by composite score (confidence * abs(expected_return))
        filtered_signals.sort(
            key=lambda s: s.confidence * abs(s.expected_return), 
            reverse=True
        )
        
        # Limit number of signals
        if max_signals:
            filtered_signals = filtered_signals[:max_signals]
        
        return filtered_signals
    
    def get_signal_summary(self, signals: List[TradingSignal]) -> Dict:
        """Generate summary statistics for signals"""
        if not signals:
            return {}
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        strong_signals = [s for s in signals if s.strength == SignalStrength.STRONG]
        medium_signals = [s for s in signals if s.strength == SignalStrength.MEDIUM]
        weak_signals = [s for s in signals if s.strength == SignalStrength.WEAK]
        
        summary = {
            'total_signals': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'strong_signals': len(strong_signals),
            'medium_signals': len(medium_signals),
            'weak_signals': len(weak_signals),
            'avg_confidence': np.mean([s.confidence for s in signals]),
            'avg_expected_return': np.mean([s.expected_return for s in signals]),
            'avg_expected_return_buy': np.mean([s.expected_return for s in buy_signals]) if buy_signals else 0,
            'avg_expected_return_sell': np.mean([s.expected_return for s in sell_signals]) if sell_signals else 0,
            'top_signals': [s.to_dict() for s in signals[:10]]  # Top 10 signals
        }
        
        return summary
    
    def save_signals(self, signals: List[TradingSignal], output_path: str = None):
        """Save signals to database and/or file"""
        if not signals:
            return
        
        # Convert to DataFrame
        signals_data = [s.to_dict() for s in signals]
        df = pd.DataFrame(signals_data)
        
        # Save to file if path provided
        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Signals saved to {output_path}")
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create table if not exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        strength TEXT NOT NULL,
                        confidence REAL,
                        price_target REAL,
                        current_price REAL,
                        expected_return REAL,
                        risk_reward_ratio REAL,
                        timestamp TEXT,
                        horizon TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert signals
                df.to_sql('trading_signals', conn, if_exists='append', index=False)
                
            self.logger.info(f"Saved {len(signals)} signals to database")
            
        except Exception as e:
            self.logger.error(f"Error saving signals to database: {e}")
    
    def generate_all_signals(self, output_path: str = None) -> List[TradingSignal]:
        """Generate signals for all available stocks"""
        try:
            # Get available symbols from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT symbol 
                    FROM stock_data 
                    WHERE date >= date('now', '-30 days')
                    ORDER BY symbol
                """)
                symbols = [row[0] for row in cursor.fetchall()]
            
            if not symbols:
                self.logger.warning("No symbols found in database")
                return []
            
            # Generate signals
            all_signals = self.generate_signals_batch(symbols)
            
            # Filter and rank signals
            filtered_signals = self.filter_signals(
                all_signals,
                min_confidence=self.confidence_threshold,
                max_signals=100  # Top 100 signals
            )
            
            # Save signals
            self.save_signals(filtered_signals, output_path)
            
            # Log summary
            summary = self.get_signal_summary(filtered_signals)
            self.logger.info(f"Signal generation summary: {summary}")
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error in generate_all_signals: {e}")
            return []


class SignalValidator:
    """Validate and backtest trading signals"""
    
    def __init__(self, config: Dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def validate_signal_accuracy(self, lookback_days: int = 30) -> Dict:
        """Validate historical signal accuracy"""
        try:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=lookback_days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get historical signals
                signals_df = pd.read_sql_query("""
                    SELECT * FROM trading_signals
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, conn, params=[start_date.isoformat(), end_date.isoformat()])
                
                if signals_df.empty:
                    return {'message': 'No historical signals found'}
                
                # Validate signal outcomes
                validation_results = []
                
                for _, signal_row in signals_df.iterrows():
                    symbol = signal_row['symbol']
                    signal_date = pd.to_datetime(signal_row['timestamp']).date()
                    predicted_return = signal_row['expected_return']
                    
                    # Get actual return
                    actual_return = self._get_actual_return(symbol, signal_date, 1)
                    
                    if actual_return is not None:
                        validation_results.append({
                            'symbol': symbol,
                            'predicted_return': predicted_return,
                            'actual_return': actual_return,
                            'correct_direction': np.sign(predicted_return) == np.sign(actual_return),
                            'absolute_error': abs(predicted_return - actual_return)
                        })
                
                # Calculate validation metrics
                if validation_results:
                    results_df = pd.DataFrame(validation_results)
                    
                    return {
                        'total_signals': len(results_df),
                        'directional_accuracy': results_df['correct_direction'].mean(),
                        'mean_absolute_error': results_df['absolute_error'].mean(),
                        'correlation': np.corrcoef(results_df['predicted_return'], results_df['actual_return'])[0, 1]
                    }
                
                return {'message': 'No validation results available'}
                
        except Exception as e:
            self.logger.error(f"Error validating signals: {e}")
            return {'error': str(e)}
    
    def _get_actual_return(self, symbol: str, signal_date, horizon_days: int = 1) -> Optional[float]:
        """Get actual return for a stock after signal date"""
        try:
            end_date = signal_date + timedelta(days=horizon_days + 5)  # Add buffer for weekends
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT date, close FROM stock_data
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    ORDER BY date
                """, conn, params=[symbol, signal_date.isoformat(), end_date.isoformat()])
                
                if len(df) >= 2:
                    initial_price = df.iloc[0]['close']
                    final_price = df.iloc[horizon_days]['close'] if len(df) > horizon_days else df.iloc[-1]['close']
                    return (final_price / initial_price) - 1
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting actual return for {symbol}: {e}")
            return None
