"""
Stock Price Forecasting Module
Multi-timeframe price predictions and forecast analysis
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass

from ..models.tft import TemporalFusionTransformer
from ..data.processors import DataProcessor
from ..data.features import FeatureEngineer


@dataclass
class PriceForecast:
    """Data class for price forecasts"""
    symbol: str
    current_price: float
    forecast_date: datetime
    horizon: str  # '1d', '5d', '22d'
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float
    expected_return: float
    volatility_forecast: float
    
    def to_dict(self) -> Dict:
        """Convert forecast to dictionary"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'forecast_date': self.forecast_date.isoformat(),
            'horizon': self.horizon,
            'predicted_price': self.predicted_price,
            'confidence_lower': self.confidence_lower,
            'confidence_upper': self.confidence_upper,
            'confidence_level': self.confidence_level,
            'expected_return': self.expected_return,
            'volatility_forecast': self.volatility_forecast
        }


class ForecastGenerator:
    """Generate price forecasts using TFT model"""
    
    def __init__(self, config: Dict, model_path: str):
        self.config = config
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # Forecasting parameters
        self.forecast_config = config['prediction']['forecasting']
        self.prediction_horizons = config['model']['tft']['prediction_horizons']
        self.quantiles = config['model']['tft']['quantiles']
        
        # Initialize components
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.db_path = config['data']['database']['path']
        
        # Load model
        self._load_model()
        
        # Initialize data processors
        self.data_processor = DataProcessor(config, self.db_path)
        self.feature_engineer = FeatureEngineer(config, self.db_path)
        
        # Horizon mapping
        self.horizon_map = {
            1: '1d',
            5: '5d', 
            22: '22d'
        }
    
    def _load_model(self):
        """Load the trained TFT model"""
        try:
            self.model = TemporalFusionTransformer(self.config).to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.logger.info(f"Forecasting model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _prepare_forecast_data(self, symbol: str, end_date: str = None) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Prepare data for forecasting"""
        try:
            # Get stock data
            stock_data = self.data_processor.get_stock_data_from_db(symbol, end_date=end_date)
            
            if stock_data.empty or len(stock_data) < self.config['model']['tft']['lookback_window']:
                return None
            
            # Get current price
            current_price = float(stock_data['close'].iloc[-1])
            
            # Add features
            stock_data_with_features = self.feature_engineer.engineer_features(stock_data, symbol)
            
            # Process data
            processed_data = self.data_processor.process_single_stock(symbol, end_date=end_date)
            
            if not processed_data or len(processed_data['X']) == 0:
                return None
            
            # Get the most recent sequence
            X = torch.FloatTensor(processed_data['X'][-1:]).to(self.device)
            
            # Static features (simplified)
            static_features = torch.FloatTensor([[hash(symbol) % 1000 / 1000.0] * 10]).to(self.device)
            
            return static_features, X, current_price
            
        except Exception as e:
            self.logger.error(f"Error preparing forecast data for {symbol}: {e}")
            return None
    
    def _calculate_confidence_intervals(self, quantile_predictions: torch.Tensor) -> Tuple[float, float]:
        """Calculate confidence intervals from quantile predictions"""
        # Assuming quantiles are [0.1, 0.5, 0.9]
        if len(quantile_predictions) >= 3:
            lower = float(quantile_predictions[0])  # 10th percentile
            upper = float(quantile_predictions[2])  # 90th percentile
            return lower, upper
        else:
            # Fallback: use standard deviation estimate
            median = float(quantile_predictions[len(quantile_predictions)//2])
            std = abs(median) * 0.2  # Rough estimate
            return median - 1.645*std, median + 1.645*std  # 90% confidence interval
    
    def _estimate_volatility(self, predictions: Dict[str, torch.Tensor], current_price: float) -> float:
        """Estimate volatility from prediction uncertainty"""
        try:
            # Use prediction dispersion as volatility proxy
            price_predictions = []
            for task in ['daily_price', 'weekly_price', 'monthly_price']:
                if task in predictions:
                    price_predictions.append(float(predictions[task][0]))
            
            if len(price_predictions) > 1:
                # Calculate coefficient of variation
                pred_prices = [current_price * (1 + ret) for ret in price_predictions]
                volatility = np.std(pred_prices) / np.mean(pred_prices)
                return min(volatility, 0.5)  # Cap at 50%
            
            return 0.2  # Default volatility estimate
            
        except Exception as e:
            self.logger.warning(f"Error estimating volatility: {e}")
            return 0.2
    
    def generate_forecast(self, symbol: str, current_price: float = None) -> List[PriceForecast]:
        """Generate forecasts for all horizons for a single stock"""
        try:
            # Prepare data
            data = self._prepare_forecast_data(symbol)
            if data is None:
                return []
            
            static_features, dynamic_features, data_current_price = data
            
            # Use provided current price or data price
            if current_price is None:
                current_price = data_current_price
            
            # Generate predictions
            with torch.no_grad():
                predictions = self.model(static_features, dynamic_features)
            
            forecasts = []
            
            # Generate forecast for each horizon
            for i, horizon_days in enumerate(self.prediction_horizons):
                try:
                    horizon_str = self.horizon_map.get(horizon_days, f'{horizon_days}d')
                    
                    # Get prediction for this horizon
                    if f'{horizon_str.replace("d", "_")}_price' in predictions:
                        task_key = f'{horizon_str.replace("d", "_")}_price'
                    else:
                        # Fallback to daily prediction
                        task_key = 'daily_price'
                    
                    pred_return = float(predictions[task_key][0])
                    predicted_price = current_price * (1 + pred_return)
                    
                    # Calculate confidence intervals (simplified)
                    # In practice, you would use quantile predictions
                    volatility = self._estimate_volatility(predictions, current_price)
                    confidence_lower = predicted_price * (1 - 1.645 * volatility)
                    confidence_upper = predicted_price * (1 + 1.645 * volatility)
                    
                    # Create forecast
                    forecast = PriceForecast(
                        symbol=symbol,
                        current_price=current_price,
                        forecast_date=datetime.now() + timedelta(days=horizon_days),
                        horizon=horizon_str,
                        predicted_price=predicted_price,
                        confidence_lower=confidence_lower,
                        confidence_upper=confidence_upper,
                        confidence_level=0.9,
                        expected_return=pred_return,
                        volatility_forecast=volatility
                    )
                    
                    forecasts.append(forecast)
                    
                except Exception as e:
                    self.logger.error(f"Error generating forecast for {symbol} horizon {horizon_days}: {e}")
                    continue
            
            return forecasts
            
        except Exception as e:
            self.logger.error(f"Error generating forecasts for {symbol}: {e}")
            return []
    
    def generate_forecasts_batch(self, symbols: List[str]) -> Dict[str, List[PriceForecast]]:
        """Generate forecasts for multiple stocks"""
        all_forecasts = {}
        
        self.logger.info(f"Generating forecasts for {len(symbols)} symbols")
        
        for i, symbol in enumerate(symbols):
            try:
                forecasts = self.generate_forecast(symbol)
                if forecasts:
                    all_forecasts[symbol] = forecasts
                
                # Progress logging
                if (i + 1) % 20 == 0:
                    self.logger.info(f"Generated forecasts for {i + 1}/{len(symbols)} symbols")
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        self.logger.info(f"Generated forecasts for {len(all_forecasts)} symbols")
        return all_forecasts
    
    def rank_stocks_by_forecast(self, forecasts: Dict[str, List[PriceForecast]], 
                               horizon: str = '5d', 
                               min_expected_return: float = 0.02) -> List[Dict]:
        """Rank stocks by forecast attractiveness"""
        stock_rankings = []
        
        for symbol, symbol_forecasts in forecasts.items():
            # Find forecast for specified horizon
            horizon_forecast = None
            for forecast in symbol_forecasts:
                if forecast.horizon == horizon:
                    horizon_forecast = forecast
                    break
            
            if horizon_forecast is None:
                continue
            
            # Filter by minimum expected return
            if horizon_forecast.expected_return < min_expected_return:
                continue
            
            # Calculate ranking score
            risk_adjusted_return = horizon_forecast.expected_return / max(horizon_forecast.volatility_forecast, 0.01)
            confidence_width = (horizon_forecast.confidence_upper - horizon_forecast.confidence_lower) / horizon_forecast.predicted_price
            uncertainty_penalty = 1 - (confidence_width / 0.5)  # Penalize wide confidence intervals
            
            ranking_score = risk_adjusted_return * uncertainty_penalty
            
            stock_rankings.append({
                'symbol': symbol,
                'expected_return': horizon_forecast.expected_return,
                'predicted_price': horizon_forecast.predicted_price,
                'current_price': horizon_forecast.current_price,
                'volatility_forecast': horizon_forecast.volatility_forecast,
                'risk_adjusted_return': risk_adjusted_return,
                'ranking_score': ranking_score,
                'confidence_lower': horizon_forecast.confidence_lower,
                'confidence_upper': horizon_forecast.confidence_upper,
                'horizon': horizon
            })
        
        # Sort by ranking score
        stock_rankings.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        return stock_rankings
    
    def get_top_picks(self, symbols: List[str] = None, horizon: str = '5d', 
                     top_n: int = None) -> List[Dict]:
        """Get top stock picks based on forecasts"""
        if top_n is None:
            top_n = self.forecast_config['top_picks']
        
        if symbols is None:
            # Get all available symbols
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT symbol 
                    FROM stock_data 
                    WHERE date >= date('now', '-7 days')
                    ORDER BY symbol
                """)
                symbols = [row[0] for row in cursor.fetchall()]
        
        # Generate forecasts
        all_forecasts = self.generate_forecasts_batch(symbols)
        
        # Rank stocks
        rankings = self.rank_stocks_by_forecast(all_forecasts, horizon)
        
        # Apply sector diversification if enabled
        if self.forecast_config['sector_diversification']:
            rankings = self._apply_sector_diversification(rankings)
        
        # Apply volume filter
        min_volume = self.forecast_config['min_volume']
        if min_volume > 0:
            rankings = self._filter_by_volume(rankings, min_volume)
        
        return rankings[:top_n]
    
    def _apply_sector_diversification(self, rankings: List[Dict]) -> List[Dict]:
        """Apply sector diversification to stock rankings"""
        try:
            # Get sector information for stocks
            with sqlite3.connect(self.db_path) as conn:
                sector_df = pd.read_sql_query("""
                    SELECT symbol, sector FROM stock_metadata
                    WHERE symbol IN ({})
                """.format(','.join(['?' for _ in range(len(rankings))])), 
                conn, params=[r['symbol'] for r in rankings])
            
            # Create sector mapping
            sector_map = dict(zip(sector_df['symbol'], sector_df['sector']))
            
            # Group by sector and limit selections
            diversified_rankings = []
            sector_counts = {}
            max_per_sector = 5  # Maximum 5 stocks per sector
            
            for ranking in rankings:
                symbol = ranking['symbol']
                sector = sector_map.get(symbol, 'Unknown')
                
                if sector_counts.get(sector, 0) < max_per_sector:
                    diversified_rankings.append(ranking)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            return diversified_rankings
            
        except Exception as e:
            self.logger.warning(f"Error applying sector diversification: {e}")
            return rankings
    
    def _filter_by_volume(self, rankings: List[Dict], min_volume: int) -> List[Dict]:
        """Filter stocks by minimum volume"""
        try:
            # Get recent volume data
            with sqlite3.connect(self.db_path) as conn:
                volume_df = pd.read_sql_query("""
                    SELECT symbol, AVG(volume) as avg_volume
                    FROM stock_data
                    WHERE date >= date('now', '-30 days')
                    GROUP BY symbol
                    HAVING AVG(volume) >= ?
                """, conn, params=[min_volume])
            
            high_volume_symbols = set(volume_df['symbol'])
            
            return [r for r in rankings if r['symbol'] in high_volume_symbols]
            
        except Exception as e:
            self.logger.warning(f"Error filtering by volume: {e}")
            return rankings
    
    def save_forecasts(self, forecasts: Dict[str, List[PriceForecast]], output_path: str = None):
        """Save forecasts to database and/or file"""
        if not forecasts:
            return
        
        # Flatten forecasts to list
        all_forecasts = []
        for symbol, symbol_forecasts in forecasts.items():
            for forecast in symbol_forecasts:
                all_forecasts.append(forecast.to_dict())
        
        # Convert to DataFrame
        df = pd.DataFrame(all_forecasts)
        
        # Save to file if path provided
        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Forecasts saved to {output_path}")
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create table if not exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS price_forecasts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        current_price REAL,
                        forecast_date TEXT,
                        horizon TEXT,
                        predicted_price REAL,
                        confidence_lower REAL,
                        confidence_upper REAL,
                        confidence_level REAL,
                        expected_return REAL,
                        volatility_forecast REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert forecasts
                df.to_sql('price_forecasts', conn, if_exists='append', index=False)
                
            self.logger.info(f"Saved {len(all_forecasts)} forecasts to database")
            
        except Exception as e:
            self.logger.error(f"Error saving forecasts to database: {e}")
    
    def generate_forecast_report(self, forecasts: Dict[str, List[PriceForecast]], 
                               rankings: List[Dict] = None) -> Dict:
        """Generate comprehensive forecast report"""
        if not forecasts:
            return {}
        
        # Calculate summary statistics
        all_forecasts = []
        for symbol_forecasts in forecasts.values():
            all_forecasts.extend(symbol_forecasts)
        
        # Group by horizon
        horizon_stats = {}
        for horizon in ['1d', '5d', '22d']:
            horizon_forecasts = [f for f in all_forecasts if f.horizon == horizon]
            
            if horizon_forecasts:
                returns = [f.expected_return for f in horizon_forecasts]
                volatilities = [f.volatility_forecast for f in horizon_forecasts]
                
                horizon_stats[horizon] = {
                    'count': len(horizon_forecasts),
                    'avg_expected_return': np.mean(returns),
                    'median_expected_return': np.median(returns),
                    'std_expected_return': np.std(returns),
                    'avg_volatility': np.mean(volatilities),
                    'positive_forecasts': sum(1 for r in returns if r > 0),
                    'negative_forecasts': sum(1 for r in returns if r < 0)
                }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(forecasts),
            'total_forecasts': len(all_forecasts),
            'horizon_statistics': horizon_stats,
        }
        
        if rankings:
            report['top_picks'] = rankings[:10]
            report['avg_top_10_return'] = np.mean([r['expected_return'] for r in rankings[:10]])
        
        return report


class ForecastValidator:
    """Validate forecast accuracy"""
    
    def __init__(self, config: Dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def validate_forecast_accuracy(self, lookback_days: int = 60) -> Dict:
        """Validate historical forecast accuracy"""
        try:
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=lookback_days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get historical forecasts
                forecasts_df = pd.read_sql_query("""
                    SELECT * FROM price_forecasts
                    WHERE created_at BETWEEN ? AND ?
                    AND forecast_date <= ?
                    ORDER BY created_at
                """, conn, params=[start_date.isoformat(), end_date.isoformat(), end_date.isoformat()])
                
                if forecasts_df.empty:
                    return {'message': 'No historical forecasts found for validation'}
                
                # Validate forecasts
                validation_results = []
                
                for _, forecast_row in forecasts_df.iterrows():
                    symbol = forecast_row['symbol']
                    forecast_date = pd.to_datetime(forecast_row['forecast_date']).date()
                    predicted_price = forecast_row['predicted_price']
                    confidence_lower = forecast_row['confidence_lower']
                    confidence_upper = forecast_row['confidence_upper']
                    horizon = forecast_row['horizon']
                    
                    # Get actual price
                    actual_price = self._get_actual_price(symbol, forecast_date)
                    
                    if actual_price is not None:
                        # Calculate errors
                        absolute_error = abs(predicted_price - actual_price)
                        relative_error = absolute_error / actual_price
                        
                        # Check if actual price is within confidence interval
                        within_confidence = confidence_lower <= actual_price <= confidence_upper
                        
                        validation_results.append({
                            'symbol': symbol,
                            'horizon': horizon,
                            'predicted_price': predicted_price,
                            'actual_price': actual_price,
                            'absolute_error': absolute_error,
                            'relative_error': relative_error,
                            'within_confidence': within_confidence
                        })
                
                if validation_results:
                    results_df = pd.DataFrame(validation_results)
                    
                    # Calculate metrics by horizon
                    horizon_metrics = {}
                    for horizon in results_df['horizon'].unique():
                        horizon_data = results_df[results_df['horizon'] == horizon]
                        
                        horizon_metrics[horizon] = {
                            'count': len(horizon_data),
                            'mean_absolute_error': horizon_data['absolute_error'].mean(),
                            'mean_relative_error': horizon_data['relative_error'].mean(),
                            'confidence_coverage': horizon_data['within_confidence'].mean(),
                            'rmse': np.sqrt((horizon_data['absolute_error'] ** 2).mean())
                        }
                    
                    return {
                        'total_validations': len(results_df),
                        'overall_mean_absolute_error': results_df['absolute_error'].mean(),
                        'overall_mean_relative_error': results_df['relative_error'].mean(),
                        'overall_confidence_coverage': results_df['within_confidence'].mean(),
                        'horizon_metrics': horizon_metrics
                    }
                
                return {'message': 'No validation results available'}
                
        except Exception as e:
            self.logger.error(f"Error validating forecasts: {e}")
            return {'error': str(e)}
    
    def _get_actual_price(self, symbol: str, target_date) -> Optional[float]:
        """Get actual price for a stock on a specific date"""
        try:
            # Allow some flexibility for weekends/holidays
            start_date = target_date - timedelta(days=2)
            end_date = target_date + timedelta(days=2)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT date, close FROM stock_data
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                    ORDER BY ABS(julianday(date) - julianday(?))
                    LIMIT 1
                """, conn, params=[symbol, start_date.isoformat(), end_date.isoformat(), target_date.isoformat()])
                
                if not df.empty:
                    return float(df.iloc[0]['close'])
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting actual price for {symbol} on {target_date}: {e}")
            return None
