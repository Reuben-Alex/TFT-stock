"""
Model Evaluation Module
Comprehensive evaluation metrics and analysis tools for stock prediction models
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from scipy import stats


class FinancialMetrics:
    """Financial performance metrics for stock prediction models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_returns(self, predictions: np.ndarray, actuals: np.ndarray, 
                         threshold: float = 0.01) -> Dict[str, float]:
        """
        Calculate trading returns based on predictions
        
        Args:
            predictions: Model predictions
            actuals: Actual returns
            threshold: Threshold for trading signal
            
        Returns:
            Dictionary of return metrics
        """
        # Generate trading signals
        buy_signals = predictions > threshold
        sell_signals = predictions < -threshold
        
        # Calculate strategy returns
        strategy_returns = np.zeros_like(actuals)
        strategy_returns[buy_signals] = actuals[buy_signals]
        strategy_returns[sell_signals] = -actuals[sell_signals]
        
        # Performance metrics
        total_return = np.sum(strategy_returns)
        annualized_return = np.mean(strategy_returns) * 252  # Assuming daily returns
        volatility = np.std(strategy_returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 6%)
        risk_free_rate = 0.06
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Hit rate (percentage of profitable trades)
        profitable_trades = strategy_returns > 0
        hit_rate = np.mean(profitable_trades) if len(profitable_trades) > 0 else 0
        
        # Win/Loss ratio
        winning_returns = strategy_returns[strategy_returns > 0]
        losing_returns = strategy_returns[strategy_returns < 0]
        
        avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
        avg_loss = np.mean(np.abs(losing_returns)) if len(losing_returns) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'win_loss_ratio': win_loss_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def information_ratio(self, predictions: np.ndarray, actuals: np.ndarray, 
                         benchmark_returns: np.ndarray = None) -> float:
        """Calculate Information Ratio"""
        if benchmark_returns is None:
            benchmark_returns = np.zeros_like(actuals)
        
        active_returns = actuals - benchmark_returns
        prediction_errors = predictions - active_returns
        
        tracking_error = np.std(prediction_errors)
        active_return = np.mean(active_returns)
        
        return active_return / tracking_error if tracking_error > 0 else 0
    
    def calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar Ratio (Annual Return / Max Drawdown)"""
        annual_return = np.mean(returns) * 252
        
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.abs(np.min(drawdown))
        
        return annual_return / max_drawdown if max_drawdown > 0 else 0


class PredictionAnalyzer:
    """Analyze prediction quality and model performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def directional_accuracy(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)"""
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        
        correct_directions = pred_direction == actual_direction
        return np.mean(correct_directions)
    
    def prediction_intervals(self, predictions: np.ndarray, actuals: np.ndarray, 
                           confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate prediction interval coverage"""
        # Calculate prediction errors
        errors = actuals - predictions
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_bound = np.quantile(errors, lower_quantile)
        upper_bound = np.quantile(errors, upper_quantile)
        
        # Check coverage
        within_interval = (errors >= lower_bound) & (errors <= upper_bound)
        coverage = np.mean(within_interval)
        
        return {
            'coverage': coverage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    def prediction_bias(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """Analyze prediction bias"""
        errors = predictions - actuals
        
        # Overall bias
        mean_bias = np.mean(errors)
        median_bias = np.median(errors)
        
        # Conditional bias (for positive/negative predictions)
        pos_pred_mask = predictions > 0
        neg_pred_mask = predictions < 0
        
        pos_bias = np.mean(errors[pos_pred_mask]) if np.any(pos_pred_mask) else 0
        neg_bias = np.mean(errors[neg_pred_mask]) if np.any(neg_pred_mask) else 0
        
        return {
            'mean_bias': mean_bias,
            'median_bias': median_bias,
            'positive_prediction_bias': pos_bias,
            'negative_prediction_bias': neg_bias
        }
    
    def regime_analysis(self, predictions: np.ndarray, actuals: np.ndarray, 
                       market_conditions: np.ndarray = None) -> Dict[str, Dict]:
        """Analyze model performance across different market regimes"""
        if market_conditions is None:
            # Simple regime classification based on volatility
            volatility = np.abs(actuals)
            high_vol_threshold = np.quantile(volatility, 0.75)
            low_vol_threshold = np.quantile(volatility, 0.25)
            
            market_conditions = np.where(volatility > high_vol_threshold, 'high_volatility',
                                       np.where(volatility < low_vol_threshold, 'low_volatility', 'medium_volatility'))
        
        regimes = {}
        for regime in np.unique(market_conditions):
            mask = market_conditions == regime
            regime_preds = predictions[mask]
            regime_actuals = actuals[mask]
            
            if len(regime_preds) > 0:
                regimes[regime] = {
                    'mse': mean_squared_error(regime_actuals, regime_preds),
                    'mae': mean_absolute_error(regime_actuals, regime_preds),
                    'directional_accuracy': self.directional_accuracy(regime_preds, regime_actuals),
                    'correlation': np.corrcoef(regime_preds, regime_actuals)[0, 1] if len(regime_preds) > 1 else 0,
                    'sample_size': len(regime_preds)
                }
        
        return regimes


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.financial_metrics = FinancialMetrics()
        self.prediction_analyzer = PredictionAnalyzer()
    
    def evaluate_predictions(self, predictions: Dict[str, np.ndarray], 
                           actuals: Dict[str, np.ndarray],
                           dates: np.ndarray = None,
                           symbols: List[str] = None) -> Dict[str, Dict]:
        """
        Comprehensive evaluation of model predictions
        
        Args:
            predictions: Dictionary of predictions by task
            actuals: Dictionary of actual values by task
            dates: Optional array of dates
            symbols: Optional list of symbols
            
        Returns:
            Dictionary of evaluation results
        """
        results = {}
        
        for task_name in predictions.keys():
            if task_name not in actuals:
                continue
            
            pred = predictions[task_name]
            actual = actuals[task_name]
            
            self.logger.info(f"Evaluating {task_name}...")
            
            # Basic regression metrics
            mse = mean_squared_error(actual, pred)
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual, pred)
            
            # Correlation
            correlation = np.corrcoef(pred, actual)[0, 1] if len(pred) > 1 else 0
            
            # Directional accuracy
            directional_acc = self.prediction_analyzer.directional_accuracy(pred, actual)
            
            # Prediction intervals
            pred_intervals = self.prediction_analyzer.prediction_intervals(pred, actual)
            
            # Bias analysis
            bias_analysis = self.prediction_analyzer.prediction_bias(pred, actual)
            
            # Financial metrics (if applicable)
            financial_results = {}
            if 'price' in task_name:
                financial_results = self.financial_metrics.calculate_returns(pred, actual)
            
            # Regime analysis
            regime_results = self.prediction_analyzer.regime_analysis(pred, actual)
            
            results[task_name] = {
                'regression_metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'correlation': correlation
                },
                'directional_accuracy': directional_acc,
                'prediction_intervals': pred_intervals,
                'bias_analysis': bias_analysis,
                'financial_metrics': financial_results,
                'regime_analysis': regime_results
            }
        
        return results
    
    def generate_evaluation_report(self, evaluation_results: Dict, 
                                 output_path: str = None) -> pd.DataFrame:
        """Generate a comprehensive evaluation report"""
        report_data = []
        
        for task_name, results in evaluation_results.items():
            row = {'task': task_name}
            
            # Add regression metrics
            for metric, value in results['regression_metrics'].items():
                row[f'reg_{metric}'] = value
            
            # Add directional accuracy
            row['directional_accuracy'] = results['directional_accuracy']
            
            # Add financial metrics
            for metric, value in results['financial_metrics'].items():
                row[f'fin_{metric}'] = value
            
            # Add bias metrics
            for metric, value in results['bias_analysis'].items():
                row[f'bias_{metric}'] = value
            
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            self.logger.info(f"Evaluation report saved to {output_path}")
        
        return report_df
    
    def plot_prediction_analysis(self, predictions: np.ndarray, actuals: np.ndarray, 
                               dates: np.ndarray = None, title: str = "Prediction Analysis",
                               output_path: str = None):
        """Create visualization plots for prediction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot: Predictions vs Actuals
        axes[0, 0].scatter(actuals, predictions, alpha=0.6)
        axes[0, 0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Predictions vs Actuals')
        
        # Residuals plot
        residuals = predictions - actuals
        axes[0, 1].scatter(predictions, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predictions')
        
        # Time series plot (if dates available)
        if dates is not None:
            axes[1, 0].plot(dates, actuals, label='Actual', alpha=0.7)
            axes[1, 0].plot(dates, predictions, label='Predicted', alpha=0.7)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Time Series Comparison')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Distribution of residuals
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Prediction analysis plot saved to {output_path}")
        
        return fig
    
    def feature_importance_analysis(self, model: torch.nn.Module, 
                                  test_loader: torch.utils.data.DataLoader) -> Dict[str, np.ndarray]:
        """Analyze feature importance from TFT model"""
        model.eval()
        
        static_importances = []
        dynamic_importances = []
        
        with torch.no_grad():
            for batch in test_loader:
                static_features = batch['static_features']
                dynamic_features = batch['dynamic_features']
                
                predictions = model(static_features, dynamic_features)
                
                if 'static_weights' in predictions:
                    static_importances.append(predictions['static_weights'].cpu().numpy())
                
                if 'dynamic_weights' in predictions:
                    dynamic_importances.append(predictions['dynamic_weights'].cpu().numpy())
        
        results = {}
        
        if static_importances:
            static_importances = np.concatenate(static_importances, axis=0)
            results['static_importance'] = np.mean(static_importances, axis=(0, 1))
        
        if dynamic_importances:
            dynamic_importances = np.concatenate(dynamic_importances, axis=0)
            results['dynamic_importance'] = np.mean(dynamic_importances, axis=(0, 1))
        
        return results
    
    def backtesting_analysis(self, predictions: np.ndarray, actuals: np.ndarray,
                           dates: pd.DatetimeIndex, initial_capital: float = 100000) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform backtesting analysis with trading simulation
        
        Args:
            predictions: Model predictions
            actuals: Actual returns
            dates: Trading dates
            initial_capital: Initial capital for simulation
            
        Returns:
            Backtesting results
        """
        # Simple momentum strategy based on predictions
        signals = np.sign(predictions)
        
        # Calculate portfolio returns
        portfolio_returns = signals * actuals
        
        # Calculate cumulative performance
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        # Benchmark (buy and hold)
        benchmark_returns = (1 + actuals).cumprod()
        benchmark_value = initial_capital * benchmark_returns
        
        # Performance metrics
        total_return = (portfolio_value[-1] / initial_capital) - 1
        benchmark_return = (benchmark_value[-1] / initial_capital) - 1
        excess_return = total_return - benchmark_return
        
        # Risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        max_dd = self._calculate_max_drawdown(portfolio_value)
        
        # Sharpe ratio
        risk_free_rate = 0.06  # 6% annual risk-free rate
        sharpe_ratio = (np.mean(portfolio_returns) * 252 - risk_free_rate) / volatility
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'portfolio_value': portfolio_value,
            'benchmark_value': benchmark_value
        }
    
    def _calculate_max_drawdown(self, portfolio_value: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - peak) / peak
        return np.min(drawdown)
