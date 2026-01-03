"""
Portfolio Optimization Module
Risk-adjusted portfolio construction and optimization using TFT predictions
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import linalg
from dataclasses import dataclass

from .forecasting import ForecastGenerator, PriceForecast
from .signals import TradingSignal


@dataclass
class PortfolioAllocation:
    """Data class for portfolio allocation"""
    symbol: str
    weight: float
    expected_return: float
    current_price: float
    target_price: float
    shares: int
    allocation_value: float
    sector: str = None
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'weight': self.weight,
            'expected_return': self.expected_return,
            'current_price': self.current_price,
            'target_price': self.target_price,
            'shares': self.shares,
            'allocation_value': self.allocation_value,
            'sector': self.sector
        }


@dataclass
class Portfolio:
    """Portfolio class containing allocations and performance metrics"""
    allocations: List[PortfolioAllocation]
    total_value: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    diversification_score: float
    creation_date: datetime
    
    def to_dict(self) -> Dict:
        return {
            'allocations': [alloc.to_dict() for alloc in self.allocations],
            'total_value': self.total_value,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_estimate': self.max_drawdown_estimate,
            'diversification_score': self.diversification_score,
            'creation_date': self.creation_date.isoformat(),
            'num_positions': len(self.allocations)
        }


class RiskModel:
    """Risk modeling and correlation estimation"""
    
    def __init__(self, config: Dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def estimate_covariance_matrix(self, symbols: List[str], lookback_days: int = 252) -> np.ndarray:
        """Estimate covariance matrix for given symbols"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days + 50)  # Extra buffer for missing dates
            
            # Get returns for all symbols
            returns_data = {}
            
            with sqlite3.connect(self.db_path) as conn:
                for symbol in symbols:
                    df = pd.read_sql_query("""
                        SELECT date, close FROM stock_data
                        WHERE symbol = ? AND date BETWEEN ? AND ?
                        ORDER BY date
                    """, conn, params=[symbol, start_date.isoformat(), end_date.isoformat()])
                    
                    if len(df) > 20:  # Minimum data requirement
                        df['returns'] = df['close'].pct_change()
                        df = df.dropna()
                        returns_data[symbol] = df.set_index('date')['returns']
            
            if len(returns_data) < 2:
                # Return identity matrix if insufficient data
                n = len(symbols)
                return np.eye(n) * 0.04  # Assume 20% annual volatility
            
            # Align returns data
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 60:  # Minimum observations
                n = len(symbols)
                return np.eye(n) * 0.04
            
            # Calculate covariance matrix (annualized)
            cov_matrix = returns_df.cov().values * 252  # Annualize
            
            # Ensure positive semi-definite
            eigenvals, eigenvecs = linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)
            cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            return cov_matrix
            
        except Exception as e:
            self.logger.error(f"Error estimating covariance matrix: {e}")
            # Return diagonal matrix as fallback
            n = len(symbols)
            return np.eye(n) * 0.04
    
    def estimate_beta(self, symbol: str, market_symbol: str = '^NSEI', lookback_days: int = 252) -> float:
        """Estimate stock beta relative to market"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days + 50)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get stock returns
                stock_df = pd.read_sql_query("""
                    SELECT date, close FROM stock_data
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                """, conn, params=[symbol, start_date.isoformat(), end_date.isoformat()])
                
                # Get market returns
                market_df = pd.read_sql_query("""
                    SELECT date, close FROM market_indices
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                """, conn, params=[market_symbol, start_date.isoformat(), end_date.isoformat()])
                
                if len(stock_df) < 60 or len(market_df) < 60:
                    return 1.0  # Default beta
                
                # Calculate returns
                stock_df['returns'] = stock_df['close'].pct_change()
                market_df['returns'] = market_df['close'].pct_change()
                
                # Merge on date
                merged = pd.merge(stock_df[['date', 'returns']], 
                                market_df[['date', 'returns']], 
                                on='date', suffixes=('_stock', '_market'))
                merged = merged.dropna()
                
                if len(merged) < 60:
                    return 1.0
                
                # Calculate beta
                covariance = np.cov(merged['returns_stock'], merged['returns_market'])[0, 1]
                market_variance = np.var(merged['returns_market'])
                
                beta = covariance / market_variance if market_variance > 0 else 1.0
                
                # Constrain beta to reasonable range
                return max(0.1, min(beta, 3.0))
                
        except Exception as e:
            self.logger.warning(f"Error estimating beta for {symbol}: {e}")
            return 1.0


class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Portfolio constraints
        self.portfolio_config = config['prediction']['portfolio']
        self.max_positions = self.portfolio_config['max_positions']
        self.max_sector_weight = self.portfolio_config['max_sector_weight']
        self.risk_free_rate = self.portfolio_config['risk_free_rate']
        self.target_volatility = self.portfolio_config['target_volatility']
        
        # Initialize risk model
        self.risk_model = RiskModel(config, config['data']['database']['path'])
        self.db_path = config['data']['database']['path']
    
    def mean_variance_optimization(self, expected_returns: np.ndarray, 
                                 covariance_matrix: np.ndarray,
                                 risk_aversion: float = 1.0) -> np.ndarray:
        """Mean-variance optimization to find optimal weights"""
        n = len(expected_returns)
        
        # Objective function: minimize -return + risk_aversion * risk
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            return -portfolio_return + risk_aversion * portfolio_risk
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (0 to 0.3 for individual weights to ensure diversification)
        bounds = [(0, 0.3) for _ in range(n)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            self.logger.warning("Optimization failed, using equal weights")
            return initial_weights
    
    def risk_parity_optimization(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Risk parity optimization - equal risk contribution"""
        n = covariance_matrix.shape[0]
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target_risk = np.ones(n) / n
            return np.sum((contrib / np.sum(contrib) - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        # Bounds
        bounds = [(0.01, 0.5) for _ in range(n)]
        
        # Initial guess
        initial_weights = np.ones(n) / n
        
        # Optimize
        result = minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            return initial_weights
    
    def apply_sector_constraints(self, weights: np.ndarray, symbols: List[str]) -> np.ndarray:
        """Apply sector diversification constraints"""
        try:
            # Get sector information
            with sqlite3.connect(self.db_path) as conn:
                sector_df = pd.read_sql_query("""
                    SELECT symbol, COALESCE(sector, 'Unknown') as sector 
                    FROM stock_metadata
                    WHERE symbol IN ({})
                """.format(','.join(['?' for _ in symbols])), conn, params=symbols)
            
            sector_map = dict(zip(sector_df['symbol'], sector_df['sector']))
            
            # Group weights by sector
            sector_weights = {}
            for i, symbol in enumerate(symbols):
                sector = sector_map.get(symbol, 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]
            
            # Scale down overweight sectors
            adjusted_weights = weights.copy()
            for sector, total_weight in sector_weights.items():
                if total_weight > self.max_sector_weight:
                    # Find symbols in this sector
                    sector_indices = [i for i, symbol in enumerate(symbols) 
                                    if sector_map.get(symbol, 'Unknown') == sector]
                    
                    # Scale down proportionally
                    scale_factor = self.max_sector_weight / total_weight
                    for idx in sector_indices:
                        adjusted_weights[idx] *= scale_factor
            
            # Renormalize
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.warning(f"Error applying sector constraints: {e}")
            return weights
    
    def create_portfolio(self, forecasts: Dict[str, List[PriceForecast]], 
                        signals: List[TradingSignal] = None,
                        portfolio_value: float = 100000,
                        optimization_method: str = 'mean_variance') -> Portfolio:
        """Create optimized portfolio from forecasts and signals"""
        try:
            self.logger.info("Creating optimized portfolio...")
            
            # Select stocks for portfolio
            candidate_stocks = self._select_candidate_stocks(forecasts, signals)
            
            if len(candidate_stocks) < 2:
                self.logger.warning("Insufficient candidate stocks for portfolio creation")
                return None
            
            # Limit to max positions
            candidate_stocks = candidate_stocks[:self.max_positions]
            symbols = [stock['symbol'] for stock in candidate_stocks]
            
            self.logger.info(f"Optimizing portfolio with {len(symbols)} stocks")
            
            # Extract expected returns
            expected_returns = np.array([stock['expected_return'] for stock in candidate_stocks])
            
            # Estimate covariance matrix
            covariance_matrix = self.risk_model.estimate_covariance_matrix(symbols)
            
            # Optimize weights
            if optimization_method == 'mean_variance':
                weights = self.mean_variance_optimization(expected_returns, covariance_matrix)
            elif optimization_method == 'risk_parity':
                weights = self.risk_parity_optimization(covariance_matrix)
            else:
                weights = np.ones(len(symbols)) / len(symbols)  # Equal weights
            
            # Apply sector constraints
            weights = self.apply_sector_constraints(weights, symbols)
            
            # Create portfolio allocations
            allocations = []
            total_allocated_value = 0
            
            for i, (symbol, weight) in enumerate(zip(symbols, weights)):
                stock_info = candidate_stocks[i]
                
                allocation_value = portfolio_value * weight
                shares = int(allocation_value / stock_info['current_price'])
                actual_allocation_value = shares * stock_info['current_price']
                
                allocation = PortfolioAllocation(
                    symbol=symbol,
                    weight=weight,
                    expected_return=stock_info['expected_return'],
                    current_price=stock_info['current_price'],
                    target_price=stock_info['target_price'],
                    shares=shares,
                    allocation_value=actual_allocation_value,
                    sector=stock_info.get('sector', 'Unknown')
                )
                
                allocations.append(allocation)
                total_allocated_value += actual_allocation_value
            
            # Calculate portfolio metrics
            portfolio_expected_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_expected_return - self.risk_free_rate) / portfolio_volatility
            
            # Estimate maximum drawdown (simplified)
            max_drawdown_estimate = portfolio_volatility * 2  # Rough estimate
            
            # Calculate diversification score
            diversification_score = self._calculate_diversification_score(weights, covariance_matrix)
            
            portfolio = Portfolio(
                allocations=allocations,
                total_value=total_allocated_value,
                expected_return=portfolio_expected_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_estimate=max_drawdown_estimate,
                diversification_score=diversification_score,
                creation_date=datetime.now()
            )
            
            self.logger.info(f"Portfolio created: {len(allocations)} positions, "
                           f"Expected Return: {portfolio_expected_return:.2%}, "
                           f"Volatility: {portfolio_volatility:.2%}, "
                           f"Sharpe: {sharpe_ratio:.3f}")
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio: {e}")
            return None
    
    def _select_candidate_stocks(self, forecasts: Dict[str, List[PriceForecast]], 
                               signals: List[TradingSignal] = None) -> List[Dict]:
        """Select candidate stocks for portfolio optimization"""
        candidates = []
        
        # Use signals if available, otherwise use forecasts
        if signals:
            signal_map = {s.symbol: s for s in signals}
        else:
            signal_map = {}
        
        for symbol, symbol_forecasts in forecasts.items():
            # Find 5-day forecast (weekly)
            forecast_5d = None
            for forecast in symbol_forecasts:
                if forecast.horizon == '5d':
                    forecast_5d = forecast
                    break
            
            if forecast_5d is None:
                continue
            
            # Minimum return threshold
            if forecast_5d.expected_return < 0.01:  # 1% minimum
                continue
            
            # Use signal confidence if available
            confidence = 1.0
            if symbol in signal_map:
                confidence = signal_map[symbol].confidence
            
            # Get sector information
            sector = self._get_stock_sector(symbol)
            
            candidates.append({
                'symbol': symbol,
                'expected_return': forecast_5d.expected_return,
                'current_price': forecast_5d.current_price,
                'target_price': forecast_5d.predicted_price,
                'volatility': forecast_5d.volatility_forecast,
                'confidence': confidence,
                'sector': sector,
                'risk_adjusted_return': forecast_5d.expected_return / max(forecast_5d.volatility_forecast, 0.01)
            })
        
        # Sort by risk-adjusted return
        candidates.sort(key=lambda x: x['risk_adjusted_return'] * x['confidence'], reverse=True)
        
        return candidates
    
    def _get_stock_sector(self, symbol: str) -> str:
        """Get stock sector information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT sector FROM stock_metadata WHERE symbol = ?", (symbol,))
                result = cursor.fetchone()
                return result[0] if result and result[0] else 'Unknown'
        except:
            return 'Unknown'
    
    def _calculate_diversification_score(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """Calculate portfolio diversification score (0-1, higher is better)"""
        try:
            # Portfolio variance
            portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
            
            # Weighted average of individual variances
            individual_vars = np.diag(covariance_matrix)
            weighted_avg_var = np.dot(weights**2, individual_vars)
            
            # Diversification ratio
            diversification_ratio = 1 - (portfolio_var / weighted_avg_var)
            
            return max(0, min(1, diversification_ratio))
            
        except Exception as e:
            self.logger.warning(f"Error calculating diversification score: {e}")
            return 0.5
    
    def rebalance_portfolio(self, current_portfolio: Portfolio, 
                          new_forecasts: Dict[str, List[PriceForecast]],
                          rebalance_threshold: float = 0.05) -> Optional[Portfolio]:
        """Rebalance existing portfolio based on new forecasts"""
        try:
            # Get current symbols and weights
            current_symbols = [alloc.symbol for alloc in current_portfolio.allocations]
            current_weights = np.array([alloc.weight for alloc in current_portfolio.allocations])
            
            # Update expected returns with new forecasts
            updated_returns = []
            for symbol in current_symbols:
                if symbol in new_forecasts:
                    # Find 5-day forecast
                    forecast_5d = None
                    for forecast in new_forecasts[symbol]:
                        if forecast.horizon == '5d':
                            forecast_5d = forecast
                            break
                    
                    if forecast_5d:
                        updated_returns.append(forecast_5d.expected_return)
                    else:
                        # Use existing expected return
                        current_alloc = next(a for a in current_portfolio.allocations if a.symbol == symbol)
                        updated_returns.append(current_alloc.expected_return)
                else:
                    # Use existing expected return
                    current_alloc = next(a for a in current_portfolio.allocations if a.symbol == symbol)
                    updated_returns.append(current_alloc.expected_return)
            
            updated_returns = np.array(updated_returns)
            
            # Re-optimize with updated returns
            covariance_matrix = self.risk_model.estimate_covariance_matrix(current_symbols)
            new_weights = self.mean_variance_optimization(updated_returns, covariance_matrix)
            new_weights = self.apply_sector_constraints(new_weights, current_symbols)
            
            # Check if rebalancing is needed
            weight_changes = np.abs(new_weights - current_weights)
            if np.max(weight_changes) < rebalance_threshold:
                self.logger.info("No rebalancing needed")
                return current_portfolio
            
            self.logger.info(f"Rebalancing portfolio - max weight change: {np.max(weight_changes):.3f}")
            
            # Create new portfolio with updated weights
            new_allocations = []
            total_value = current_portfolio.total_value
            
            for i, symbol in enumerate(current_symbols):
                current_alloc = current_portfolio.allocations[i]
                new_weight = new_weights[i]
                
                allocation_value = total_value * new_weight
                shares = int(allocation_value / current_alloc.current_price)
                actual_allocation_value = shares * current_alloc.current_price
                
                new_allocation = PortfolioAllocation(
                    symbol=symbol,
                    weight=new_weight,
                    expected_return=updated_returns[i],
                    current_price=current_alloc.current_price,
                    target_price=current_alloc.current_price * (1 + updated_returns[i]),
                    shares=shares,
                    allocation_value=actual_allocation_value,
                    sector=current_alloc.sector
                )
                
                new_allocations.append(new_allocation)
            
            # Update portfolio metrics
            portfolio_expected_return = np.dot(new_weights, updated_returns)
            portfolio_volatility = np.sqrt(np.dot(new_weights, np.dot(covariance_matrix, new_weights)))
            sharpe_ratio = (portfolio_expected_return - self.risk_free_rate) / portfolio_volatility
            diversification_score = self._calculate_diversification_score(new_weights, covariance_matrix)
            
            rebalanced_portfolio = Portfolio(
                allocations=new_allocations,
                total_value=sum(a.allocation_value for a in new_allocations),
                expected_return=portfolio_expected_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_estimate=portfolio_volatility * 2,
                diversification_score=diversification_score,
                creation_date=datetime.now()
            )
            
            return rebalanced_portfolio
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            return None
    
    def save_portfolio(self, portfolio: Portfolio, output_path: str = None):
        """Save portfolio to database and/or file"""
        if not portfolio:
            return
        
        # Convert to DataFrame
        allocations_df = pd.DataFrame([alloc.to_dict() for alloc in portfolio.allocations])
        
        # Save to file if path provided
        if output_path:
            portfolio_data = portfolio.to_dict()
            with open(output_path.replace('.csv', '.json'), 'w') as f:
                import json
                json.dump(portfolio_data, f, indent=2)
            
            allocations_df.to_csv(output_path, index=False)
            self.logger.info(f"Portfolio saved to {output_path}")
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create tables if not exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS portfolios (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_value REAL,
                        expected_return REAL,
                        expected_volatility REAL,
                        sharpe_ratio REAL,
                        max_drawdown_estimate REAL,
                        diversification_score REAL,
                        creation_date TEXT,
                        num_positions INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_allocations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id INTEGER,
                        symbol TEXT,
                        weight REAL,
                        expected_return REAL,
                        current_price REAL,
                        target_price REAL,
                        shares INTEGER,
                        allocation_value REAL,
                        sector TEXT,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                    )
                """)
                
                # Insert portfolio
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO portfolios 
                    (total_value, expected_return, expected_volatility, sharpe_ratio,
                     max_drawdown_estimate, diversification_score, creation_date, num_positions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    portfolio.total_value, portfolio.expected_return, portfolio.expected_volatility,
                    portfolio.sharpe_ratio, portfolio.max_drawdown_estimate, portfolio.diversification_score,
                    portfolio.creation_date.isoformat(), len(portfolio.allocations)
                ))
                
                portfolio_id = cursor.lastrowid
                
                # Insert allocations
                for allocation in portfolio.allocations:
                    cursor.execute("""
                        INSERT INTO portfolio_allocations
                        (portfolio_id, symbol, weight, expected_return, current_price,
                         target_price, shares, allocation_value, sector)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        portfolio_id, allocation.symbol, allocation.weight, allocation.expected_return,
                        allocation.current_price, allocation.target_price, allocation.shares,
                        allocation.allocation_value, allocation.sector
                    ))
                
                conn.commit()
                
            self.logger.info(f"Portfolio saved to database with ID: {portfolio_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio to database: {e}")
