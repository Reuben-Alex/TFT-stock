"""
Gradio Dashboard for TFT Stock Prediction System
Interactive web interface for model predictions, signals, and portfolio optimization
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import gradio as gr

from ..prediction.signals import SignalGenerator
from ..prediction.forecasting import ForecastGenerator
from ..prediction.portfolio import PortfolioOptimizer
from ..data.collectors import StockDataCollector


class StockDashboard:
    """Main dashboard class for the TFT stock prediction system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dashboard configuration
        self.dashboard_config = config['interface']['dashboard']
        self.max_stocks_display = self.dashboard_config['max_stocks_display']
        self.default_charts = self.dashboard_config['default_charts']
        
        # Database path
        self.db_path = config['data']['database']['path']
        
        # Initialize components (will be loaded when needed)
        self.signal_generator = None
        self.forecast_generator = None
        self.portfolio_optimizer = None
        
        # Model path
        self.model_path = 'models/tft_best_model.pth'
        
    def _init_components(self):
        """Initialize prediction components if not already loaded"""
        try:
            if os.path.exists(self.model_path):
                if self.signal_generator is None:
                    self.signal_generator = SignalGenerator(self.config, self.model_path)
                
                if self.forecast_generator is None:
                    self.forecast_generator = ForecastGenerator(self.config, self.model_path)
                
                if self.portfolio_optimizer is None:
                    self.portfolio_optimizer = PortfolioOptimizer(self.config)
                    
            else:
                self.logger.warning(f"Model not found at {self.model_path}")
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
    
    def get_available_stocks(self) -> List[str]:
        """Get list of available stocks from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT symbol 
                    FROM stock_data 
                    WHERE date >= date('now', '-30 days')
                    ORDER BY symbol
                    LIMIT 200
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting available stocks: {e}")
            return []
    
    def create_stock_chart(self, symbol: str, days: int = 90) -> go.Figure:
        """Create interactive stock price chart"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT date, open, high, low, close, volume
                    FROM stock_data
                    WHERE symbol = ? AND date >= ?
                    ORDER BY date
                """, conn, params=[symbol, start_date.isoformat()])
            
            if df.empty:
                return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
            
            df['date'] = pd.to_datetime(df['date'])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Stock Price', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name='Volume',
                    marker_color='rgba(0,100,80,0.6)'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"{symbol} - {days} Days",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                template="plotly_white",
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating chart for {symbol}: {e}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
    
    def get_trading_signals_display(self) -> Tuple[pd.DataFrame, str]:
        """Get current trading signals for display"""
        try:
            self._init_components()
            
            if self.signal_generator is None:
                return pd.DataFrame(), "Model not available. Please train the model first."
            
            # Generate fresh signals for top stocks
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT symbol 
                    FROM stock_data 
                    WHERE date >= date('now', '-7 days')
                    ORDER BY symbol
                    LIMIT 50
                """)
                symbols = [row[0] for row in cursor.fetchall()]
            
            if not symbols:
                return pd.DataFrame(), "No recent stock data found."
            
            # Generate signals
            signals = self.signal_generator.generate_signals_batch(symbols[:30])  # Limit for performance
            
            if not signals:
                return pd.DataFrame(), "No trading signals generated."
            
            # Convert to DataFrame for display
            signals_data = []
            for signal in signals:
                signals_data.append({
                    'Symbol': signal.symbol,
                    'Signal': signal.signal_type.name,
                    'Strength': signal.strength.name,
                    'Confidence': f"{signal.confidence:.1%}",
                    'Current Price': f"₹{signal.current_price:.2f}",
                    'Target Price': f"₹{signal.price_target:.2f}",
                    'Expected Return': f"{signal.expected_return:.1%}",
                    'Timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M')
                })
            
            df = pd.DataFrame(signals_data)
            
            # Summary message
            summary = f"Generated {len(signals)} trading signals. "
            summary += f"Buy: {len([s for s in signals if s.signal_type.name == 'BUY'])}, "
            summary += f"Sell: {len([s for s in signals if s.signal_type.name == 'SELL'])}"
            
            return df, summary
            
        except Exception as e:
            self.logger.error(f"Error getting trading signals: {e}")
            return pd.DataFrame(), f"Error: {str(e)}"
    
    def get_price_forecasts_display(self, horizon: str = "5d") -> Tuple[pd.DataFrame, str]:
        """Get price forecasts for display"""
        try:
            self._init_components()
            
            if self.forecast_generator is None:
                return pd.DataFrame(), "Model not available. Please train the model first."
            
            # Get top stocks for forecasting
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT symbol 
                    FROM stock_data 
                    WHERE date >= date('now', '-7 days')
                    ORDER BY symbol
                    LIMIT 30
                """)
                symbols = [row[0] for row in cursor.fetchall()]
            
            if not symbols:
                return pd.DataFrame(), "No recent stock data found."
            
            # Generate forecasts
            all_forecasts = self.forecast_generator.generate_forecasts_batch(symbols[:20])
            
            if not all_forecasts:
                return pd.DataFrame(), "No forecasts generated."
            
            # Extract forecasts for specified horizon
            forecast_data = []
            for symbol, forecasts in all_forecasts.items():
                for forecast in forecasts:
                    if forecast.horizon == horizon:
                        forecast_data.append({
                            'Symbol': symbol,
                            'Current Price': f"₹{forecast.current_price:.2f}",
                            'Predicted Price': f"₹{forecast.predicted_price:.2f}",
                            'Expected Return': f"{forecast.expected_return:.1%}",
                            'Confidence Range': f"₹{forecast.confidence_lower:.2f} - ₹{forecast.confidence_upper:.2f}",
                            'Volatility': f"{forecast.volatility_forecast:.1%}",
                            'Forecast Date': forecast.forecast_date.strftime('%Y-%m-%d'),
                            'Horizon': horizon
                        })
            
            df = pd.DataFrame(forecast_data)
            
            # Sort by expected return
            if not df.empty:
                df = df.sort_values('Expected Return', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
            
            summary = f"Generated {len(forecast_data)} forecasts for {horizon} horizon."
            
            return df, summary
            
        except Exception as e:
            self.logger.error(f"Error getting forecasts: {e}")
            return pd.DataFrame(), f"Error: {str(e)}"
    
    def create_portfolio_display(self, portfolio_value: float = 100000) -> Tuple[pd.DataFrame, str, go.Figure]:
        """Create optimized portfolio display"""
        try:
            self._init_components()
            
            if self.forecast_generator is None or self.portfolio_optimizer is None:
                return pd.DataFrame(), "Model not available. Please train the model first.", go.Figure()
            
            # Generate forecasts for portfolio optimization
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT symbol 
                    FROM stock_data 
                    WHERE date >= date('now', '-7 days')
                    ORDER BY symbol
                    LIMIT 50
                """)
                symbols = [row[0] for row in cursor.fetchall()]
            
            if not symbols:
                return pd.DataFrame(), "No recent stock data found.", go.Figure()
            
            # Generate forecasts
            all_forecasts = self.forecast_generator.generate_forecasts_batch(symbols[:30])
            
            if not all_forecasts:
                return pd.DataFrame(), "No forecasts available for portfolio creation.", go.Figure()
            
            # Create optimized portfolio
            portfolio = self.portfolio_optimizer.create_portfolio(
                all_forecasts, 
                portfolio_value=portfolio_value
            )
            
            if not portfolio:
                return pd.DataFrame(), "Failed to create portfolio.", go.Figure()
            
            # Convert to display format
            portfolio_data = []
            for allocation in portfolio.allocations:
                portfolio_data.append({
                    'Symbol': allocation.symbol,
                    'Weight': f"{allocation.weight:.1%}",
                    'Shares': allocation.shares,
                    'Current Price': f"₹{allocation.current_price:.2f}",
                    'Target Price': f"₹{allocation.target_price:.2f}",
                    'Allocation Value': f"₹{allocation.allocation_value:,.0f}",
                    'Expected Return': f"{allocation.expected_return:.1%}",
                    'Sector': allocation.sector or 'Unknown'
                })
            
            df = pd.DataFrame(portfolio_data)
            
            # Portfolio summary
            summary = f"""
            Portfolio Summary:
            • Total Value: ₹{portfolio.total_value:,.0f}
            • Number of Positions: {len(portfolio.allocations)}
            • Expected Return: {portfolio.expected_return:.1%}
            • Expected Volatility: {portfolio.expected_volatility:.1%}
            • Sharpe Ratio: {portfolio.sharpe_ratio:.3f}
            • Diversification Score: {portfolio.diversification_score:.1%}
            """
            
            # Create portfolio allocation pie chart
            fig = px.pie(
                values=[a.allocation_value for a in portfolio.allocations],
                names=[a.symbol for a in portfolio.allocations],
                title="Portfolio Allocation"
            )
            fig.update_layout(height=500)
            
            return df, summary, fig
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio: {e}")
            return pd.DataFrame(), f"Error: {str(e)}", go.Figure()
    
    def get_model_performance(self) -> Tuple[str, go.Figure]:
        """Get model performance metrics and charts"""
        try:
            # Get recent signals and validate performance
            with sqlite3.connect(self.db_path) as conn:
                # Get recent signals
                signals_df = pd.read_sql_query("""
                    SELECT symbol, signal_type, expected_return, confidence, timestamp
                    FROM trading_signals
                    WHERE timestamp >= date('now', '-30 days')
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, conn)
                
                # Get recent forecasts
                forecasts_df = pd.read_sql_query("""
                    SELECT symbol, predicted_price, expected_return, created_at
                    FROM price_forecasts
                    WHERE created_at >= date('now', '-30 days')
                    ORDER BY created_at DESC
                    LIMIT 100
                """, conn)
            
            performance_text = f"""
            Model Performance (Last 30 Days):
            • Trading Signals Generated: {len(signals_df)}
            • Price Forecasts Generated: {len(forecasts_df)}
            • Average Signal Confidence: {signals_df['confidence'].mean():.1%}
            • Average Expected Return: {signals_df['expected_return'].mean():.1%}
            """
            
            if not signals_df.empty:
                # Create performance chart
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Signal Distribution', 'Confidence Distribution', 
                                  'Expected Returns', 'Signal Timeline'),
                    specs=[[{"type": "pie"}, {"type": "histogram"}],
                          [{"type": "histogram"}, {"type": "scatter"}]]
                )
                
                # Signal type distribution
                signal_counts = signals_df['signal_type'].value_counts()
                fig.add_trace(
                    go.Pie(labels=signal_counts.index, values=signal_counts.values, name="Signals"),
                    row=1, col=1
                )
                
                # Confidence distribution
                fig.add_trace(
                    go.Histogram(x=signals_df['confidence'], name="Confidence", nbinsx=20),
                    row=1, col=2
                )
                
                # Expected returns distribution
                fig.add_trace(
                    go.Histogram(x=signals_df['expected_return'], name="Expected Returns", nbinsx=20),
                    row=2, col=1
                )
                
                # Signal timeline
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
                daily_counts = signals_df.groupby(signals_df['timestamp'].dt.date).size()
                fig.add_trace(
                    go.Scatter(x=daily_counts.index, y=daily_counts.values, 
                             mode='lines+markers', name="Daily Signals"),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, showlegend=False)
                
            else:
                fig = go.Figure().add_annotation(text="No recent performance data available", x=0.5, y=0.5)
            
            return performance_text, fig
            
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return f"Error: {str(e)}", go.Figure()
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""
        
        with gr.Blocks(title="TFT Stock Prediction System") as interface:
            
            gr.Markdown("""
            # 🚀 TFT Stock Prediction System
            
            **Comprehensive AI-powered stock analysis for Indian markets (NSE/BSE)**
            
            This system uses a Temporal Fusion Transformer (TFT) model to generate:
            - Daily trading signals (Buy/Sell/Hold)
            - Multi-timeframe price forecasts
            - Optimized portfolio recommendations
            - Risk-adjusted investment strategies
            """)
            
            with gr.Tabs():
                
                # Stock Analysis Tab
                with gr.TabItem("📈 Stock Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            stock_dropdown = gr.Dropdown(
                                choices=self.get_available_stocks(),
                                label="Select Stock",
                                value=None,
                                interactive=True
                            )
                            days_slider = gr.Slider(
                                minimum=30,
                                maximum=365,
                                value=90,
                                step=30,
                                label="Chart Period (Days)"
                            )
                            analyze_btn = gr.Button("📊 Analyze Stock", variant="primary")
                        
                        with gr.Column(scale=3):
                            stock_chart = gr.Plot(label="Stock Price Chart")
                    
                    # Update chart when button is clicked
                    analyze_btn.click(
                        fn=self.create_stock_chart,
                        inputs=[stock_dropdown, days_slider],
                        outputs=stock_chart
                    )
                
                # Trading Signals Tab
                with gr.TabItem("⚡ Trading Signals"):
                    with gr.Row():
                        refresh_signals_btn = gr.Button("🔄 Generate Fresh Signals", variant="primary")
                    
                    signals_output = gr.Dataframe(
                        label="Current Trading Signals",
                        interactive=False
                    )
                    
                    signals_summary = gr.Textbox(
                        label="Signals Summary",
                        interactive=False
                    )
                    
                    refresh_signals_btn.click(
                        fn=self.get_trading_signals_display,
                        outputs=[signals_output, signals_summary]
                    )
                
                # Price Forecasts Tab
                with gr.TabItem("🔮 Price Forecasts"):
                    with gr.Row():
                        horizon_dropdown = gr.Dropdown(
                            choices=["1d", "5d", "22d"],
                            value="5d",
                            label="Forecast Horizon"
                        )
                        refresh_forecasts_btn = gr.Button("🔄 Generate Forecasts", variant="primary")
                    
                    forecasts_output = gr.Dataframe(
                        label="Price Forecasts",
                        interactive=False
                    )
                    
                    forecasts_summary = gr.Textbox(
                        label="Forecasts Summary",
                        interactive=False
                    )
                    
                    refresh_forecasts_btn.click(
                        fn=self.get_price_forecasts_display,
                        inputs=horizon_dropdown,
                        outputs=[forecasts_output, forecasts_summary]
                    )
                
                # Portfolio Optimization Tab
                with gr.TabItem(" Portfolio Optimization"):
                    with gr.Row():
                        portfolio_value_input = gr.Number(
                            value=100000,
                            label="Portfolio Value (₹)",
                            minimum=10000,
                            maximum=10000000
                        )
                        create_portfolio_btn = gr.Button("🎯 Create Optimal Portfolio", variant="primary")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            portfolio_output = gr.Dataframe(
                                label="Portfolio Allocations",
                                interactive=False
                            )
                        
                        with gr.Column(scale=1):
                            portfolio_chart = gr.Plot(label="Allocation Chart")
                    
                    portfolio_summary = gr.Textbox(
                        label="Portfolio Summary",
                        interactive=False,
                        lines=8
                    )
                    
                    create_portfolio_btn.click(
                        fn=self.create_portfolio_display,
                        inputs=portfolio_value_input,
                        outputs=[portfolio_output, portfolio_summary, portfolio_chart]
                    )
                
                # Model Performance Tab
                with gr.TabItem(" Model Performance"):
                    with gr.Row():
                        refresh_performance_btn = gr.Button("🔄 Refresh Performance", variant="primary")
                    
                    performance_text = gr.Textbox(
                        label="Performance Summary",
                        interactive=False,
                        lines=6
                    )
                    
                    performance_chart = gr.Plot(label="Performance Charts")
                    
                    refresh_performance_btn.click(
                        fn=self.get_model_performance,
                        outputs=[performance_text, performance_chart]
                    )
            
            gr.Markdown("""
            ---
            **Disclaimer**: This system is for educational and research purposes only. 
            Past performance does not guarantee future results. Always consult with a 
            financial advisor before making investment decisions.
            """)
        
        return interface
    
    def launch(self):
        """Launch the Gradio dashboard"""
        try:
            interface = self.create_interface()
            
            # Launch configuration
            launch_config = {
                'server_name': '0.0.0.0',
                'server_port': self.config['interface']['gradio']['port'],
                'share': self.config['interface']['gradio']['share'],
                'debug': self.config['interface']['gradio']['debug'],
                'show_error': True,
                'quiet': False
            }
            
            self.logger.info(f"Launching dashboard on port {launch_config['server_port']}")
            
            interface.launch(
                server_name="0.0.0.0",  # Accept connections from any IP
                server_port=7861,
                share=False,
                **launch_config
            )
            
        except Exception as e:
            self.logger.error(f"Error launching dashboard: {e}")
            raise
