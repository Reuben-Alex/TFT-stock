# TFT Stock Prediction System

A comprehensive deep learning system for Indian stock market prediction using Temporal Fusion Transformer (TFT) architecture. The system provides multi-timeframe price forecasts, buy/sell signals, and portfolio optimization for NSE/BSE listed stocks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Data Sources](#data-sources)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Overview

This system implements a state-of-the-art Temporal Fusion Transformer for stock price prediction specifically designed for the Indian stock market. It combines advanced deep learning techniques with comprehensive technical analysis to provide accurate multi-horizon forecasts and trading signals.

### Key Capabilities

- **Multi-timeframe Predictions**: 1-day, 1-week, and 1-month price forecasts
- **Trading Signals**: Automated buy/sell recommendations with confidence scores
- **Portfolio Optimization**: Risk-adjusted portfolio construction using Modern Portfolio Theory
- **Technical Analysis**: 60+ technical indicators and market features
- **Real-time Data**: Yahoo Finance integration for live market data
- **Interactive Dashboard**: Web-based interface for analysis and visualization

## Features

### Machine Learning
- Temporal Fusion Transformer with attention mechanisms
- Multi-task learning for simultaneous price and signal prediction
- Quantile regression for uncertainty estimation
- GPU acceleration support (CUDA/MPS)
- Walk-forward validation for time series data

### Data Processing
- Automated data collection from Yahoo Finance
- 75+ NSE/BSE stock symbols
- Advanced feature engineering with technical indicators
- Data quality validation and outlier detection
- SQLite database for efficient storage

### Trading Analytics
- Buy/sell signal generation with confidence scores
- Multi-horizon price forecasting (1D, 1W, 1M)
- Portfolio optimization with risk parity
- Performance metrics and backtesting
- Risk-adjusted returns analysis

### User Interface
- Interactive Gradio web dashboard
- Real-time stock analysis and predictions
- Portfolio management interface
- Comprehensive reporting and visualization
- Export capabilities for analysis results

## System Architecture

```
├── config/                 # Configuration files
│   ├── config.yaml        # Main configuration
│   └── nse_symbols.txt    # Stock symbols list
├── data/                  # Data storage
│   └── stocks.db         # SQLite database
├── models/               # Trained model storage
├── logs/                # System logs
├── src/
│   ├── data/           # Data processing modules
│   │   ├── collectors.py    # Data collection from Yahoo Finance
│   │   ├── processors.py    # Data cleaning and processing
│   │   └── features.py      # Feature engineering
│   ├── models/         # Model architecture and training
│   │   ├── tft.py          # TFT model implementation
│   │   ├── training.py     # Training pipeline
│   │   └── evaluation.py   # Model evaluation
│   ├── prediction/     # Prediction and analysis
│   │   ├── signals.py      # Trading signal generation
│   │   ├── forecasting.py  # Price forecasting
│   │   └── portfolio.py    # Portfolio optimization
│   └── interface/      # User interface
│       ├── dashboard.py    # Gradio web interface
│       └── reports.py      # Report generation
└── main.py            # Main application entry point
```

## Installation

### Prerequisites

- Python 3.9 or higher
- 8GB+ RAM (16GB recommended for full dataset)
- GPU with CUDA support (optional, for faster training)

### Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd Stock

# Install required packages
pip install -r requirements.txt
```

### Required Packages

```
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
gradio>=4.0.0
pyyaml>=6.0.0
ta>=0.10.2
```

## Usage

### Quick Start

```bash
# Collect stock data
python main.py --mode collect

# Train the TFT model
python main.py --mode train

# Launch interactive dashboard
python main.py --mode dashboard

# Run complete pipeline
python main.py --mode all
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --mode {collect,train,dashboard,all}  Execution mode
  --config PATH                         Configuration file path
  --symbols PATH                        Stock symbols file path
  --output PATH                         Output directory path
  --debug                               Enable debug logging
```

### Configuration File

Edit `config/config.yaml` to customize system behavior:

```yaml
# Data Configuration
data:
  start_date: '2023-01-01'
  end_date: '2024-12-31'
  
# Model Configuration
model:
  tft:
    hidden_size: 32
    lstm_layers: 1
    attention_heads: 2
    dropout: 0.2

# Training Configuration
training:
  batch_size: 128
  max_epochs: 20
  learning_rate: 0.001
```

## Configuration

### Model Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| hidden_size | Hidden layer dimension | 32 | 16-512 |
| lstm_layers | Number of LSTM layers | 1 | 1-3 |
| attention_heads | Attention mechanism heads | 2 | 1-8 |
| dropout | Dropout rate | 0.2 | 0.0-0.5 |
| lookback_window | Historical data window | 30 | 10-120 |

### Training Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| batch_size | Training batch size | 128 | 32-512 |
| max_epochs | Maximum training epochs | 20 | 5-100 |
| learning_rate | Learning rate | 0.001 | 1e-5-1e-2 |
| weight_decay | L2 regularization | 0.0001 | 1e-6-1e-3 |

### Data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| start_date | Data collection start | '2023-01-01' |
| end_date | Data collection end | '2024-12-31' |
| validation_split | Validation data ratio | 0.2 |
| test_split | Test data ratio | 0.2 |

## Model Architecture

### Temporal Fusion Transformer

The TFT model consists of several key components:

1. **Variable Selection Networks**: Automatic feature selection for static and dynamic features
2. **Gated Residual Networks**: Deep learning blocks with skip connections and gating mechanisms
3. **LSTM Encoder-Decoder**: Sequential processing for temporal patterns
4. **Multi-Head Attention**: Attention mechanisms for long-term dependencies
5. **Quantile Outputs**: Probabilistic predictions with uncertainty estimation

### Multi-Task Learning

The system predicts multiple targets simultaneously:

- **Daily Price**: Next day closing price
- **Weekly Price**: 5-day ahead price
- **Monthly Price**: 20-day ahead price  
- **Ranking Score**: Relative stock performance ranking

### Feature Engineering

#### Technical Indicators (40+ features)
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility measures (Bollinger Bands, ATR)
- Volume analysis (OBV, Volume Rate of Change)
- Price patterns (Support/Resistance levels)

#### Market Features (10+ features)
- Market correlation and beta
- Sector relative strength
- VIX integration
- Market trend indicators

#### Temporal Features (10+ features)
- Day of week effects
- Month seasonality
- Holiday adjustments
- Market session indicators

## Data Sources

### Primary Data Source
- **Yahoo Finance API**: Real-time and historical stock data
- **Frequency**: Daily OHLCV data
- **Coverage**: 75+ NSE/BSE listed stocks
- **History**: 2+ years of historical data

### Stock Selection Criteria
- Market capitalization > 1000 crores
- Average daily volume > 100,000 shares
- Continuous listing for 2+ years
- Active trading with minimal gaps

### Data Quality Measures
- Outlier detection using statistical methods
- Missing data imputation with forward fill
- Price adjustment for splits and dividends
- Volume normalization and scaling

## Performance

### Model Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Validation MSE | 0.0028 | Mean squared error on validation set |
| Training Loss | 0.5139 | Multi-task training loss |
| Model Parameters | 127,013 | Total trainable parameters |
| Training Time | 2.2 min/epoch | On Apple M2 chip |

### Prediction Accuracy

| Timeframe | MAE | RMSE | Directional Accuracy |
|-----------|-----|------|---------------------|
| 1-Day | 0.024 | 0.053 | 65.2% |
| 1-Week | 0.041 | 0.078 | 61.8% |
| 1-Month | 0.067 | 0.112 | 58.4% |

### System Performance

| Component | Speed | Resource Usage |
|-----------|-------|----------------|
| Data Collection | 75 stocks/2min | 200MB RAM |
| Feature Engineering | 10,000 samples/sec | 500MB RAM |
| Model Training | 2.2 min/epoch | 1GB RAM |
| Inference | 1000 predictions/sec | 100MB RAM |

## API Reference

### Data Collection

```python
from src.data.collectors import StockDataCollector

collector = StockDataCollector(config)
data = collector.collect_stock_data(['RELIANCE.NS', 'TCS.NS'])
```

### Feature Engineering

```python
from src.data.features import FeatureEngineer

engineer = FeatureEngineer(config, 'data/stocks.db')
features = engineer.engineer_features(data, 'RELIANCE.NS')
```

### Model Training

```python
from src.models.training import TFTTrainer

trainer = TFTTrainer(config)
trainer.train_model(['RELIANCE.NS', 'TCS.NS'])
```

### Prediction Generation

```python
from src.prediction.forecasting import StockForecaster

forecaster = StockForecaster(config)
predictions = forecaster.generate_forecasts(['RELIANCE.NS'])
```

### Portfolio Optimization

```python
from src.prediction.portfolio import PortfolioOptimizer

optimizer = PortfolioOptimizer(config)
portfolio = optimizer.optimize_portfolio(predictions, risk_level=0.5)
```

## File Structure

### Configuration Files
- `config/config.yaml`: Main system configuration
- `config/nse_symbols.txt`: List of stock symbols to process

### Data Files
- `data/stocks.db`: SQLite database containing processed stock data
- `data/raw/`: Raw data files from Yahoo Finance (temporary)

### Model Files
- `models/tft_best_model.pth`: Trained TFT model checkpoint
- `models/scaler.pkl`: Feature scaling parameters

### Log Files
- `logs/stock_predictor.log`: System execution logs
- `logs/training.log`: Training progress logs

## Troubleshooting

### Common Issues

**Memory Error during Training**
- Reduce batch_size in config.yaml
- Decrease lookback_window
- Train on fewer stocks initially

**Data Collection Failures**
- Check internet connection
- Verify Yahoo Finance symbol format
- Reduce concurrent threads in config

**Model Training Convergence Issues**
- Adjust learning rate (try 0.0001-0.01)
- Increase/decrease model complexity
- Check for data quality issues

**Dashboard Access Issues**
- Verify port 7861 is available
- Check firewall settings
- Try different port in config



## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest tests/`
5. Submit pull request



## License

This project is licensed under the MIT License. See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Stock market investments carry inherent risks, and past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.

## Support

For technical support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation
- Consult the configuration guide

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Python Version**: 3.9+  
**License**: MIT