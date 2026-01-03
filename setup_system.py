#!/usr/bin/env python3
"""
Setup and Installation Script for TFT Stock Prediction System
This script helps set up the system and verify all dependencies
"""

import sys
import os
import subprocess
import logging
from pathlib import Path


def setup_logging():
    """Setup logging for the setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    logger = setup_logging()
    
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True


def install_requirements():
    """Install required packages"""
    logger = setup_logging()
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False
    
    try:
        logger.info("Installing requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    logger = setup_logging()
    
    directories = [
        "data",
        "models", 
        "logs",
        "reports"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def verify_imports():
    """Verify that all critical imports work"""
    logger = setup_logging()
    
    critical_imports = [
        "torch",
        "pandas", 
        "numpy",
        "yfinance",
        "sklearn",
        "plotly",
        "gradio"
    ]
    
    failed_imports = []
    
    for module in critical_imports:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError as e:
            logger.error(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"Failed imports: {failed_imports}")
        return False
    
    logger.info("All critical imports successful")
    return True


def test_system_components():
    """Test basic system components"""
    logger = setup_logging()
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        # Test data collector
        from src.data.collectors import StockSymbolManager
        import yaml
        
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test symbol manager
        symbol_manager = StockSymbolManager(config)
        symbols = symbol_manager.get_nse_symbols()
        
        if symbols:
            logger.info(f"✓ Symbol manager working - found {len(symbols)} symbols")
        else:
            logger.warning("⚠ Symbol manager returned no symbols")
        
        logger.info("Basic system components test passed")
        return True
        
    except Exception as e:
        logger.error(f"System components test failed: {e}")
        return False


def main():
    """Main setup function"""
    logger = setup_logging()
    
    logger.info("=== TFT Stock Prediction System Setup ===")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Verify imports
    if not verify_imports():
        logger.error("Some imports failed. Please check the error messages above.")
        sys.exit(1)
    
    # Test system components
    if not test_system_components():
        logger.warning("System components test had issues, but installation continues")
    
    logger.info("=== Setup Complete ===")
    logger.info("""
Next steps:
1. Collect stock data: python main.py --mode collect
2. Train the model: python main.py --mode train
3. Generate predictions: python main.py --mode predict
4. Launch dashboard: python main.py --mode dashboard

Or run everything: python main.py --mode all
""")


if __name__ == "__main__":
    main()
