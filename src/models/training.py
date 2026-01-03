"""
Training Pipeline for TFT Model
Handles model training, validation, and hyperparameter optimization
"""

import logging
import os
import sqlite3
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml

from .tft import TemporalFusionTransformer, TFTLoss
from ..data.processors import DataProcessor
from ..data.features import FeatureEngineer


class StockDataset(Dataset):
    """PyTorch Dataset for stock time series data"""
    
    def __init__(self, processed_data: Dict[str, Dict], mode: str = 'train'):
        self.processed_data = processed_data
        self.mode = mode
        self.symbols = list(processed_data.keys())
        
        # Create index mapping
        self.sample_indices = []
        for symbol, data in processed_data.items():
            for i in range(len(data['X'])):
                self.sample_indices.append((symbol, i))
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        symbol, sample_idx = self.sample_indices[idx]
        data = self.processed_data[symbol]
        
        # Get sequence data
        X = torch.FloatTensor(data['X'][sample_idx])  # (seq_len, features)
        y = torch.FloatTensor(data['y'][sample_idx])  # (output_size,)
        
        # Static features (simplified - using symbol hash as placeholder)
        static_features = torch.FloatTensor([hash(symbol) % 1000 / 1000.0] * 10)
        
        # Convert date to timestamp for PyTorch compatibility
        date_obj = data['dates'][sample_idx]
        if hasattr(date_obj, 'timestamp'):
            date_timestamp = float(date_obj.timestamp())
        else:
            date_timestamp = float(date_obj)
        
        return {
            'static_features': static_features,
            'dynamic_features': X,
            'targets': y,
            'symbol': symbol,
            'date': date_timestamp
        }


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.mode_dict = {'min': torch.lt, 'max': torch.gt}
        self.monitor_op = self.mode_dict[mode]
        
    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < (self.best_score - self.min_delta):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class ModelCheckpoint:
    """Model checkpointing utility"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 mode: str = 'min', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.mode_dict = {'min': torch.lt, 'max': torch.gt}
        self.monitor_op = self.mode_dict[mode]
        
    def __call__(self, model: nn.Module, current_score: float, 
                 epoch: int, optimizer: optim.Optimizer = None):
        if not self.save_best_only:
            self._save_checkpoint(model, epoch, current_score, optimizer)
        else:
            if self.best_score is None or current_score < self.best_score:
                self.best_score = current_score
                self._save_checkpoint(model, epoch, current_score, optimizer)
                
    def _save_checkpoint(self, model: nn.Module, epoch: int, 
                        score: float, optimizer: optim.Optimizer = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, self.filepath)


class TFTTrainer:
    """Main training class for TFT model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Training parameters
        self.batch_size = config['training']['batch_size']
        self.max_epochs = config['training']['max_epochs']
        self.learning_rate = float(config['training']['learning_rate'])
        self.weight_decay = float(config['training']['weight_decay'])
        
        # Device configuration - optimized for Apple Silicon
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
    def prepare_data(self, symbols: List[str] = None) -> Dict[str, DataLoader]:
        """Prepare datasets and dataloaders"""
        self.logger.info("Preparing training data...")
        
        # Initialize data processor and feature engineer
        db_path = self.config['data']['database']['path']
        processor = DataProcessor(self.config, db_path)
        feature_engineer = FeatureEngineer(self.config, db_path)
        
        # Get available symbols
        if symbols is None:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM stock_data LIMIT 100")  # Limit for testing
                symbols = [row[0] for row in cursor.fetchall()]
        
        # Process data for all symbols
        all_processed_data = {}
        
        for symbol in symbols:
            try:
                # Get raw stock data
                stock_data = processor.get_stock_data_from_db(symbol)
                
                if stock_data.empty:
                    continue
                
                # Add features
                stock_data_with_features = feature_engineer.engineer_features(stock_data, symbol)
                
                # Process for training
                processed = processor.process_single_stock(symbol)
                
                if processed:
                    all_processed_data[symbol] = processed
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_processed_data:
            raise ValueError("No data processed successfully")
        
        self.logger.info(f"Processed data for {len(all_processed_data)} stocks")
        
        # Split data by time (walk-forward validation)
        train_data, val_data, test_data = self._temporal_split(all_processed_data)
        
        # Create datasets
        train_dataset = StockDataset(train_data, mode='train')
        val_dataset = StockDataset(val_data, mode='val')
        test_dataset = StockDataset(test_data, mode='test')
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }
    
    def _temporal_split(self, processed_data: Dict[str, Dict]) -> Tuple[Dict, Dict, Dict]:
        """Split data temporally for walk-forward validation"""
        train_data = {}
        val_data = {}
        test_data = {}
        
        validation_config = self.config['training']['validation']
        train_years = validation_config['initial_train_years']
        val_months = validation_config['validation_months']
        test_months = validation_config['test_months']
        
        for symbol, data in processed_data.items():
            dates = pd.to_datetime(data['dates'])
            
            if len(dates) < 252:  # Less than 1 year of data
                continue
            
            # Calculate split dates
            total_days = len(dates)
            train_days = int(train_years * 252)  # Approximate trading days per year
            val_days = int(val_months * 21)  # Approximate trading days per month
            test_days = int(test_months * 21)
            
            if total_days < train_days + val_days + test_days:
                # Proportional split if insufficient data
                train_end = int(0.7 * total_days)
                val_end = int(0.85 * total_days)
            else:
                train_end = train_days
                val_end = train_days + val_days
            
            # Split the data
            train_data[symbol] = {
                'X': data['X'][:train_end],
                'y': data['y'][:train_end],
                'dates': data['dates'][:train_end],
                'scalers': data['scalers'],
                'feature_cols': data['feature_cols'],
                'target_cols': data['target_cols']
            }
            
            val_data[symbol] = {
                'X': data['X'][train_end:val_end],
                'y': data['y'][train_end:val_end],
                'dates': data['dates'][train_end:val_end],
                'scalers': data['scalers'],
                'feature_cols': data['feature_cols'],
                'target_cols': data['target_cols']
            }
            
            test_data[symbol] = {
                'X': data['X'][val_end:],
                'y': data['y'][val_end:],
                'dates': data['dates'][val_end:],
                'scalers': data['scalers'],
                'feature_cols': data['feature_cols'],
                'target_cols': data['target_cols']
            }
        
        return train_data, val_data, test_data
    
    def build_model(self):
        """Initialize model, criterion, optimizer, and scheduler"""
        self.logger.info("Building TFT model...")
        
        # Initialize model with default config first
        self.model = TemporalFusionTransformer(self.config).to(self.device)
        
        # Initialize criterion
        self.criterion = TFTLoss(self.config)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize scheduler
        scheduler_config = self.config['training']['scheduler']
        if scheduler_config['type'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(scheduler_config['factor']),
                patience=int(scheduler_config['patience']),
                min_lr=float(scheduler_config['min_lr'])
            )
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            static_features = batch['static_features'].to(self.device)
            dynamic_features = batch['dynamic_features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(static_features, dynamic_features)
            
            # Prepare targets for multi-task learning
            # Each price task expects (batch_size, quantiles * horizons)
            # For simplicity, we'll repeat the target for each quantile
            batch_size = targets.shape[0]
            num_quantiles = len(self.config['model']['tft']['quantiles'])
            num_horizons = len(self.config['model']['tft']['prediction_horizons'])
            
            task_targets = {}
            
            # Fix target shape issues - ensure we have proper dimensions
            if len(targets.shape) == 1:
                targets = targets.unsqueeze(1)  # Make it (batch_size, 1)
            
            # For price tasks, create targets for each quantile and horizon
            for task_name in ['daily_price', 'weekly_price', 'monthly_price']:
                # Use the first target value for all tasks (simplified)
                target_val = targets[:, 0] if targets.shape[1] > 0 else targets.flatten()
                
                # Repeat target for each quantile and horizon combination
                if len(target_val.shape) == 0:  # scalar
                    target_val = target_val.unsqueeze(0)
                repeated_target = target_val.unsqueeze(1).repeat(1, num_quantiles * num_horizons)
                task_targets[task_name] = repeated_target
            
            # Ranking score uses single value
            ranking_target = targets[:, 0] if targets.shape[1] > 0 else targets.flatten()
            if len(ranking_target.shape) == 0:
                ranking_target = ranking_target.unsqueeze(0)
            task_targets['ranking_score'] = ranking_target
            
            # Compute loss
            losses = self.criterion(predictions, task_targets)
            total_loss_batch = losses['total_loss']
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {total_loss_batch.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                static_features = batch['static_features'].to(self.device)
                dynamic_features = batch['dynamic_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                predictions = self.model(static_features, dynamic_features)
                
                # Prepare targets for multi-task learning
                batch_size = targets.shape[0]
                num_quantiles = len(self.config['model']['tft']['quantiles'])
                num_horizons = len(self.config['model']['tft']['prediction_horizons'])
                
                task_targets = {}
                
                # Fix target shape issues - ensure we have proper dimensions
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)  # Make it (batch_size, 1)
                
                # For price tasks, create targets for each quantile and horizon
                for task_name in ['daily_price', 'weekly_price', 'monthly_price']:
                    # Use the first target value for all tasks (simplified)
                    target_val = targets[:, 0] if targets.shape[1] > 0 else targets.flatten()
                    
                    # Repeat target for each quantile and horizon combination
                    if len(target_val.shape) == 0:  # scalar
                        target_val = target_val.unsqueeze(0)
                    repeated_target = target_val.unsqueeze(1).repeat(1, num_quantiles * num_horizons)
                    task_targets[task_name] = repeated_target
                
                # Ranking score uses single value
                ranking_target = targets[:, 0] if targets.shape[1] > 0 else targets.flatten()
                if len(ranking_target.shape) == 0:
                    ranking_target = ranking_target.unsqueeze(0)
                task_targets['ranking_score'] = ranking_target
                
                # Compute loss
                losses = self.criterion(predictions, task_targets)
                total_loss += losses['total_loss'].item()
                num_batches += 1
                
                # Store predictions for metrics
                all_predictions.append(predictions['daily_price'].cpu())
                all_targets.append(targets[:, 0].cpu())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Calculate additional metrics
        if all_predictions:
            predictions_tensor = torch.cat(all_predictions)
            targets_tensor = torch.cat(all_targets)
            
            # Fix shape mismatch: predictions has 9 values (3 quantiles * 3 horizons), targets has 1
            pred_np = predictions_tensor.numpy()
            target_np = targets_tensor.numpy()
            
            # Use only the median quantile (middle prediction) to match target shape
            if pred_np.shape[1] > 1:
                pred_np = pred_np[:, pred_np.shape[1] // 2]  # Middle quantile
            else:
                pred_np = pred_np[:, 0]
                
            mse = mean_squared_error(target_np, pred_np)
            mae = mean_absolute_error(target_np, pred_np)
            
            return {'loss': avg_loss, 'mse': mse, 'mae': mae}
        
        return {'loss': avg_loss}
    
    def train_model(self, symbols: List[str] = None):
        """Main training loop"""
        self.logger.info("Starting TFT model training...")
        
        # Prepare data
        dataloaders = self.prepare_data(symbols)
        
        # Build model
        self.build_model()
        
        # Initialize training utilities
        early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            min_delta=self.config['training']['early_stopping']['min_delta'],
            monitor=self.config['training']['early_stopping']['monitor']
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join('models', 'tft_best_model.pth'),
            monitor='val_loss'
        )
        
        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Save best model
            model_checkpoint(self.model, val_metrics['loss'], epoch, self.optimizer)
            
            # Early stopping check
            if early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Update best validation loss
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            mse_info = f", Val MSE: {val_metrics['mse']:.4f}" if 'mse' in val_metrics else ""
            self.logger.info(
                f"Epoch {epoch + 1}/{self.max_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}"
                f"{mse_info}"
                f" - {epoch_time:.2f}s"
            )
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s. Best val loss: {best_val_loss:.4f}")
        
        # Save training history
        self._save_training_history()
        
        return self.history
    
    def _save_training_history(self):
        """Save training history to file"""
        history_path = 'models/training_history.pkl'
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Model loaded from {checkpoint_path}")
        
        return checkpoint.get('score', None)
    
    def evaluate_model(self, test_loader: DataLoader = None) -> Dict[str, float]:
        """Evaluate model on test set"""
        if test_loader is None:
            test_loader = self.test_loader
        
        if test_loader is None:
            raise ValueError("No test dataloader available")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                static_features = batch['static_features'].to(self.device)
                dynamic_features = batch['dynamic_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                predictions = self.model(static_features, dynamic_features)
                
                # Store predictions
                all_predictions.append(predictions['daily_price'].cpu())
                all_targets.append(targets[:, 0].cpu())
        
        # Calculate metrics
        predictions_tensor = torch.cat(all_predictions)
        targets_tensor = torch.cat(all_targets)
        
        mse = mean_squared_error(targets_tensor.numpy(), predictions_tensor.numpy())
        mae = mean_absolute_error(targets_tensor.numpy(), predictions_tensor.numpy())
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_direction = (predictions_tensor > 0).float()
        true_direction = (targets_tensor > 0).float()
        directional_accuracy = (pred_direction == true_direction).float().mean().item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
        
        self.logger.info("Test Results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return metrics
