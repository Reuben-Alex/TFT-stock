"""
Temporal Fusion Transformer (TFT) Model Implementation
Multi-task learning architecture for stock price prediction with attention mechanisms
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit activation function"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super(GatedLinearUnit, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        linear_out = self.linear(x)  # (batch_size, seq_len, hidden_size * 2)
        linear_out = self.dropout(linear_out)
        
        # Split into two parts for gating
        a, b = linear_out.chunk(2, dim=-1)  # Each: (batch_size, seq_len, hidden_size)
        
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network with skip connections"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int = None, 
                 dropout: float = 0.1):
        super(GatedResidualNetwork, self).__init__()
        
        if output_size is None:
            output_size = input_size
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear_skip = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.linear_gate = nn.Linear(output_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, ..., input_size)
        Returns:
            Output tensor of shape (batch_size, ..., output_size)
        """
        # Main path
        hidden = self.dropout(F.elu(self.linear1(x)))
        hidden = self.linear2(hidden)
        
        # Gating mechanism
        gate = torch.sigmoid(self.linear_gate(hidden))
        
        # Skip connection
        if self.linear_skip is not None:
            x_skip = self.linear_skip(x)
        else:
            x_skip = x
            
        # Apply gating and residual connection
        output = gate * hidden + x_skip
        
        return self.layer_norm(output)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance"""
    
    def __init__(self, input_size: int, num_features: int, hidden_size: int, dropout: float = 0.1):
        super(VariableSelectionNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # Feature processing networks
        self.feature_networks = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_features)
        ])
        
        # Variable selection network
        self.selection_network = GatedResidualNetwork(
            input_size * num_features, hidden_size, num_features, dropout
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features, input_size)
        Returns:
            selected_features: (batch_size, seq_len, hidden_size)
            feature_weights: (batch_size, seq_len, num_features)
        """
        batch_size, seq_len, num_features, input_size = x.shape
        
        # Process each feature through its network
        processed_features = []
        for i in range(num_features):
            feature_output = self.feature_networks[i](x[:, :, i, :])  # (batch_size, seq_len, hidden_size)
            processed_features.append(feature_output)
        
        processed_features = torch.stack(processed_features, dim=2)  # (batch_size, seq_len, num_features, hidden_size)
        
        # Variable selection
        flattened_input = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, num_features * input_size)
        selection_weights = self.selection_network(flattened_input)  # (batch_size, seq_len, num_features)
        selection_weights = F.softmax(selection_weights, dim=-1)
        
        # Apply selection weights
        selected_features = torch.sum(
            processed_features * selection_weights.unsqueeze(-1), 
            dim=2
        )  # (batch_size, seq_len, hidden_size)
        
        return selected_features, selection_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: Input tensors of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-task stock prediction
    """
    
    def __init__(self, config: Dict):
        super(TemporalFusionTransformer, self).__init__()
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model dimensions
        self.static_features = config['model']['tft']['static_features']
        self.dynamic_features = config['model']['tft']['dynamic_features']
        self.hidden_size = config['model']['tft']['hidden_size']
        self.lstm_layers = config['model']['tft']['lstm_layers']
        self.attention_heads = config['model']['tft']['attention_heads']
        self.dropout = config['model']['tft']['dropout']
        self.lookback_window = config['model']['tft']['lookback_window']
        self.prediction_horizons = config['model']['tft']['prediction_horizons']
        self.quantiles = config['model']['tft']['quantiles']
        
        # Static feature processing
        self.static_context_network = GatedResidualNetwork(
            self.static_features, self.hidden_size, self.hidden_size, self.dropout
        )
        
        # Dynamic feature processing
        self.dynamic_context_network = GatedResidualNetwork(
            self.dynamic_features, self.hidden_size, self.hidden_size, self.dropout
        )
        
        # Variable selection networks
        self.static_selection = VariableSelectionNetwork(
            1, self.static_features, self.hidden_size, self.dropout
        )
        
        self.dynamic_selection = VariableSelectionNetwork(
            1, self.dynamic_features, self.hidden_size, self.dropout
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(
            self.hidden_size, 
            self.hidden_size, 
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0
        )
        
        # Attention mechanism
        self.self_attention = MultiHeadAttention(
            self.hidden_size, self.attention_heads, self.dropout
        )
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(self.hidden_size, max_len=self.lookback_window)
        
        # Multi-task output heads
        self.task_configs = config['model']['tasks']
        self.output_heads = nn.ModuleDict()
        
        for task_name, task_config in self.task_configs.items():
            if 'price' in task_name:
                # Regression heads for price prediction
                output_size = len(self.quantiles) * len(self.prediction_horizons)
            else:
                # Ranking head
                output_size = 1
                
            self.output_heads[task_name] = nn.Sequential(
                GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout),
                nn.Linear(self.hidden_size, output_size)
            )
        
        # Initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, static_features: torch.Tensor, dynamic_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of TFT model
        
        Args:
            static_features: (batch_size, static_features)
            dynamic_features: (batch_size, seq_len, dynamic_features)
            
        Returns:
            Dictionary of task predictions
        """
        batch_size = dynamic_features.shape[0]
        seq_len = dynamic_features.shape[1]
        
        # Process static features
        static_context = self.static_context_network(static_features)  # (batch_size, hidden_size)
        static_context = static_context.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)
        
        # Process dynamic features
        dynamic_context = self.dynamic_context_network(dynamic_features)  # (batch_size, seq_len, hidden_size)
        
        # Variable selection for static features
        static_expanded = static_features.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, static_features)
        static_reshaped = static_expanded.view(batch_size, seq_len, self.static_features, 1)
        static_selected, static_weights = self.static_selection(static_reshaped)
        
        # Variable selection for dynamic features
        # dynamic_features is already (batch_size, seq_len, dynamic_features)
        # We need to add the last dimension for the VariableSelectionNetwork
        dynamic_reshaped = dynamic_features.unsqueeze(-1)  # (batch_size, seq_len, dynamic_features, 1)
        dynamic_selected, dynamic_weights = self.dynamic_selection(dynamic_reshaped)
        
        # Combine selected features
        combined_features = static_selected + dynamic_selected  # (batch_size, seq_len, hidden_size)
        
        # Add positional encoding
        combined_features_pe = combined_features.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        combined_features_pe = self.pos_encoding(combined_features_pe)
        combined_features = combined_features_pe.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(combined_features)
        
        # Self-attention
        attended_features, attention_weights = self.self_attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        final_features = lstm_out + attended_features
        
        # Multi-task predictions
        predictions = {}
        
        for task_name, output_head in self.output_heads.items():
            task_pred = output_head(final_features[:, -1, :])  # Use last timestep
            predictions[task_name] = task_pred
        
        # Store attention weights for interpretability
        predictions['attention_weights'] = attention_weights
        predictions['static_weights'] = static_weights
        predictions['dynamic_weights'] = dynamic_weights
        
        return predictions
    
    def predict_quantiles(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Reshape price predictions into quantile format
        
        Args:
            predictions: Raw predictions (batch_size, output_size)
            
        Returns:
            Quantile predictions (batch_size, horizons, quantiles)
        """
        batch_size = predictions.shape[0]
        num_quantiles = len(self.quantiles)
        num_horizons = len(self.prediction_horizons)
        
        # Reshape to (batch_size, horizons, quantiles)
        quantile_preds = predictions.view(batch_size, num_horizons, num_quantiles)
        
        return quantile_preds
    
    def compute_quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
        """
        Compute quantile loss for probabilistic predictions
        
        Args:
            predictions: Model predictions (batch_size, num_quantiles)
            targets: True values (batch_size,)
            quantiles: List of quantile levels
            
        Returns:
            Quantile loss
        """
        losses = []
        
        for i, q in enumerate(quantiles):
            pred_q = predictions[:, i]
            error = targets - pred_q
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def get_feature_importance(self, static_weights: torch.Tensor, dynamic_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract feature importance from variable selection weights
        
        Args:
            static_weights: Static feature selection weights
            dynamic_weights: Dynamic feature selection weights
            
        Returns:
            Feature importance dictionary
        """
        importance = {
            'static_importance': static_weights.mean(dim=(0, 1)),  # Average across batch and time
            'dynamic_importance': dynamic_weights.mean(dim=(0, 1))  # Average across batch and time
        }
        
        return importance


class TFTLoss(nn.Module):
    """
    Multi-task loss function for TFT model
    """
    
    def __init__(self, config: Dict):
        super(TFTLoss, self).__init__()
        
        self.config = config
        self.task_configs = config['model']['tasks']
        self.quantiles = config['model']['tft']['quantiles']
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of target values
            
        Returns:
            Dictionary of task losses and total loss
        """
        losses = {}
        total_loss = 0
        
        for task_name, task_config in self.task_configs.items():
            if task_name not in predictions or task_name not in targets:
                continue
                
            pred = predictions[task_name]
            target = targets[task_name]
            weight = task_config['weight']
            loss_type = task_config['loss']
            
            if loss_type == 'mse':
                task_loss = F.mse_loss(pred, target)
            elif loss_type == 'ranking':
                # Pairwise ranking loss
                task_loss = self._ranking_loss(pred, target)
            elif loss_type == 'quantile':
                task_loss = self._quantile_loss(pred, target)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            losses[f'{task_name}_loss'] = task_loss
            total_loss += weight * task_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _ranking_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss"""
        # Simple ranking loss based on relative ordering
        batch_size = predictions.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Create pairs
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # (batch, batch)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # (batch, batch)
        
        # Sign agreement loss
        sign_agreement = torch.sign(pred_diff) * torch.sign(target_diff)
        loss = torch.relu(1 - sign_agreement)
        
        # Mask diagonal and average
        mask = torch.eye(batch_size, device=predictions.device) == 0
        loss = loss[mask].mean()
        
        return loss
    
    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Quantile regression loss"""
        losses = []
        
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, i]
            error = targets - pred_q
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
        
        return torch.stack(losses).mean()
