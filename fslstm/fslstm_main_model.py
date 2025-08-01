"""
Federated Stacked LSTM (FSLSTM) Model for Anomaly Detection in Smart Buildings.

This module implements the main FSLSTM architecture with three stacked LSTM layers
for privacy-preserving anomaly detection in IoT sensor networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LSTMLayer(nn.Module):
    """Custom LSTM layer with improved gradient flow."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.2,
        batch_first: bool = True
    ):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            output: LSTM output tensor
            hidden: Final hidden state tuple
        """
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        return output, hidden


class FSLSTMModel(nn.Module):
    """
    Federated Stacked LSTM Model for anomaly detection in smart buildings.
    
    Architecture:
    - 3 stacked LSTM layers
    - Fully connected layer
    - Output layer with softmax/linear activation
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        fc_size: int = 100,
        dropout: float = 0.2,
        task_type: str = "classification",
        sequence_length: int = 60
    ):
        super(FSLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.dropout = dropout
        self.task_type = task_type
        self.sequence_length = sequence_length
        
        # Stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            LSTMLayer(input_size, hidden_size, dropout, batch_first=True)
        )
        
        # Additional LSTM layers
        for i in range(1, num_layers):
            self.lstm_layers.append(
                LSTMLayer(hidden_size, hidden_size, dropout, batch_first=True)
            )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, fc_size)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc_activation = nn.ReLU()
        
        # Output layer
        if task_type == "classification":
            self.output_layer = nn.Linear(fc_size, num_classes)
        elif task_type == "regression":
            self.output_layer = nn.Linear(fc_size, 1)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the FSLSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional dictionary of hidden states for each layer
            
        Returns:
            Dictionary containing:
                - output: Model predictions
                - features: Last hidden state features
                - attention_weights: Attention weights (if applicable)
        """
        batch_size, seq_len, _ = x.shape
        
        if hidden is None:
            hidden = {}
        
        # Forward through stacked LSTM layers
        current_input = x
        new_hidden = {}
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            layer_hidden = hidden.get(f'layer_{i}', None)
            current_input, layer_hidden = lstm_layer(current_input, layer_hidden)
            new_hidden[f'layer_{i}'] = layer_hidden
        
        # Extract the last time step output
        last_hidden = current_input[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layer
        fc_output = self.fc(last_hidden)
        fc_output = self.fc_activation(fc_output)
        fc_output = self.fc_dropout(fc_output)
        
        # Output layer
        output = self.output_layer(fc_output)
        
        # Apply activation based on task type
        if self.task_type == "classification":
            if self.num_classes == 2:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)
        # For regression, no activation is applied
        
        return {
            'output': output,
            'features': last_hidden,
            'hidden_states': new_hidden
        }
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make predictions with the model.
        
        Args:
            x: Input tensor
            threshold: Classification threshold for binary tasks
            
        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            results = self.forward(x)
            output = results['output']
            
            if self.task_type == "classification":
                if self.num_classes == 2:
                    predictions = (output > threshold).float()
                else:
                    predictions = torch.argmax(output, dim=1)
            else:  # regression
                predictions = output
                
        return predictions
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated learning."""
        return {name: param.clone().detach() for name, param in self.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters for federated learning."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name])
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Get model gradients for federated learning."""
        gradients = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
        return gradients


class MultiTaskFSLSTM(nn.Module):
    """
    Multi-task FSLSTM for handling different sensor types simultaneously.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        tasks: Dict[str, Dict[str, Any]] = None,
        fc_size: int = 100,
        dropout: float = 0.2,
        sequence_length: int = 60
    ):
        super(MultiTaskFSLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tasks = tasks or {}
        self.fc_size = fc_size
        self.dropout = dropout
        self.sequence_length = sequence_length
        
        # Shared LSTM backbone
        self.shared_lstm = FSLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,  # Will be overridden by task-specific heads
            fc_size=fc_size,
            dropout=dropout,
            task_type="classification"
        )
        
        # Remove the output layer from shared model
        self.shared_lstm.output_layer = nn.Identity()
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in self.tasks.items():
            if task_config["type"] == "classification":
                num_classes = task_config.get("classes", 2)
                self.task_heads[task_name] = nn.Linear(fc_size, num_classes)
            elif task_config["type"] == "regression":
                self.task_heads[task_name] = nn.Linear(fc_size, 1)
    
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task learning."""
        # Get shared features
        shared_output = self.shared_lstm(x)
        features = shared_output['output']  # This is now fc_output due to Identity()
        
        results = {
            'features': shared_output['features'],
            'hidden_states': shared_output['hidden_states']
        }
        
        if task and task in self.task_heads:
            # Single task prediction
            task_output = self.task_heads[task](features)
            
            # Apply task-specific activation
            task_config = self.tasks[task]
            if task_config["type"] == "classification":
                if task_config.get("classes", 2) == 2:
                    task_output = torch.sigmoid(task_output)
                else:
                    task_output = F.softmax(task_output, dim=1)
            
            results[task] = task_output
        else:
            # Multi-task prediction
            for task_name, head in self.task_heads.items():
                task_output = head(features)
                
                # Apply task-specific activation
                task_config = self.tasks[task_name]
                if task_config["type"] == "classification":
                    if task_config.get("classes", 2) == 2:
                        task_output = torch.sigmoid(task_output)
                    else:
                        task_output = F.softmax(task_output, dim=1)
                
                results[task_name] = task_output
        
        return results


class AttentionFSLSTM(FSLSTMModel):
    """
    FSLSTM with attention mechanism for better feature learning.
    """
    
    def __init__(self, *args, **kwargs):
        super(AttentionFSLSTM, self).__init__(*args, **kwargs)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with attention mechanism."""
        batch_size, seq_len, _ = x.shape
        
        if hidden is None:
            hidden = {}
        
        # Forward through stacked LSTM layers
        current_input = x
        new_hidden = {}
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            layer_hidden = hidden.get(f'layer_{i}', None)
            current_input, layer_hidden = lstm_layer(current_input, layer_hidden)
            new_hidden[f'layer_{i}'] = layer_hidden
        
        # Apply attention mechanism
        attended_output, attention_weights = self.attention(
            current_input, current_input, current_input
        )
        
        # Residual connection and layer normalization
        attended_output = self.layer_norm(current_input + attended_output)
        
        # Extract the last time step output
        last_hidden = attended_output[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layer
        fc_output = self.fc(last_hidden)
        fc_output = self.fc_activation(fc_output)
        fc_output = self.fc_dropout(fc_output)
        
        # Output layer
        output = self.output_layer(fc_output)
        
        # Apply activation based on task type
        if self.task_type == "classification":
            if self.num_classes == 2:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)
        
        return {
            'output': output,
            'features': last_hidden,
            'hidden_states': new_hidden,
            'attention_weights': attention_weights
        }


def create_fslstm_model(config: Dict[str, Any]) -> FSLSTMModel:
    """
    Factory function to create FSLSTM model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured FSLSTM model
    """
    model_config = config.get('model', {})
    
    # Determine model type
    model_type = model_config.get('type', 'standard')
    
    # Common parameters
    common_params = {
        'input_size': model_config.get('input_size', 10),
        'hidden_size': model_config.get('hidden_size', 128),
        'num_layers': model_config.get('lstm_layers', 3),
        'num_classes': model_config.get('num_classes', 2),
        'fc_size': model_config.get('fc_size', 100),
        'dropout': model_config.get('dropout', 0.2),
        'task_type': model_config.get('task_type', 'classification'),
        'sequence_length': config.get('data', {}).get('sequence_length', 60)
    }
    
    if model_type == 'multitask':
        tasks = config.get('tasks', {})
        return MultiTaskFSLSTM(
            tasks=tasks,
            **common_params
        )
    elif model_type == 'attention':
        return AttentionFSLSTM(**common_params)
    else:
        return FSLSTMModel(**common_params)


if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'input_size': 10,
            'hidden_size': 128,
            'lstm_layers': 3,
            'num_classes': 2,
            'fc_size': 100,
            'dropout': 0.2,
            'task_type': 'classification'
        },
        'data': {
            'sequence_length': 60
        }
    }
    
    # Create model
    model = create_fslstm_model(config)
    
    # Test forward pass
    batch_size = 32
    seq_len = 60
    input_size = 10
    
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    
    print(f"Model output shape: {output['output'].shape}")
    print(f"Features shape: {output['features'].shape}")
    print("Model created successfully!")
        