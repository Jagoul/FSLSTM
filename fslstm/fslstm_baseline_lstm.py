"""
Centralized LSTM Baseline for comparison with FSLSTM.

This module implements a centralized LSTM model that serves as a baseline
for comparing the performance of the federated FSLSTM approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CentralizedLSTM(nn.Module):
    """
    Centralized LSTM model for anomaly detection.
    This serves as a baseline to compare against the federated approach.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        fc_size: int = 100,
        dropout: float = 0.2,
        task_type: str = "classification"
    ):
        """
        Initialize centralized LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden units per LSTM layer
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            fc_size: Fully connected layer size
            dropout: Dropout rate
            task_type: Type of task (classification/regression)
        """
        super(CentralizedLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.dropout = dropout
        self.task_type = task_type
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc = nn.Linear(hidden_size, fc_size)
        self.fc_activation = nn.ReLU()
        self.fc_dropout = nn.Dropout(dropout)
        
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
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Dictionary containing output and features
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last time step output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Fully connected layers
        fc_out = self.fc(last_output)
        fc_out = self.fc_activation(fc_out)
        fc_out = self.fc_dropout(fc_out)
        
        # Output layer
        output = self.output_layer(fc_out)
        
        # Apply activation based on task type
        if self.task_type == "classification":
            if self.num_classes == 2:
                output = torch.sigmoid(output)
            else:
                output = torch.softmax(output, dim=1)
        # For regression, no activation is applied
        
        return {
            'output': output,
            'features': last_output,
            'lstm_output': lstm_out
        }
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Make predictions with the model."""
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


class CentralizedLSTMTrainer:
    """
    Trainer for centralized LSTM model.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            device: Computing device
        """
        self.config = config
        self.device = device
        
        # Extract configurations
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        # Create model
        self.model = CentralizedLSTM(
            input_size=model_config.get('input_size', 10),
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('lstm_layers', 3),
            num_classes=model_config.get('num_classes', 2),
            fc_size=model_config.get('fc_size', 100),
            dropout=model_config.get('dropout', 0.2),
            task_type=model_config.get('task_type', 'classification')
        ).to(device)
        
        # Training parameters
        self.learning_rate = training_config.get('learning_rate', 0.001)
        self.batch_size = training_config.get('batch_size', 1024)
        self.max_epochs = training_config.get('max_epochs', 100)
        self.patience = training_config.get('patience', 10)
        self.gradient_clip = training_config.get('gradient_clip', 1.0)
        
        # Create optimizer
        optimizer_name = training_config.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Create loss function
        if self.model.task_type == 'classification':
            if self.model.num_classes == 2:
                self.criterion = nn.BCELoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif self.model.task_type == 'regression':
            self.criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _combine_datasets(self, client_datasets: Dict[str, DataLoader]) -> DataLoader:
        """
        Combine all client datasets into a single centralized dataset.
        
        Args:
            client_datasets: Dictionary of client datasets
            
        Returns:
            Combined DataLoader
        """
        all_datasets = []
        
        for client_id, data_loader in client_datasets.items():
            all_datasets.append(data_loader.dataset)
        
        # Combine all datasets
        combined_dataset = ConcatDataset(all_datasets)
        
        # Create new DataLoader
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return combined_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            predictions = outputs['output']
            
            # Calculate loss
            if self.model.task_type == 'classification':
                if self.model.num_classes == 2:
                    if targets.dim() > 1:
                        targets = targets.squeeze()
                    loss = self.criterion(predictions.squeeze(), targets.float())
                    
                    # Calculate accuracy
                    pred_labels = (predictions > 0.5).float().squeeze()
                    correct_predictions += (pred_labels == targets.float()).sum().item()
                else:
                    loss = self.criterion(predictions, targets.long())
                    pred_labels = torch.argmax(predictions, dim=1)
                    correct_predictions += (pred_labels == targets).sum().item()
            elif self.model.task_type == 'regression':
                if targets.dim() > 1:
                    targets = targets.squeeze()
                loss = self.criterion(predictions.squeeze(), targets.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_samples += data.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples if self.model.task_type == 'classification' else 0.0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                predictions = outputs['output']
                
                # Calculate loss
                if self.model.task_type == 'classification':
                    if self.model.num_classes == 2:
                        if targets.dim() > 1:
                            targets = targets.squeeze()
                        loss = self.criterion(predictions.squeeze(), targets.float())
                        
                        pred_labels = (predictions > 0.5).float().squeeze()
                        correct_predictions += (pred_labels == targets.float()).sum().item()
                    else:
                        loss = self.criterion(predictions, targets.long())
                        pred_labels = torch.argmax(predictions, dim=1)
                        correct_predictions += (pred_labels == targets).sum().item()
                elif self.model.task_type == 'regression':
                    if targets.dim() > 1:
                        targets = targets.squeeze()
                    loss = self.criterion(predictions.squeeze(), targets.float())
                
                total_loss += loss.item()
                total_samples += data.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples if self.model.task_type == 'classification' else 0.0
        
        return avg_loss, accuracy
    
    def fit(
        self,
        client_datasets: Dict[str, DataLoader],
        val_datasets: Dict[str, DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Train the centralized model.
        
        Args:
            client_datasets: Dictionary of client training datasets
            val_datasets: Dictionary of client validation datasets
            
        Returns:
            Training results
        """
        logger.info("Starting centralized LSTM training")
        
        # Combine all client datasets
        train_loader = self._combine_datasets(client_datasets)
        
        val_loader = None
        if val_datasets:
            val_loader = self._combine_datasets(val_datasets)
        
        logger.info(f"Combined dataset size: {len(train_loader.dataset)} samples")
        logger.info(f"Training batches: {len(train_loader)}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self.validate_epoch(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )
            
            if val_loader:
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        
        # Restore best model if validation was used
        if val_loader and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        training_results = {
            'total_time': total_time,
            'epochs_trained': len(self.training_history['train_loss']),
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1] if val_loader else None,
            'history': self.training_history
        }
        
        logger.info(f"Centralized training completed in {total_time:.2f} seconds")
        
        return training_results
    
    def evaluate(self, test_datasets: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            test_datasets: Dictionary of test datasets
            
        Returns:
            Evaluation results
        """
        test_loader = self._combine_datasets(test_datasets)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_scores = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                predictions = outputs['output']
                
                all_targets.extend(targets.cpu().numpy())
                
                if self.model.task_type == 'classification':
                    if self.model.num_classes == 2:
                        scores = predictions.squeeze().cpu().numpy()
                        pred_labels = (predictions > 0.5).float().squeeze().cpu().numpy()
                    else:
                        scores = torch.max(predictions, dim=1)[0].cpu().numpy()
                        pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
                    
                    all_scores.extend(scores)
                    all_predictions.extend(pred_labels)
                elif self.model.task_type == 'regression':
                    pred_values = predictions.squeeze().cpu().numpy()
                    all_predictions.extend(pred_values)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        
        results = {}
        
        if self.model.task_type == 'classification':
            results['accuracy'] = accuracy_score(y_true, y_pred)
            results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            results['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        else:
            results['mae'] = mean_absolute_error(y_true, y_pred)
            results['mse'] = mean_squared_error(y_true, y_pred)
            results['rmse'] = np.sqrt(results['mse'])
            results['r2'] = r2_score(y_true, y_pred)
        
        return results
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
        logger.info(f"Centralized LSTM model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        logger.info(f"Centralized LSTM model loaded from {path}")
        return self.model


def create_centralized_lstm_trainer(config: Dict[str, Any]) -> CentralizedLSTMTrainer:
    """
    Factory function to create a centralized LSTM trainer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CentralizedLSTMTrainer instance
    """
    device = torch.device(config.get('training', {}).get('device', 'cpu'))
    return CentralizedLSTMTrainer(config, device)


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
        'training': {
            'learning_rate': 0.001,
            'batch_size': 1024,
            'max_epochs': 50,
            'optimizer': 'adam',
            'device': 'cpu'
        }
    }
    
    # Create trainer
    trainer = create_centralized_lstm_trainer(config)
    
    # Test model creation
    print("Centralized LSTM trainer created successfully")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    
    # Test forward pass
    batch_size = 32
    seq_len = 60
    input_size = 10
    
    x = torch.randn(batch_size, seq_len, input_size)
    output = trainer.model(x)
    
    print(f"Test output shape: {output['output'].shape}")
    print("Centralized LSTM baseline ready for training!")