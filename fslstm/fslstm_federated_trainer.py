"""
Federated Training Module for FSLSTM.

This module implements the federated learning training process with secure aggregation
and privacy-preserving mechanisms for smart building anomaly detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import copy
import random
from collections import defaultdict
import time
from tqdm import tqdm

from ..models.fslstm import FSLSTMModel, create_fslstm_model
from ..utils.privacy import SecureAggregation, DifferentialPrivacy
from ..utils.logger import TrainingLogger
from .aggregation import FederatedAveraging, WeightedAveraging

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Represents a federated learning client (sensor) in the smart building network.
    """
    
    def __init__(
        self,
        client_id: str,
        model: FSLSTMModel,
        train_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device = torch.device('cpu')
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        
        # Training configuration
        training_config = config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 0.001)
        self.local_epochs = config.get('federated', {}).get('local_epochs', 5)
        self.batch_size = training_config.get('batch_size', 1024)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Training metrics
        self.training_history = []
        self.local_updates = 0
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for local training."""
        optimizer_name = self.config.get('training', {}).get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('training', {}).get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on task type."""
        task_type = self.config.get('model', {}).get('task_type', 'classification')
        
        if task_type == 'classification':
            num_classes = self.config.get('model', {}).get('num_classes', 2)
            if num_classes == 2:
                return nn.BCELoss()
            else:
                return nn.CrossEntropyLoss()
        elif task_type == 'regression':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def local_train(self, global_model_params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Perform local training on client data.
        
        Args:
            global_model_params: Global model parameters from server
            
        Returns:
            Dictionary containing local model parameters and training metrics
        """
        # Set global parameters
        self.model.set_parameters(global_model_params)
        self.model.train()
        
        epoch_losses = []
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                predictions = outputs['output']
                
                # Calculate loss
                if self.model.task_type == 'classification':
                    if self.model.num_classes == 2:
                        # Binary classification
                        if targets.dim() > 1:
                            targets = targets.squeeze()
                        loss = self.criterion(predictions.squeeze(), targets.float())
                    else:
                        # Multi-class classification
                        loss = self.criterion(predictions, targets.long())
                elif self.model.task_type == 'regression':
                    if targets.dim() > 1:
                        targets = targets.squeeze()
                    loss = self.criterion(predictions.squeeze(), targets.float())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                total_samples += data.size(0)
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
        
        # Update local training statistics
        self.local_updates += 1
        avg_loss = np.mean(epoch_losses)
        
        training_metrics = {
            'client_id': self.client_id,
            'local_loss': avg_loss,
            'samples_trained': total_samples,
            'local_epochs': self.local_epochs,
            'local_updates': self.local_updates
        }
        
        self.training_history.append(training_metrics)
        
        return {
            'parameters': self.model.get_parameters(),
            'metrics': training_metrics,
            'num_samples': len(self.train_loader.dataset)
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                predictions = outputs['output']
                
                # Calculate loss
                if self.model.task_type == 'classification':
                    if self.model.num_classes == 2:
                        if targets.dim() > 1:
                            targets = targets.squeeze()
                        loss = self.criterion(predictions.squeeze(), targets.float())
                        
                        # Calculate accuracy for binary classification
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
        
        avg_loss = total_loss / len(test_loader)
        
        metrics = {'loss': avg_loss}
        
        if self.model.task_type == 'classification':
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            metrics['accuracy'] = accuracy
        
        return metrics


class FederatedServer:
    """
    Federated learning server that coordinates training across multiple clients.
    """
    
    def __init__(
        self,
        model: FSLSTMModel,
        clients: List[FederatedClient],
        config: Dict[str, Any],
        logger: Optional[TrainingLogger] = None,
        device: torch.device = torch.device('cpu')
    ):
        self.global_model = model.to(device)
        self.clients = clients
        self.config = config
        self.logger = logger
        self.device = device
        
        # Federated learning configuration
        fed_config = config.get('federated', {})
        self.num_rounds = fed_config.get('num_rounds', 50)
        self.clients_per_round = fed_config.get('clients_per_round', len(clients))
        self.min_clients = fed_config.get('min_clients', 1)
        
        # Aggregation method
        aggregation_method = fed_config.get('aggregation', 'fedavg')
        if aggregation_method == 'fedavg':
            self.aggregator = FederatedAveraging()
        elif aggregation_method == 'weighted':
            self.aggregator = WeightedAveraging()
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        # Privacy mechanisms
        privacy_config = config.get('privacy', {})
        self.secure_aggregation = None
        self.differential_privacy = None
        
        if privacy_config.get('secure_aggregation', False):
            self.secure_aggregation = SecureAggregation()
        
        if privacy_config.get('differential_privacy', False):
            epsilon = privacy_config.get('epsilon', 1.0)
            delta = privacy_config.get('delta', 1e-5)
            self.differential_privacy = DifferentialPrivacy(epsilon, delta)
        
        # Training history
        self.training_history = {
            'global_loss': [],
            'client_metrics': [],
            'round_times': []
        }
        
    def select_clients(self, round_num: int) -> List[FederatedClient]:
        """Select clients for current training round."""
        if self.clients_per_round >= len(self.clients):
            return self.clients
        
        # Random selection with seed for reproducibility
        random.seed(round_num)
        selected_clients = random.sample(self.clients, self.clients_per_round)
        
        return selected_clients
    
    def train_round(self, round_num: int) -> Dict[str, Any]:
        """Execute a single federated training round."""
        round_start_time = time.time()
        
        logger.info(f"Starting federated round {round_num + 1}/{self.num_rounds}")
        
        # Select clients for this round
        selected_clients = self.select_clients(round_num)
        logger.info(f"Selected {len(selected_clients)} clients for training")
        
        # Get current global model parameters
        global_params = self.global_model.get_parameters()
        
        # Collect client updates
        client_updates = []
        client_metrics = []
        
        for client in tqdm(selected_clients, desc="Client Training"):
            try:
                # Local training
                client_result = client.local_train(global_params)
                client_updates.append(client_result)
                client_metrics.append(client_result['metrics'])
                
            except Exception as e:
                logger.error(f"Error in client {client.client_id} training: {str(e)}")
                continue
        
        if len(client_updates) < self.min_clients:
            logger.warning(f"Insufficient clients completed training: {len(client_updates)}")
            return None
        
        # Apply privacy mechanisms
        if self.differential_privacy:
            client_updates = self.differential_privacy.add_noise(client_updates)
        
        if self.secure_aggregation:
            client_updates = self.secure_aggregation.aggregate(client_updates)
        
        # Aggregate client updates
        aggregated_params = self.aggregator.aggregate(client_updates)
        
        # Update global model
        self.global_model.set_parameters(aggregated_params)
        
        # Calculate round metrics
        round_time = time.time() - round_start_time
        avg_client_loss = np.mean([m['local_loss'] for m in client_metrics])
        
        round_metrics = {
            'round': round_num + 1,
            'num_clients': len(client_updates),
            'avg_client_loss': avg_client_loss,
            'round_time': round_time,
            'client_metrics': client_metrics
        }
        
        # Log metrics
        if self.logger:
            self.logger.log_round_metrics(round_metrics)
        
        # Store history
        self.training_history['client_metrics'].append(client_metrics)
        self.training_history['round_times'].append(round_time)
        
        logger.info(f"Round {round_num + 1} completed in {round_time:.2f}s")
        logger.info(f"Average client loss: {avg_client_loss:.4f}")
        
        return round_metrics
    
    def train(self) -> Dict[str, Any]:
        """Execute complete federated training process."""
        logger.info("Starting federated training")
        logger.info(f"Configuration: {self.num_rounds} rounds, {self.clients_per_round} clients per round")
        
        training_start_time = time.time()
        
        for round_num in range(self.num_rounds):
            round_metrics = self.train_round(round_num)
            
            if round_metrics is None:
                logger.warning(f"Skipping round {round_num + 1} due to insufficient clients")
                continue
        
        total_training_time = time.time() - training_start_time
        
        training_results = {
            'total_time': total_training_time,
            'num_rounds': self.num_rounds,
            'history': self.training_history,
            'final_model_params': self.global_model.get_parameters()
        }
        
        logger.info(f"Federated training completed in {total_training_time:.2f}s")
        
        return training_results
    
    def evaluate_global_model(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """Evaluate global model on test data from multiple clients."""
        self.global_model.eval()
        
        all_results = {}
        
        for client_id, test_loader in test_loaders.items():
            client_results = {}
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    outputs = self.global_model(data)
                    predictions = outputs['output']
                    
                    # Calculate loss based on task type
                    if self.global_model.task_type == 'classification':
                        if self.global_model.num_classes == 2:
                            criterion = nn.BCELoss()
                            if targets.dim() > 1:
                                targets = targets.squeeze()
                            loss = criterion(predictions.squeeze(), targets.float())
                            
                            pred_labels = (predictions > 0.5).float().squeeze()
                            correct_predictions += (pred_labels == targets.float()).sum().item()
                        else:
                            criterion = nn.CrossEntropyLoss()
                            loss = criterion(predictions, targets.long())
                            pred_labels = torch.argmax(predictions, dim=1)
                            correct_predictions += (pred_labels == targets).sum().item()
                    
                    elif self.global_model.task_type == 'regression':
                        criterion = nn.MSELoss()
                        if targets.dim() > 1:
                            targets = targets.squeeze()
                        loss = criterion(predictions.squeeze(), targets.float())
                    
                    total_loss += loss.item()
                    total_samples += data.size(0)
            
            avg_loss = total_loss / len(test_loader)
            client_results['loss'] = avg_loss
            
            if self.global_model.task_type == 'classification':
                accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
                client_results['accuracy'] = accuracy
            
            all_results[client_id] = client_results
        
        # Calculate overall metrics
        overall_loss = np.mean([r['loss'] for r in all_results.values()])
        overall_results = {'overall_loss': overall_loss}
        
        if self.global_model.task_type == 'classification':
            overall_accuracy = np.mean([r['accuracy'] for r in all_results.values()])
            overall_results['overall_accuracy'] = overall_accuracy
        
        return {
            'client_results': all_results,
            'overall_results': overall_results
        }


class FSLSTMTrainer:
    """
    Main trainer class for FSLSTM federated learning.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[TrainingLogger] = None):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.get('training', {}).get('device', 'cpu'))
        
        # Create global model
        self.global_model = create_fslstm_model(config)
        
        # Federated components will be initialized when training starts
        self.server = None
        self.clients = []
    
    def setup_federated_learning(
        self, 
        client_datasets: Dict[str, DataLoader]
    ) -> None:
        """
        Setup federated learning components.
        
        Args:
            client_datasets: Dictionary mapping client IDs to their DataLoaders
        """
        # Create clients
        self.clients = []
        for client_id, train_loader in client_datasets.items():
            # Create local model copy
            local_model = create_fslstm_model(self.config)
            
            client = FederatedClient(
                client_id=client_id,
                model=local_model,
                train_loader=train_loader,
                config=self.config,
                device=self.device
            )
            self.clients.append(client)
        
        # Create server
        self.server = FederatedServer(
            model=self.global_model,
            clients=self.clients,
            config=self.config,
            logger=self.logger,
            device=self.device
        )
        
        logger.info(f"Federated learning setup complete with {len(self.clients)} clients")
    
    def federated_fit(
        self, 
        client_datasets: Dict[str, DataLoader],
        test_datasets: Optional[Dict[str, DataLoader]] = None
    ) -> Dict[str, Any]:
        """
        Train FSLSTM model using federated learning.
        
        Args:
            client_datasets: Training datasets for each client
            test_datasets: Optional test datasets for evaluation
            
        Returns:
            Training results and metrics
        """
        # Setup federated learning
        self.setup_federated_learning(client_datasets)
        
        # Train the model
        training_results = self.server.train()
        
        # Evaluate if test datasets provided
        if test_datasets:
            evaluation_results = self.server.evaluate_global_model(test_datasets)
            training_results['evaluation'] = evaluation_results
        
        return training_results
    
    def save_model(self, path: str) -> None:
        """Save the trained global model."""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config,
            'model_class': type(self.global_model).__name__
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> FSLSTMModel:
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Create model from config
        model = create_fslstm_model(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        logger.info(f"Model loaded from {path}")
        return model