"""
Federated Aggregation Methods for FSLSTM.

This module implements various aggregation algorithms for combining
client model updates in federated learning scenarios.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class FederatedAggregator(ABC):
    """
    Abstract base class for federated aggregation methods.
    """
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates.
        
        Args:
            client_updates: List of client update dictionaries
            
        Returns:
            Aggregated model parameters
        """
        pass


class FederatedAveraging(FederatedAggregator):
    """
    Federated Averaging (FedAvg) algorithm.
    
    This is the standard federated learning aggregation method that computes
    a weighted average of client model parameters based on their dataset sizes.
    """
    
    def __init__(self, weighted: bool = True):
        """
        Initialize FedAvg aggregator.
        
        Args:
            weighted: Whether to weight by client dataset sizes
        """
        self.weighted = weighted
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using federated averaging.
        
        Args:
            client_updates: List of dictionaries containing:
                - 'parameters': Client model parameters
                - 'num_samples': Number of training samples
                - 'metrics': Training metrics
        
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        logger.debug(f"Aggregating updates from {len(client_updates)} clients")
        
        # Get first client's parameters to initialize structure
        first_params = client_updates[0]['parameters']
        aggregated_params = {}
        
        # Initialize aggregated parameters
        for param_name in first_params.keys():
            aggregated_params[param_name] = torch.zeros_like(first_params[param_name])
        
        # Calculate weights
        if self.weighted:
            total_samples = sum(update['num_samples'] for update in client_updates)
            weights = [update['num_samples'] / total_samples for update in client_updates]
        else:
            weights = [1.0 / len(client_updates) for _ in client_updates]
        
        # Aggregate parameters
        for i, client_update in enumerate(client_updates):
            client_params = client_update['parameters']
            weight = weights[i]
            
            for param_name in client_params.keys():
                aggregated_params[param_name] += weight * client_params[param_name]
        
        logger.debug(f"Aggregation completed with weights: {[f'{w:.3f}' for w in weights]}")
        
        return aggregated_params


class WeightedAveraging(FederatedAggregator):
    """
    Weighted averaging with custom weighting schemes.
    """
    
    def __init__(self, weighting_scheme: str = "uniform"):
        """
        Initialize weighted averaging.
        
        Args:
            weighting_scheme: Weighting scheme ('uniform', 'loss_based', 'performance_based')
        """
        self.weighting_scheme = weighting_scheme
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate with custom weighting schemes."""
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Get first client's parameters
        first_params = client_updates[0]['parameters']
        aggregated_params = {}
        
        # Initialize aggregated parameters
        for param_name in first_params.keys():
            aggregated_params[param_name] = torch.zeros_like(first_params[param_name])
        
        # Calculate weights based on scheme
        weights = self._calculate_weights(client_updates)
        
        # Aggregate parameters
        for i, client_update in enumerate(client_updates):
            client_params = client_update['parameters']
            weight = weights[i]
            
            for param_name in client_params.keys():
                aggregated_params[param_name] += weight * client_params[param_name]
        
        return aggregated_params
    
    def _calculate_weights(self, client_updates: List[Dict[str, Any]]) -> List[float]:
        """Calculate weights based on the specified scheme."""
        if self.weighting_scheme == "uniform":
            return [1.0 / len(client_updates) for _ in client_updates]
        
        elif self.weighting_scheme == "loss_based":
            # Weight inversely proportional to loss (better clients get higher weight)
            losses = [update['metrics']['local_loss'] for update in client_updates]
            
            # Invert losses and normalize
            inv_losses = [1.0 / (loss + 1e-8) for loss in losses]
            total_inv_loss = sum(inv_losses)
            weights = [inv_loss / total_inv_loss for inv_loss in inv_losses]
            
            return weights
        
        elif self.weighting_scheme == "performance_based":
            # Weight based on training performance
            num_samples = [update['num_samples'] for update in client_updates]
            losses = [update['metrics']['local_loss'] for update in client_updates]
            
            # Combine sample size and performance
            performance_scores = []
            for i in range(len(client_updates)):
                # Higher samples and lower loss = higher score
                score = num_samples[i] / (losses[i] + 1e-8)
                performance_scores.append(score)
            
            total_score = sum(performance_scores)
            weights = [score / total_score for score in performance_scores]
            
            return weights
        
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")


class FederatedProx(FederatedAggregator):
    """
    FedProx aggregation with proximal term handling.
    """
    
    def __init__(self, mu: float = 0.01):
        """
        Initialize FedProx aggregator.
        
        Args:
            mu: Proximal term coefficient
        """
        self.mu = mu
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate using FedProx approach.
        
        Note: This is a simplified version. Full FedProx requires modifications
        to the client training procedure as well.
        """
        # For now, use standard FedAvg aggregation
        # The proximal term is typically handled during client training
        fedavg = FederatedAveraging(weighted=True)
        return fedavg.aggregate(client_updates)


class FederatedNova(FederatedAggregator):
    """
    FedNova aggregation that normalizes client updates.
    """
    
    def __init__(self):
        """Initialize FedNova aggregator."""
        pass
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate using FedNova normalization.
        
        This method normalizes client updates by their effective learning rates.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Get first client's parameters
        first_params = client_updates[0]['parameters']
        aggregated_params = {}
        
        # Initialize aggregated parameters
        for param_name in first_params.keys():
            aggregated_params[param_name] = torch.zeros_like(first_params[param_name])
        
        # Calculate normalized weights
        total_samples = sum(update['num_samples'] for update in client_updates)
        local_epochs = [update['metrics'].get('local_epochs', 1) for update in client_updates]
        
        # Normalize by effective batch size
        normalized_weights = []
        total_normalized_weight = 0
        
        for i, client_update in enumerate(client_updates):
            num_samples = client_update['num_samples']
            epochs = local_epochs[i]
            
            # Effective learning rate normalization
            effective_weight = num_samples * epochs
            normalized_weights.append(effective_weight)
            total_normalized_weight += effective_weight
        
        # Normalize weights
        normalized_weights = [w / total_normalized_weight for w in normalized_weights]
        
        # Aggregate parameters
        for i, client_update in enumerate(client_updates):
            client_params = client_update['parameters']
            weight = normalized_weights[i]
            
            for param_name in client_params.keys():
                aggregated_params[param_name] += weight * client_params[param_name]
        
        return aggregated_params


class AdaptiveAggregation(FederatedAggregator):
    """
    Adaptive aggregation that adjusts weights based on client reliability.
    """
    
    def __init__(self, alpha: float = 0.1, window_size: int = 5):
        """
        Initialize adaptive aggregator.
        
        Args:
            alpha: Learning rate for weight adaptation
            window_size: Window size for performance tracking
        """
        self.alpha = alpha
        self.window_size = window_size
        self.client_history = {}  # Track client performance history
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate with adaptive weighting based on client reliability."""
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Update client history
        self._update_client_history(client_updates)
        
        # Get first client's parameters
        first_params = client_updates[0]['parameters']
        aggregated_params = {}
        
        # Initialize aggregated parameters
        for param_name in first_params.keys():
            aggregated_params[param_name] = torch.zeros_like(first_params[param_name])
        
        # Calculate adaptive weights
        weights = self._calculate_adaptive_weights(client_updates)
        
        # Aggregate parameters
        for i, client_update in enumerate(client_updates):
            client_params = client_update['parameters']
            weight = weights[i]
            
            for param_name in client_params.keys():
                aggregated_params[param_name] += weight * client_params[param_name]
        
        return aggregated_params
    
    def _update_client_history(self, client_updates: List[Dict[str, Any]]):
        """Update performance history for each client."""
        for update in client_updates:
            client_id = update['metrics']['client_id']
            loss = update['metrics']['local_loss']
            
            if client_id not in self.client_history:
                self.client_history[client_id] = []
            
            self.client_history[client_id].append(loss)
            
            # Keep only recent history
            if len(self.client_history[client_id]) > self.window_size:
                self.client_history[client_id] = self.client_history[client_id][-self.window_size:]
    
    def _calculate_adaptive_weights(self, client_updates: List[Dict[str, Any]]) -> List[float]:
        """Calculate adaptive weights based on client reliability."""
        weights = []
        reliability_scores = []
        
        for update in client_updates:
            client_id = update['metrics']['client_id']
            num_samples = update['num_samples']
            
            # Calculate reliability score
            if client_id in self.client_history and len(self.client_history[client_id]) > 1:
                # Use consistency (inverse of loss variance) as reliability measure
                losses = self.client_history[client_id]
                consistency = 1.0 / (np.var(losses) + 1e-8)
                avg_loss = np.mean(losses)
                
                # Combine consistency and performance
                reliability = consistency / (avg_loss + 1e-8)
            else:
                # Default reliability for new clients
                reliability = 1.0
            
            # Combine with dataset size
            score = reliability * np.sqrt(num_samples)
            reliability_scores.append(score)
        
        # Normalize weights
        total_score = sum(reliability_scores)
        weights = [score / total_score for score in reliability_scores]
        
        return weights


class SecureAggregation(FederatedAggregator):
    """
    Secure aggregation with privacy protection.
    
    This is a simplified implementation. In practice, secure aggregation
    would use cryptographic protocols like secure multi-party computation.
    """
    
    def __init__(self, noise_scale: float = 0.01):
        """
        Initialize secure aggregator.
        
        Args:
            noise_scale: Scale of noise added for privacy
        """
        self.noise_scale = noise_scale
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate with basic privacy protection.
        
        Note: This is a simplified implementation for demonstration.
        Real secure aggregation requires cryptographic protocols.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Use standard FedAvg for aggregation
        fedavg = FederatedAveraging(weighted=True)
        aggregated_params = fedavg.aggregate(client_updates)
        
        # Add noise for privacy (simplified approach)
        if self.noise_scale > 0:
            for param_name in aggregated_params.keys():
                noise = torch.randn_like(aggregated_params[param_name]) * self.noise_scale
                aggregated_params[param_name] += noise
        
        logger.debug(f"Applied secure aggregation with noise scale: {self.noise_scale}")
        
        return aggregated_params


def create_aggregator(method: str, **kwargs) -> FederatedAggregator:
    """
    Factory function to create aggregation methods.
    
    Args:
        method: Aggregation method name
        **kwargs: Additional arguments for the aggregator
        
    Returns:
        FederatedAggregator instance
    """
    method = method.lower()
    
    if method == "fedavg":
        return FederatedAveraging(**kwargs)
    elif method == "weighted":
        return WeightedAveraging(**kwargs)
    elif method == "fedprox":
        return FederatedProx(**kwargs)
    elif method == "fednova":
        return FederatedNova(**kwargs)
    elif method == "adaptive":
        return AdaptiveAggregation(**kwargs)
    elif method == "secure":
        return SecureAggregation(**kwargs)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


if __name__ == "__main__":
    # Example usage and testing
    
    # Create dummy client updates
    dummy_updates = []
    for i in range(3):
        params = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(2, 10),
            'layer2.bias': torch.randn(2)
        }
        
        update = {
            'parameters': params,
            'num_samples': np.random.randint(100, 1000),
            'metrics': {
                'client_id': f'client_{i}',
                'local_loss': np.random.uniform(0.1, 1.0),
                'local_epochs': 5
            }
        }
        dummy_updates.append(update)
    
    # Test different aggregation methods
    methods = ['fedavg', 'weighted', 'adaptive', 'secure']
    
    for method in methods:
        print(f"\nTesting {method.upper()} aggregation:")
        aggregator = create_aggregator(method)
        aggregated = aggregator.aggregate(dummy_updates)
        
        print(f"  Aggregated parameters keys: {list(aggregated.keys())}")
        print(f"  First parameter shape: {aggregated['layer1.weight'].shape}")
        print(f"  Parameter norm: {torch.norm(aggregated['layer1.weight']):.4f}")
    
    print("\nAggregation methods tested successfully!")