"""
FSLSTM: Federated Stacked LSTM for Anomaly Detection in Smart Buildings

A privacy-by-design federated learning framework for anomaly detection in smart buildings
using stacked Long Short-Term Memory (LSTM) networks.
"""

__version__ = "1.0.0"
__author__ = "FSLSTM Contributors"
__email__ = ""
__description__ = "Federated Stacked LSTM for Smart Building Anomaly Detection"

# Core imports
from .models.fslstm import FSLSTMModel, MultiTaskFSLSTM, AttentionFSLSTM, create_fslstm_model
from .training.federated_trainer import FSLSTMTrainer, FederatedClient, FederatedServer
from .data.data_loader import SensorDataset, SensorDataProcessor, FederatedDataLoader
from .evaluation.evaluator import Evaluator, AnomalyDetector
from .utils.config import Config, ModelConfig, FederatedConfig, TrainingConfig, DataConfig

# Baseline models
from .baselines.centralized_lstm import CentralizedLSTM
from .baselines.federated_lr import FederatedLogisticRegression
from .baselines.federated_gru import FederatedGRU

# Utilities
from .utils.logger import TrainingLogger
from .utils.privacy import SecureAggregation, DifferentialPrivacy
from .visualization.plots import ResultVisualizer

# Make commonly used classes available at package level
__all__ = [
    # Core classes
    'FSLSTMModel',
    'MultiTaskFSLSTM', 
    'AttentionFSLSTM',
    'create_fslstm_model',
    'FSLSTMTrainer',
    'FederatedClient',
    'FederatedServer',
    'SensorDataset',
    'SensorDataProcessor', 
    'FederatedDataLoader',
    'Evaluator',
    'AnomalyDetector',
    
    # Configuration
    'Config',
    'ModelConfig',
    'FederatedConfig',
    'TrainingConfig',
    'DataConfig',
    
    # Baselines
    'CentralizedLSTM',
    'FederatedLogisticRegression',
    'FederatedGRU',
    
    # Utilities
    'TrainingLogger',
    'SecureAggregation',
    'DifferentialPrivacy',
    'ResultVisualizer',
    
    # Version info
    '__version__',
    '__author__',
    '__description__'
]

# Package metadata
__package_info__ = {
    'name': 'fslstm',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'email': __email__,
    'url': 'https://github.com/your-username/FSLSTM',
    'license': 'MIT',
    'keywords': [
        'federated learning',
        'anomaly detection',
        'smart buildings',
        'IoT',
        'LSTM',
        'privacy-preserving',
        'deep learning'
    ]
}

def get_version():
    """Get package version."""
    return __version__

def get_package_info():
    """Get package information."""
    return __package_info__

def print_citation():
    """Print citation information for the FSLSTM framework."""
    citation = """
    If you use FSLSTM in your research, please cite:

    @article{fslstm2020,
      title={A Federated Learning Approach to Anomaly Detection in Smart Buildings},
      journal={ACM Transactions on Internet of Things},
      volume={1},
      number={1},
      pages={1--24},
      year={2020},
      publisher={ACM}
    }
    """
    print(citation)

# Configure logging for the package
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())