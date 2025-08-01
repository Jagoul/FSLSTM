"""
Configuration Management for FSLSTM.

This module provides configuration classes and utilities for managing
FSLSTM model and training parameters.
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "FSLSTM"
    type: str = "standard"  # 'standard', 'multitask', 'attention'
    input_size: int = 10
    hidden_size: int = 128
    lstm_layers: int = 3
    num_classes: int = 2
    fc_size: int = 100
    dropout: float = 0.2
    task_type: str = "classification"  # 'classification', 'regression'


@dataclass 
class FederatedConfig:
    """Federated learning configuration parameters."""
    num_clients: int = 180
    clients_per_round: int = 36
    num_rounds: int = 50
    local_epochs: int = 5
    min_clients: int = 1
    aggregation: str = "fedavg"  # 'fedavg', 'weighted'
    client_selection: str = "random"  # 'random', 'all'


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 0.001
    batch_size: int = 1024
    optimizer: str = "adam"  # 'adam', 'sgd'
    momentum: float = 0.9
    weight_decay: float = 1e-4
    loss_function: str = "cross_entropy"  # 'cross_entropy', 'bce', 'mse'
    device: str = "cuda"
    seed: int = 42
    gradient_clip: float = 1.0
    early_stopping: bool = True
    patience: int = 10


@dataclass
class DataConfig:
    """Data configuration parameters."""
    window_size: int = 600  # 10 hours in minutes
    sequence_length: int = 60  # 1 hour sequences
    stride: int = 60  # 1 hour stride
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    normalize: bool = True
    scaler_type: str = "standard"  # 'standard', 'minmax'
    split_strategy: str = "sensor_type"  # 'sensor_type', 'random', 'zone'


@dataclass
class SensorConfig:
    """Sensor configuration parameters."""
    categories: list = field(default_factory=lambda: [
        "lights", "thermostat", "occupancy", "water_leakage", "building_access"
    ])
    num_sensors: int = 180


@dataclass
class PrivacyConfig:
    """Privacy configuration parameters."""
    secure_aggregation: bool = True
    differential_privacy: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    log_level: str = "INFO"
    log_dir: str = "logs"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 10
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "fslstm"


class Config:
    """
    Main configuration class that combines all configuration components.
    """
    
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        federated: Optional[FederatedConfig] = None,
        training: Optional[TrainingConfig] = None,
        data: Optional[DataConfig] = None,
        sensors: Optional[SensorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        logging: Optional[LoggingConfig] = None,
        tasks: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize configuration.
        
        Args:
            model: Model configuration
            federated: Federated learning configuration
            training: Training configuration
            data: Data configuration
            sensors: Sensor configuration
            privacy: Privacy configuration
            logging: Logging configuration
            tasks: Multi-task configuration
        """
        self.model = model or ModelConfig()
        self.federated = federated or FederatedConfig()
        self.training = training or TrainingConfig()
        self.data = data or DataConfig()
        self.sensors = sensors or SensorConfig()
        self.privacy = privacy or PrivacyConfig()
        self.logging = logging or LoggingConfig()
        self.tasks = tasks or {}
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate splits sum to 1.0
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            logger.warning(f"Data splits sum to {total_split}, not 1.0. Normalizing splits.")
            self.data.train_split /= total_split
            self.data.val_split /= total_split
            self.data.test_split /= total_split
        
        # Validate federated learning parameters
        if self.federated.clients_per_round > self.federated.num_clients:
            logger.warning("clients_per_round > num_clients. Setting clients_per_round = num_clients")
            self.federated.clients_per_round = self.federated.num_clients
        
        # Validate model parameters
        if self.model.lstm_layers < 1:
            raise ValueError("lstm_layers must be >= 1")
        
        if self.model.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        
        if self.model.dropout < 0.0 or self.model.dropout > 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
        
        # Validate training parameters
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        
        if self.training.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        # Validate privacy parameters
        if self.privacy.differential_privacy:
            if self.privacy.epsilon <= 0:
                raise ValueError("epsilon must be > 0 for differential privacy")
            if self.privacy.delta <= 0 or self.privacy.delta >= 1:
                raise ValueError("delta must be in (0, 1) for differential privacy")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        model_config = ModelConfig(**config_dict.get('model', {}))
        federated_config = FederatedConfig(**config_dict.get('federated', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        sensors_config = SensorConfig(**config_dict.get('sensors', {}))
        privacy_config = PrivacyConfig(**config_dict.get('privacy', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        tasks = config_dict.get('tasks', {})
        
        return cls(
            model=model_config,
            federated=federated_config,
            training=training_config,
            data=data_config,
            sensors=sensors_config,
            privacy=privacy_config,
            logging=logging_config,
            tasks=tasks
        )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Config instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model': asdict(self.model),
            'federated': asdict(self.federated),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'sensors': asdict(self.sensors),
            'privacy': asdict(self.privacy),
            'logging': asdict(self.logging),
            'tasks': self.tasks
        }
    
    def save(self, config_path: Union[str, Path]):
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to {config_path}")
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        for section, values in updates.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                if hasattr(section_config, '__dict__'):
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
                        else:
                            logger.warning(f"Unknown parameter {key} in section {section}")
                else:
                    setattr(self, section, values)
            else:
                logger.warning(f"Unknown configuration section: {section}")
        
        # Re-validate after updates
        self._validate()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary for model creation."""
        config_dict = asdict(self.model)
        config_dict['data'] = asdict(self.data)
        config_dict['tasks'] = self.tasks
        return {'model': config_dict, 'data': asdict(self.data), 'tasks': self.tasks}
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 50)
        print("FSLSTM Configuration Summary")
        print("=" * 50)
        
        print(f"\nModel Configuration:")
        print(f"  Type: {self.model.type}")
        print(f"  LSTM Layers: {self.model.lstm_layers}")
        print(f"  Hidden Size: {self.model.hidden_size}")
        print(f"  Task Type: {self.model.task_type}")
        print(f"  Dropout: {self.model.dropout}")
        
        print(f"\nFederated Learning Configuration:")
        print(f"  Clients: {self.federated.num_clients}")
        print(f"  Clients per Round: {self.federated.clients_per_round}")
        print(f"  Training Rounds: {self.federated.num_rounds}")
        print(f"  Local Epochs: {self.federated.local_epochs}")
        print(f"  Aggregation: {self.federated.aggregation}")
        
        print(f"\nTraining Configuration:")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Optimizer: {self.training.optimizer}")
        print(f"  Device: {self.training.device}")
        
        print(f"\nData Configuration:")
        print(f"  Sequence Length: {self.data.sequence_length}")
        print(f"  Window Size: {self.data.window_size}")
        print(f"  Train/Val/Test Split: {self.data.train_split:.1f}/{self.data.val_split:.1f}/{self.data.test_split:.1f}")
        print(f"  Normalization: {self.data.normalize}")
        
        print(f"\nPrivacy Configuration:")
        print(f"  Secure Aggregation: {self.privacy.secure_aggregation}")
        print(f"  Differential Privacy: {self.privacy.differential_privacy}")
        if self.privacy.differential_privacy:
            print(f"  Epsilon: {self.privacy.epsilon}")
            print(f"  Delta: {self.privacy.delta}")
        
        if self.tasks:
            print(f"\nMulti-Task Configuration:")
            for task_name, task_config in self.tasks.items():
                print(f"  {task_name}: {task_config}")
        
        print("=" * 50)


def create_default_config() -> Config:
    """Create default configuration for smart building anomaly detection."""
    # Multi-task configuration for different sensor types
    tasks = {
        "occupancy": {"type": "classification", "classes": 2},
        "temperature": {"type": "regression", "target": "energy_consumption"},
        "lighting": {"type": "classification", "classes": 2},
        "water_leakage": {"type": "classification", "classes": 2},
        "building_access": {"type": "classification", "classes": 2}
    }
    
    return Config(
        model=ModelConfig(
            name="FSLSTM",
            type="standard",
            input_size=7,  # Based on sensor event features
            hidden_size=128,
            lstm_layers=3,
            num_classes=2,
            fc_size=100,
            dropout=0.2,
            task_type="classification"
        ),
        federated=FederatedConfig(
            num_clients=180,
            clients_per_round=36,
            num_rounds=50,
            local_epochs=5,
            aggregation="fedavg"
        ),
        training=TrainingConfig(
            learning_rate=0.001,
            batch_size=1024,
            optimizer="adam",
            device="cuda",
            gradient_clip=1.0
        ),
        data=DataConfig(
            window_size=600,
            sequence_length=60,
            stride=60,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            split_strategy="sensor_type"
        ),
        sensors=SensorConfig(
            categories=[
                "lights", "thermostat", "occupancy", 
                "water_leakage", "building_access"
            ],
            num_sensors=180
        ),
        privacy=PrivacyConfig(
            secure_aggregation=True,
            differential_privacy=False
        ),
        tasks=tasks
    )


def create_regression_config() -> Config:
    """Create configuration for energy consumption regression task."""
    config = create_default_config()
    
    # Update for regression task
    config.model.task_type = "regression"
    config.model.num_classes = 1
    config.training.loss_function = "mse"
    
    # Single task for energy prediction
    config.tasks = {
        "energy_prediction": {"type": "regression", "target": "energy_consumption"}
    }
    
    return config


def create_multitask_config() -> Config:
    """Create configuration for multi-task learning."""
    config = create_default_config()
    
    # Update for multi-task
    config.model.type = "multitask"
    
    # Define multiple tasks
    config.tasks = {
        "occupancy_detection": {"type": "classification", "classes": 2},
        "energy_prediction": {"type": "regression", "target": "energy_consumption"},
        "fault_detection": {"type": "classification", "classes": 2},
        "zone_classification": {"type": "classification", "classes": 5}
    }
    
    return config


def load_config_with_overrides(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration from file and apply overrides.
    
    Args:
        config_path: Path to configuration file
        overrides: Optional dictionary of configuration overrides
        
    Returns:
        Config instance with overrides applied
    """
    config = Config.from_file(config_path)
    
    if overrides:
        config.update(overrides)
    
    return config


# Configuration templates for different scenarios
CONFIG_TEMPLATES = {
    'default': create_default_config,
    'regression': create_regression_config,
    'multitask': create_multitask_config
}


def get_config_template(template_name: str) -> Config:
    """
    Get a configuration template by name.
    
    Args:
        template_name: Name of the template ('default', 'regression', 'multitask')
        
    Returns:
        Config instance for the specified template
    """
    if template_name not in CONFIG_TEMPLATES:
        available = list(CONFIG_TEMPLATES.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")
    
    return CONFIG_TEMPLATES[template_name]()


if __name__ == "__main__":
    # Example usage
    
    # Create default configuration
    config = create_default_config()
    config.print_summary()
    
    # Save configuration
    config.save("example_config.yaml")
    
    # Load configuration
    loaded_config = Config.from_file("example_config.yaml")
    
    # Apply overrides
    overrides = {
        'training': {'learning_rate': 0.0005, 'batch_size': 512},
        'model': {'hidden_size': 256}
    }
    loaded_config.update(overrides)
    
    print("\nConfiguration with overrides:")
    loaded_config.print_summary()