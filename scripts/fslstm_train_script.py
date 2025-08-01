#!/usr/bin/env python3
"""
Training script for FSLSTM federated learning model.

This script handles the complete training pipeline for anomaly detection
in smart buildings using federated learning.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fslstm.utils.config import Config, load_config_with_overrides
from fslstm.data.data_loader import FederatedDataLoader
from fslstm.training.federated_trainer import FSLSTMTrainer
from fslstm.utils.logger import TrainingLogger
from fslstm.evaluation.evaluator import Evaluator


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fslstm_training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FSLSTM model for smart building anomaly detection"
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to sensor data file (CSV or Parquet)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for models and logs"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cuda/cpu)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Configuration overrides
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Override number of federated rounds"
    )
    
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=None,
        help="Override clients per round"
    )
    
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
        help="Override local epochs"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without training (for testing configuration)"
    )
    
    return parser.parse_args()


def create_config_overrides(args) -> Dict[str, Any]:
    """Create configuration overrides from command line arguments."""
    overrides = {}
    
    if args.device:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['device'] = args.device
    
    if args.learning_rate:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['learning_rate'] = args.learning_rate
    
    if args.batch_size:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['batch_size'] = args.batch_size
    
    if args.num_rounds:
        overrides['federated'] = overrides.get('federated', {})
        overrides['federated']['num_rounds'] = args.num_rounds
    
    if args.clients_per_round:
        overrides['federated'] = overrides.get('federated', {})
        overrides['federated']['clients_per_round'] = args.clients_per_round
    
    if args.local_epochs:
        overrides['federated'] = overrides.get('federated', {})
        overrides['federated']['local_epochs'] = args.local_epochs
    
    return overrides


def load_and_process_data(config: Config, data_path: str, logger) -> Dict[str, Any]:
    """Load and process sensor data."""
    logger.info(f"Loading data from {data_path}")
    
    # Create data loader
    data_loader = FederatedDataLoader(config.to_dict())
    
    # Load and process data
    data_info = data_loader.load_sensor_data(data_path)
    
    logger.info(f"Loaded {len(data_info['sequences'])} sequences")
    logger.info(f"Task type: {data_info['task_type']}")
    logger.info(f"Sequence shape: {data_info['sequences'].shape}")
    logger.info(f"Targets shape: {data_info['targets'].shape}")
    
    # Create federated datasets
    client_datasets = data_loader.create_client_datasets(
        sequences=data_info['sequences'],
        targets=data_info['targets'],
        metadata=data_info['metadata'],
        split_strategy=config.data.split_strategy
    )
    
    logger.info(f"Created datasets for {len(client_datasets)} clients")
    
    # Log client information
    for client_id, datasets in client_datasets.items():
        train_size = len(datasets['train'].dataset)
        val_size = len(datasets['val'].dataset)
        test_size = len(datasets['test'].dataset)
        logger.info(f"Client {client_id}: {train_size} train, {val_size} val, {test_size} test")
    
    return {
        'client_datasets': client_datasets,
        'metadata': data_info['metadata'],
        'task_type': data_info['task_type']
    }


def setup_experiment_directory(output_dir: str, experiment_name: str = None) -> Path:
    """Setup experiment directory structure."""
    if experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"fslstm_{timestamp}"
    
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)
    
    return exp_dir


def train_model(
    config: Config,
    client_datasets: Dict[str, Dict[str, Any]],
    exp_dir: Path,
    logger,
    resume_path: str = None
) -> Dict[str, Any]:
    """Train the FSLSTM model."""
    logger.info("Starting federated training")
    
    # Setup training logger
    training_logger = TrainingLogger(
        log_dir=exp_dir / "logs",
        experiment_name=exp_dir.name,
        config=config.to_dict()
    )
    
    # Create trainer
    trainer = FSLSTMTrainer(config.to_dict(), logger=training_logger)
    
    # Resume from checkpoint if specified
    if resume_path:
        logger.info(f"Resuming training from {resume_path}")
        # Load checkpoint logic would go here
        # trainer.load_checkpoint(resume_path)
    
    # Prepare datasets
    train_datasets = {cid: datasets['train'] for cid, datasets in client_datasets.items()}
    test_datasets = {cid: datasets['test'] for cid, datasets in client_datasets.items()}
    
    # Train the model
    start_time = time.time()
    
    training_results = trainer.federated_fit(
        client_datasets=train_datasets,
        test_datasets=test_datasets
    )
    
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    model_path = exp_dir / "checkpoints" / "fslstm_final.pth"
    trainer.save_model(str(model_path))
    logger.info(f"Final model saved to {model_path}")
    
    # Save training results
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    results_to_save = convert_for_json(training_results)
    results_to_save['training_time'] = training_time
    
    results_path = exp_dir / "results" / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    logger.info(f"Training results saved to {results_path}")
    
    return training_results


def evaluate_model(
    config: Config,
    client_datasets: Dict[str, Dict[str, Any]],
    model_path: str,
    exp_dir: Path,
    logger
) -> Dict[str, Any]:
    """Evaluate the trained model."""
    logger.info("Starting model evaluation")
    
    # Create evaluator
    evaluator = Evaluator(config.to_dict())
    
    # Load trained model
    trainer = FSLSTMTrainer(config.to_dict())
    model = trainer.load_model(model_path)
    
    # Prepare test datasets
    test_datasets = {cid: datasets['test'] for cid, datasets in client_datasets.items()}
    
    # Evaluate model
    evaluation_results = evaluator.comprehensive_evaluation(
        model=model,
        test_datasets=test_datasets,
        save_dir=exp_dir / "results"
    )
    
    # Print evaluation summary
    logger.info("Evaluation Results:")
    if 'overall_results' in evaluation_results:
        overall = evaluation_results['overall_results']
        for metric, value in overall.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save evaluation results
    import json
    eval_path = exp_dir / "results" / "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {eval_path}")
    
    return evaluation_results


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup experiment directory
    exp_dir = setup_experiment_directory(args.output_dir, args.experiment_name)
    
    # Setup logging
    logger = setup_logging(args.log_level, str(exp_dir / "logs"))
    
    logger.info("=" * 60)
    logger.info("FSLSTM Federated Learning Training")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        overrides = create_config_overrides(args)
        config = load_config_with_overrides(args.config, overrides)
        
        # Save configuration to experiment directory
        config.save(exp_dir / "configs" / "config.yaml")
        
        # Print configuration summary
        config.print_summary()
        
        # Set random seed
        set_seed(config.training.seed)
        logger.info(f"Random seed set to {config.training.seed}")
        
        # Check device availability
        if config.training.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            config.training.device = "cpu"
        
        logger.info(f"Using device: {config.training.device}")
        
        # Load and process data
        data_info = load_and_process_data(config, args.data, logger)
        
        # Update model input size based on data
        input_size = data_info['metadata']['num_features']
        config.model.input_size = input_size
        config.model.task_type = data_info['task_type']
        logger.info(f"Model input size set to {input_size}")
        logger.info(f"Task type: {data_info['task_type']}")
        
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return
        
        # Train model
        training_results = train_model(
            config=config,
            client_datasets=data_info['client_datasets'],
            exp_dir=exp_dir,
            logger=logger,
            resume_path=args.resume
        )
        
        # Evaluate model
        model_path = exp_dir / "checkpoints" / "fslstm_final.pth"
        evaluation_results = evaluate_model(
            config=config,
            client_datasets=data_info['client_datasets'],
            model_path=str(model_path),
            exp_dir=exp_dir,
            logger=logger
        )
        
        logger.info("Training and evaluation completed successfully!")
        logger.info(f"Results saved in: {exp_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()