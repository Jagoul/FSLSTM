"""
Data Loading and Preprocessing Module for FSLSTM.

This module handles loading and preprocessing of IoT sensor data from smart buildings,
including sensor event logs, energy usage data, and weather information.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SensorDataset(Dataset):
    """
    PyTorch Dataset for sensor time series data.
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize sensor dataset.
        
        Args:
            sequences: Array of shape (n_samples, sequence_length, n_features)
            labels: Array of shape (n_samples,) or (n_samples, n_targets)
            transform: Optional transform to be applied to samples
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels) if labels.dtype == np.float32 or labels.dtype == np.float64 else torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return sequence, label


class SensorDataProcessor:
    """
    Processes raw sensor data into sequences suitable for LSTM training.
    """
    
    def __init__(
        self,
        window_size: int = 600,  # 10 hours in minutes
        sequence_length: int = 60,  # 1 hour sequences
        stride: int = 60,  # 1 hour stride
        normalize: bool = True,
        scaler_type: str = 'standard'
    ):
        """
        Initialize data processor.
        
        Args:
            window_size: Size of the time window in minutes
            sequence_length: Length of each sequence for LSTM
            stride: Stride between sequences
            normalize: Whether to normalize the data
            scaler_type: Type of normalization ('standard', 'minmax')
        """
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.scaler_type = scaler_type
        
        # Initialize scalers
        if scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        # Label encoders for categorical variables
        self.label_encoders = {}
        
        # Fitted flag
        self.is_fitted = False
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        targets: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from time series data.
        
        Args:
            data: Time series data of shape (n_timesteps, n_features)
            targets: Target values of shape (n_timesteps,) or (n_timesteps, n_targets)
        
        Returns:
            Tuple of (sequences, sequence_targets)
        """
        sequences = []
        sequence_targets = []
        
        for i in range(0, len(data) - self.sequence_length + 1, self.stride):
            # Extract sequence
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            # Extract target (use the last timestep of the sequence as target)
            if targets is not None:
                if len(targets.shape) == 1:
                    target = targets[i + self.sequence_length - 1]
                else:
                    target = targets[i + self.sequence_length - 1]
                sequence_targets.append(target)
        
        sequences = np.array(sequences)
        
        if targets is not None:
            sequence_targets = np.array(sequence_targets)
        else:
            sequence_targets = np.array([])
        
        return sequences, sequence_targets
    
    def process_sensor_events(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process sensor event log data.
        
        Args:
            df: DataFrame with columns ['timestamp', 'sensor_id', 'sensor_type', 'value', 'status', 'zone_id']
        
        Returns:
            Tuple of (processed_data, metadata)
        """
        logger.info("Processing sensor event data...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Encode categorical variables
        categorical_columns = ['sensor_type', 'status', 'zone_id']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Create anomaly labels (0: normal, 1: anomalous)
        df['anomaly'] = (df['status'] != 'normal').astype(int)
        
        # Select features for modeling
        feature_columns = ['value', 'hour', 'day_of_week', 'month', 
                          'sensor_type_encoded', 'status_encoded', 'zone_id_encoded']
        
        # Group by sensor and create sequences
        all_sequences = []
        all_targets = []
        sensor_metadata = {}
        
        for sensor_id, sensor_data in df.groupby('sensor_id'):
            if len(sensor_data) < self.sequence_length:
                logger.warning(f"Sensor {sensor_id} has insufficient data: {len(sensor_data)} timesteps")
                continue
            
            # Extract features and targets
            features = sensor_data[feature_columns].values
            targets = sensor_data['anomaly'].values
            
            # Create sequences
            sequences, seq_targets = self.create_sequences(features, targets)
            
            if len(sequences) > 0:
                all_sequences.append(sequences)
                all_targets.append(seq_targets)
                
                sensor_metadata[sensor_id] = {
                    'sensor_type': sensor_data['sensor_type'].iloc[0],
                    'zone_id': sensor_data['zone_id'].iloc[0],
                    'num_sequences': len(sequences),
                    'total_timesteps': len(sensor_data)
                }
        
        if not all_sequences:
            raise ValueError("No valid sequences created from the data")
        
        # Concatenate all sequences
        processed_sequences = np.concatenate(all_sequences, axis=0)
        processed_targets = np.concatenate(all_targets, axis=0)
        
        # Normalize features if requested
        if self.normalize and not self.is_fitted:
            # Reshape for fitting scaler
            n_samples, seq_len, n_features = processed_sequences.shape
            reshaped_data = processed_sequences.reshape(-1, n_features)
            self.feature_scaler.fit(reshaped_data)
            self.is_fitted = True
        
        if self.normalize:
            n_samples, seq_len, n_features = processed_sequences.shape
            reshaped_data = processed_sequences.reshape(-1, n_features)
            normalized_data = self.feature_scaler.transform(reshaped_data)
            processed_sequences = normalized_data.reshape(n_samples, seq_len, n_features)
        
        metadata = {
            'sensor_metadata': sensor_metadata,
            'feature_columns': feature_columns,
            'num_sensors': len(sensor_metadata),
            'total_sequences': len(processed_sequences),
            'sequence_length': self.sequence_length,
            'num_features': processed_sequences.shape[2],
            'label_encoders': self.label_encoders
        }
        
        logger.info(f"Created {len(processed_sequences)} sequences from {len(sensor_metadata)} sensors")
        
        return processed_sequences, processed_targets, metadata
    
    def process_energy_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process energy usage data for regression tasks.
        
        Args:
            df: DataFrame with columns ['timestamp', 'sensor_id', 'energy_consumption', 'appliance_type']
        
        Returns:
            Tuple of (sequences, targets, metadata)
        """
        logger.info("Processing energy usage data...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Encode appliance type
        if 'appliance_type_encoded' not in df.columns:
            if 'appliance_type' not in self.label_encoders:
                self.label_encoders['appliance_type'] = LabelEncoder()
                df['appliance_type_encoded'] = self.label_encoders['appliance_type'].fit_transform(df['appliance_type'])
            else:
                df['appliance_type_encoded'] = self.label_encoders['appliance_type'].transform(df['appliance_type'])
        
        # Feature columns for energy prediction
        feature_columns = ['hour', 'day_of_week', 'month', 'is_weekend', 'appliance_type_encoded']
        
        # Group by sensor and create sequences
        all_sequences = []
        all_targets = []
        sensor_metadata = {}
        
        for sensor_id, sensor_data in df.groupby('sensor_id'):
            if len(sensor_data) < self.sequence_length:
                continue
            
            # Extract features and targets
            features = sensor_data[feature_columns].values
            targets = sensor_data['energy_consumption'].values
            
            # Create sequences
            sequences, seq_targets = self.create_sequences(features, targets)
            
            if len(sequences) > 0:
                all_sequences.append(sequences)
                all_targets.append(seq_targets)
                
                sensor_metadata[sensor_id] = {
                    'appliance_type': sensor_data['appliance_type'].iloc[0],
                    'num_sequences': len(sequences),
                    'avg_consumption': sensor_data['energy_consumption'].mean(),
                    'total_timesteps': len(sensor_data)
                }
        
        if not all_sequences:
            raise ValueError("No valid sequences created from energy data")
        
        # Concatenate all sequences
        processed_sequences = np.concatenate(all_sequences, axis=0)
        processed_targets = np.concatenate(all_targets, axis=0)
        
        # Normalize features and targets
        if self.normalize and not self.is_fitted:
            n_samples, seq_len, n_features = processed_sequences.shape
            reshaped_data = processed_sequences.reshape(-1, n_features)
            self.feature_scaler.fit(reshaped_data)
            self.target_scaler.fit(processed_targets.reshape(-1, 1))
            self.is_fitted = True
        
        if self.normalize:
            n_samples, seq_len, n_features = processed_sequences.shape
            reshaped_data = processed_sequences.reshape(-1, n_features)
            normalized_data = self.feature_scaler.transform(reshaped_data)
            processed_sequences = normalized_data.reshape(n_samples, seq_len, n_features)
            processed_targets = self.target_scaler.transform(processed_targets.reshape(-1, 1)).flatten()
        
        metadata = {
            'sensor_metadata': sensor_metadata,
            'feature_columns': feature_columns,
            'num_sensors': len(sensor_metadata),
            'total_sequences': len(processed_sequences),
            'sequence_length': self.sequence_length,
            'num_features': processed_sequences.shape[2],
            'label_encoders': self.label_encoders,
            'target_stats': {
                'mean': np.mean(processed_targets),
                'std': np.std(processed_targets),
                'min': np.min(processed_targets),
                'max': np.max(processed_targets)
            }
        }
        
        logger.info(f"Created {len(processed_sequences)} energy sequences from {len(sensor_metadata)} sensors")
        
        return processed_sequences, processed_targets, metadata


class FederatedDataLoader:
    """
    Handles data loading and distribution for federated learning scenarios.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        processor: Optional[SensorDataProcessor] = None
    ):
        """
        Initialize federated data loader.
        
        Args:
            config: Configuration dictionary
            processor: Optional data processor instance
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.federated_config = config.get('federated', {})
        
        # Initialize processor
        if processor is None:
            self.processor = SensorDataProcessor(
                window_size=self.data_config.get('window_size', 600),
                sequence_length=self.data_config.get('sequence_length', 60),
                stride=self.data_config.get('stride', 60),
                normalize=self.data_config.get('normalize', True),
                scaler_type=self.data_config.get('scaler_type', 'standard')
            )
        else:
            self.processor = processor
        
        # Data splits
        self.train_split = self.data_config.get('train_split', 0.8)
        self.val_split = self.data_config.get('val_split', 0.1)
        self.test_split = self.data_config.get('test_split', 0.1)
        
        # Batch size
        self.batch_size = config.get('training', {}).get('batch_size', 1024)
        
    def load_sensor_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load and process sensor data from file.
        
        Args:
            data_path: Path to sensor data file
        
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info(f"Loading sensor data from {data_path}")
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Determine task type based on columns
        if 'energy_consumption' in df.columns:
            sequences, targets, metadata = self.processor.process_energy_data(df)
            task_type = 'regression'
        else:
            sequences, targets, metadata = self.processor.process_sensor_events(df)
            task_type = 'classification'
        
        return {
            'sequences': sequences,
            'targets': targets,
            'metadata': metadata,
            'task_type': task_type
        }
    
    def create_client_datasets(
        self, 
        sequences: np.ndarray, 
        targets: np.ndarray,
        metadata: Dict[str, Any],
        split_strategy: str = 'sensor_type'
    ) -> Dict[str, Dict[str, DataLoader]]:
        """
        Create federated datasets for clients.
        
        Args:
            sequences: Processed sequences
            targets: Target values
            metadata: Data metadata
            split_strategy: Strategy for splitting data among clients
        
        Returns:
            Dictionary mapping client IDs to their train/val/test DataLoaders
        """
        logger.info(f"Creating federated datasets with {split_strategy} strategy")
        
        if split_strategy == 'sensor_type':
            return self._split_by_sensor_type(sequences, targets, metadata)
        elif split_strategy == 'random':
            return self._split_randomly(sequences, targets, metadata)
        elif split_strategy == 'zone':
            return self._split_by_zone(sequences, targets, metadata)
        else:
            raise ValueError(f"Unsupported split strategy: {split_strategy}")
    
    def _split_by_sensor_type(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Dict[str, Dict[str, DataLoader]]:
        """Split data by sensor type for federated learning."""
        client_datasets = {}
        sensor_metadata = metadata['sensor_metadata']
        
        # Group sensors by type
        sensor_types = {}
        for sensor_id, info in sensor_metadata.items():
            sensor_type = info['sensor_type']
            if sensor_type not in sensor_types:
                sensor_types[sensor_type] = []
            sensor_types[sensor_type].append(sensor_id)
        
        # Create datasets for each sensor type
        start_idx = 0
        for sensor_type, sensor_ids in sensor_types.items():
            # Calculate the number of sequences for this sensor type
            num_sequences = sum(sensor_metadata[sid]['num_sequences'] for sid in sensor_ids)
            end_idx = start_idx + num_sequences
            
            # Extract sequences for this client
            client_sequences = sequences[start_idx:end_idx]
            client_targets = targets[start_idx:end_idx]
            
            # Split into train/val/test
            train_size = int(self.train_split * len(client_sequences))
            val_size = int(self.val_split * len(client_sequences))
            
            train_sequences = client_sequences[:train_size]
            train_targets = client_targets[:train_size]
            
            val_sequences = client_sequences[train_size:train_size + val_size]
            val_targets = client_targets[train_size:train_size + val_size]
            
            test_sequences = client_sequences[train_size + val_size:]
            test_targets = client_targets[train_size + val_size:]
            
            # Create datasets
            train_dataset = SensorDataset(train_sequences, train_targets)
            val_dataset = SensorDataset(val_sequences, val_targets)
            test_dataset = SensorDataset(test_sequences, test_targets)
            
            # Create data loaders
            client_datasets[sensor_type] = {
                'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
                'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            }
            
            start_idx = end_idx
            
            logger.info(f"Client {sensor_type}: {len(train_sequences)} train, "
                       f"{len(val_sequences)} val, {len(test_sequences)} test sequences")
        
        return client_datasets
    
    def _split_randomly(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Dict[str, Dict[str, DataLoader]]:
        """Split data randomly among clients."""
        num_clients = self.federated_config.get('num_clients', 5)
        client_datasets = {}
        
        # Create random splits
        indices = np.random.permutation(len(sequences))
        client_size = len(sequences) // num_clients
        
        for i in range(num_clients):
            start_idx = i * client_size
            end_idx = (i + 1) * client_size if i < num_clients - 1 else len(sequences)
            
            client_indices = indices[start_idx:end_idx]
            client_sequences = sequences[client_indices]
            client_targets = targets[client_indices]
            
            # Split into train/val/test
            train_size = int(self.train_split * len(client_sequences))
            val_size = int(self.val_split * len(client_sequences))
            
            train_sequences = client_sequences[:train_size]
            train_targets = client_targets[:train_size]
            
            val_sequences = client_sequences[train_size:train_size + val_size]
            val_targets = client_targets[train_size:train_size + val_size]
            
            test_sequences = client_sequences[train_size + val_size:]
            test_targets = client_targets[train_size + val_size:]
            
            # Create datasets
            train_dataset = SensorDataset(train_sequences, train_targets)
            val_dataset = SensorDataset(val_sequences, val_targets)
            test_dataset = SensorDataset(test_sequences, test_targets)
            
            # Create data loaders
            client_datasets[f'client_{i}'] = {
                'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
                'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            }
        
        return client_datasets
    
    def _split_by_zone(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Dict[str, Dict[str, DataLoader]]:
        """Split data by building zones."""
        client_datasets = {}
        sensor_metadata = metadata['sensor_metadata']
        
        # Group sensors by zone
        zones = {}
        for sensor_id, info in sensor_metadata.items():
            zone_id = info.get('zone_id', 'unknown')
            if zone_id not in zones:
                zones[zone_id] = []
            zones[zone_id].append(sensor_id)
        
        # Create datasets for each zone
        start_idx = 0
        for zone_id, sensor_ids in zones.items():
            # Calculate the number of sequences for this zone
            num_sequences = sum(sensor_metadata[sid]['num_sequences'] for sid in sensor_ids)
            end_idx = start_idx + num_sequences
            
            # Extract sequences for this client
            client_sequences = sequences[start_idx:end_idx]
            client_targets = targets[start_idx:end_idx]
            
            # Split into train/val/test
            train_size = int(self.train_split * len(client_sequences))
            val_size = int(self.val_split * len(client_sequences))
            
            train_sequences = client_sequences[:train_size]
            train_targets = client_targets[:train_size]
            
            val_sequences = client_sequences[train_size:train_size + val_size]
            val_targets = client_targets[train_size:train_size + val_size]
            
            test_sequences = client_sequences[train_size + val_size:]
            test_targets = client_targets[train_size + val_size:]
            
            # Create datasets
            train_dataset = SensorDataset(train_sequences, train_targets)
            val_dataset = SensorDataset(val_sequences, val_targets)
            test_dataset = SensorDataset(test_sequences, test_targets)
            
            # Create data loaders
            client_datasets[zone_id] = {
                'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
                'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            }
            
            start_idx = end_idx
        
        return client_datasets