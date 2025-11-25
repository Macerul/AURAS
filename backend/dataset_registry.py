"""
Dataset Registry - Manages available datasets for comparison
Adapted for HEROES/backend/ structure
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

# Ensure backend directory is in path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from utils.data_loader import load_dataset, get_dataset_summary


class DatasetRegistry:
    """
    Registry for managing datasets in the application
    """
    
    def __init__(self, registry_path='/tmp/heroes_uploads/dataset_registry.json'):
        self.registry_path = registry_path
        self.datasets = {}
        self.load_registry()
    
    def load_registry(self):
        """Load registry from disk"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.datasets = json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                self.datasets = {}
        else:
            self.datasets = {}
    
    def save_registry(self):
        """Save registry to disk"""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.datasets, f, indent=2)
    
    def register_dataset(self, dataset_id: str, original_path: str, augmented_path: str, 
                        description: str = "", model_name: str = "default"):
        """
        Register a dataset pair (original + augmented)
        
        Args:
            dataset_id: Unique identifier for the dataset
            original_path: Path to original dataset file
            augmented_path: Path to augmented dataset file
            description: Human-readable description
            model_name: Name of the model/method used for augmentation
        """
        # Load datasets to get metadata
        try:
            df_original = load_dataset(original_path)
            df_augmented = load_dataset(augmented_path)
            
            summary_orig = get_dataset_summary(df_original, "Original")
            summary_aug = get_dataset_summary(df_augmented, "Augmented")
            
            # Create dataset entry
            self.datasets[dataset_id] = {
                'id': dataset_id,
                'description': description,
                'model_name': model_name,
                'original_path': original_path,
                'augmented_path': augmented_path,
                'registered_at': datetime.now().isoformat(),
                'original_summary': {
                    'n_rows': summary_orig['n_rows'],
                    'n_columns': summary_orig['n_columns'],
                    'n_numeric': summary_orig['n_numeric'],
                    'n_categorical': summary_orig['n_categorical'],
                    'columns': summary_orig['columns']
                },
                'augmented_summary': {
                    'n_rows': summary_aug['n_rows'],
                    'n_columns': summary_aug['n_columns'],
                    'n_numeric': summary_aug['n_numeric'],
                    'n_categorical': summary_aug['n_categorical'],
                    'columns': summary_aug['columns']
                }
            }
            
            self.save_registry()
            return True
        except Exception as e:
            print(f"Error registering dataset: {e}")
            return False
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset metadata by ID"""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self) -> List[Dict]:
        """List all registered datasets"""
        return list(self.datasets.values())
    
    def get_models(self, dataset_id: str) -> List[str]:
        """Get list of models/variants for a dataset"""
        dataset = self.datasets.get(dataset_id)
        if dataset:
            return [dataset.get('model_name', 'default')]
        return []
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Remove a dataset from registry"""
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]
            self.save_registry()
            return True
        return False
    
    def get_cache_key(self, dataset_id: str, model_name: str, filters: Dict) -> str:
        """Generate cache key for computed metrics"""
        key_data = f"{dataset_id}_{model_name}_{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()


# Global registry instance
_registry = None

def get_registry() -> DatasetRegistry:
    """Get or create global registry instance"""
    global _registry
    if _registry is None:
        _registry = DatasetRegistry()
    return _registry
