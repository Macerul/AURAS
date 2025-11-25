# backend/utils/__init__.py
"""
Heroes Utils Module
Utility per caricamento e validazione dati
"""

"""
Heroes Utils Module
Utility per caricamento e validazione dati
"""

from .data_loader import load_dataset, validate_datasets, get_dataset_summary

__all__ = [
    'load_dataset',
    'validate_datasets',
    'get_dataset_summary'
]
