"""
Examples package for Agricultural ML Showcase

This package contains working examples and demonstrations of the agricultural ML tools.
"""

from .npz_data_loading_example import (
    load_agricultural_npz_dataset,
    process_multi_spectral_data,
    visualize_agricultural_data,
    save_processed_data
)

from .synthetic_example import generate_synthetic_agricultural_data

__all__ = [
    'load_agricultural_npz_dataset',
    'process_multi_spectral_data', 
    'visualize_agricultural_data',
    'save_processed_data',
    'generate_synthetic_agricultural_data'
]
