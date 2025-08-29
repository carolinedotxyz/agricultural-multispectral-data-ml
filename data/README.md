# Data Directory

This directory contains the agricultural data used in the ML showcase.

## Processed Data

### subset_75_all_npz/
Pre-processed agricultural data with the following characteristics:

- **Data Type**: Multi-spectral imagery (RGB + NDVI)
- **Processing**: Normalized values for consistent analysis
- **Quality Filter**: 97%+ cloud-free tiles only
- **Crops**: Corn, Rice, Soybean, Wheat
- **Format**: NPZ files for efficient storage and loading

## Results

*Note: Results directories are created during analysis runs*

## Data Loading

Use the example scripts in `examples/` directory to load and work with this data:

```python
# Example loading
import numpy as np
data = np.load('data/processed/subset_75_all_npz/corn_subset.npz')
```
