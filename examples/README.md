# Sample Data Directory

This directory contains sample datasets and examples for testing the Agricultural ML Showcase tools.

## Contents

- **README.md**: This file
- **npz_data_loading_example.py**: Practical example of loading and processing NPZ data

## Quick Start

### Load and Process NPZ Data
```bash
python npz_data_loading_example.py
```

## What Each Script Does

### **npz_data_loading_example.py**
- **Loads NPZ files** and extracts data arrays
- **Processes multi-spectral data** (RGB + NDVI)
- **Calculates statistics** for each modality
- **Creates visualizations** of the data
- **Saves processed data** for further analysis
- **Shows real-world data handling patterns**

## Data Format

All sample data follows the standard multi-spectral agricultural format:
- **RGB**: (N, 3, H, W) - Normalized RGB values [0, 1]
- **NDVI**: (N, H, W) - Vegetation index values [-1, 1]
- **Cloud Mask**: (N, H, W) - Binary cloud coverage [0, 1]
- **Patch IDs**: (N,) - Unique identifiers for each sample

## NPZ Data Loading Example Features

### **Data Loading**
```python
# Load NPZ file
dataset = load_agricultural_npz_dataset("path/to/dataset.npz")

# Access data arrays
rgb_data = dataset['rgb']      # Shape: (N, 3, H, W)
ndvi_data = dataset['ndvi']    # Shape: (N, H, W)
cloud_mask = dataset['cloud_mask']  # Shape: (N, H, W)
```

### **Data Processing**
- **Format conversion** (RGB transpose for ML frameworks)
- **Statistical analysis** (mean, std, min, max per channel)
- **Vegetation classification** (high, moderate, low based on NDVI)
- **Cloud coverage analysis** (percentage, pixel counts)

### **Visualization**
- **RGB image display** with proper color mapping
- **NDVI heatmaps** with vegetation color coding
- **Cloud mask overlays** for quality assessment
- **Histogram analysis** for data distribution understanding

### **Data Export**
- **Individual arrays** saved as .npy files
- **Combined dataset** saved as .npz file
- **Processed data** ready for ML training

## Usage Examples

### **Basic Data Loading**
```python
from npz_data_loading_example import load_agricultural_npz_dataset

# Load your agricultural dataset
dataset = load_agricultural_np_data("your_data.npz")
print(f"Loaded {len(dataset['rgb'])} RGB images")
```

### **Data Processing Pipeline**
```python
from npz_data_loading_example import process_multi_spectral_data

# Process the loaded data
processed_data, statistics = process_multi_spectral_data(dataset)

# Access statistics
print(f"NDVI range: {statistics['ndvi']['min']:.3f} to {statistics['ndvi']['max']:.3f}")
print(f"Cloud coverage: {statistics['cloud_mask']['coverage_percentage']:.1f}%")
```

### **Custom Analysis**
```python
# Access raw arrays for custom processing
rgb_data = processed_data['rgb']  # (N, 3, H, W)
ndvi_data = processed_data['ndvi']  # (N, H, W)

# Your custom analysis here
# Example: Calculate vegetation density
vegetation_density = np.mean(ndvi_data > 0.3, axis=(1, 2))
```

## Use Cases

Use these examples to:
- **Test the showcase tools** with realistic data
- **Learn NPZ data handling** patterns
- **Develop your own analysis pipelines**
- **Validate tool functionality**
- **Understand agricultural data formats**
- **Practice multi-spectral data processing**

## Requirements

- NumPy for data handling
- Matplotlib for visualizations
- Pathlib for file operations

## Note

These are synthetic datasets for demonstration purposes. Real agricultural data may have different characteristics and requirements. The NPZ loading example shows the exact patterns you'd use with real agricultural datasets from satellite imagery, drone surveys, or field sensors.

## Next Steps

After running these examples:
1. **Modify the scripts** for your specific data
2. **Apply to real agricultural datasets** from your research
3. **Extend the functionality** for your specific use cases
