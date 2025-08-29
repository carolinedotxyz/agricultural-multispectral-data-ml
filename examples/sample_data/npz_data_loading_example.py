#!/usr/bin/env python3
"""
NPZ Data Loading Example for Agricultural ML
Demonstrates practical loading and processing of multi-spectral agricultural datasets.
This example shows real-world data handling patterns used in agricultural ML.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import time

def load_agricultural_npz_dataset(file_path: str):
    """
    Load and validate an agricultural NPZ dataset.
    
    Args:
        file_path: Path to the .npz file
        
    Returns:
        Dictionary containing loaded data and metadata
    """
    print(f"📂 Loading agricultural dataset: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load the NPZ file
    start_time = time.time()
    data = np.load(file_path, allow_pickle=False)
    load_time = time.time() - start_time
    
    print(f"✅ Dataset loaded in {load_time:.3f} seconds")
    
    # Extract available keys
    available_keys = list(data.keys())
    print(f"📁 Available data keys: {available_keys}")
    
    # Load individual arrays
    dataset = {}
    for key in available_keys:
        array = data[key]
        dataset[key] = array
        print(f"  {key}: {array.shape}, {array.dtype}, {array.nbytes / (1024*1024):.2f} MB")
    
    # Close the file
    data.close()
    
    return dataset

def process_multi_spectral_data(dataset: dict):
    """
    Process multi-spectral agricultural data for analysis.
    
    Args:
        dataset: Dictionary containing loaded data arrays
        
    Returns:
        Dictionary containing processed data and statistics
    """
    print("\n🔍 Processing multi-spectral data...")
    
    processed_data = {}
    statistics = {}
    
    # Process RGB data if available
    if 'rgb' in dataset:
        rgb_data = dataset['rgb']
        print(f"🔴 Processing RGB data: {rgb_data.shape}")
        
        # Ensure correct format (N, 3, H, W)
        if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
            rgb_data = np.transpose(rgb_data, (0, 3, 1, 2))
            print(f"  → Transposed to (N, 3, H, W) format")
        
        # Calculate RGB statistics
        rgb_stats = {
            'mean_per_channel': [np.mean(rgb_data[:, i, :, :]) for i in range(3)],
            'std_per_channel': [np.std(rgb_data[:, i, :, :]) for i in range(3)],
            'min_per_channel': [np.min(rgb_data[:, i, :, :]) for i in range(3)],
            'max_per_channel': [np.max(rgb_data[:, i, :, :]) for i in range(3)]
        }
        
        processed_data['rgb'] = rgb_data
        statistics['rgb'] = rgb_stats
        
        print(f"  RGB Statistics:")
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            print(f"    {channel}: mean={rgb_stats['mean_per_channel'][i]:.3f}, "
                  f"std={rgb_stats['std_per_channel'][i]:.3f}")
    
    # Process NDVI data if available
    if 'ndvi' in dataset:
        ndvi_data = dataset['ndvi']
        print(f"🟢 Processing NDVI data: {ndvi_data.shape}")
        
        # Calculate NDVI statistics
        ndvi_stats = {
            'mean': float(np.mean(ndvi_data)),
            'std': float(np.std(ndvi_data)),
            'min': float(np.min(ndvi_data)),
            'max': float(np.max(ndvi_data)),
            'vegetation_high': int(np.sum(ndvi_data > 0.6)),
            'vegetation_moderate': int(np.sum((ndvi_data > 0.2) & (ndvi_data <= 0.6))),
            'vegetation_low': int(np.sum((ndvi_data > 0) & (ndvi_data <= 0.2))),
            'non_vegetation': int(np.sum(ndvi_data <= 0))
        }
        
        processed_data['ndvi'] = ndvi_data
        statistics['ndvi'] = ndvi_stats
        
        print(f"  NDVI Statistics:")
        print(f"    Range: [{ndvi_stats['min']:.3f}, {ndvi_stats['max']:.3f}]")
        print(f"    High vegetation (>0.6): {ndvi_stats['vegetation_high']:,}")
        print(f"    Moderate vegetation (0.2-0.6): {ndvi_stats['vegetation_moderate']:,}")
        print(f"    Low vegetation (0-0.2): {ndvi_stats['vegetation_low']:,}")
    
    # Process cloud mask if available
    if 'cloud_mask' in dataset:
        cloud_data = dataset['cloud_mask']
        print(f"☁️  Processing cloud mask: {cloud_data.shape}")
        
        # Calculate cloud coverage
        cloud_coverage = np.mean(cloud_data)
        cloud_stats = {
            'coverage_percentage': float(cloud_coverage * 100),
            'clear_pixels': int(np.sum(cloud_data == 0)),
            'cloudy_pixels': int(np.sum(cloud_data == 1)),
            'total_pixels': int(cloud_data.size)
        }
        
        processed_data['cloud_mask'] = cloud_data
        statistics['cloud_mask'] = cloud_stats
        
        print(f"  Cloud Statistics:")
        print(f"    Coverage: {cloud_stats['coverage_percentage']:.2f}%")
        print(f"    Clear pixels: {cloud_stats['clear_pixels']:,}")
        print(f"    Cloudy pixels: {cloud_stats['cloudy_pixels']:,}")
    
    return processed_data, statistics

def visualize_agricultural_data(processed_data: dict, sample_index: int = 0):
    """
    Create visualizations of the agricultural data.
    
    Args:
        processed_data: Dictionary containing processed data
        sample_index: Index of the sample to visualize
    """
    print(f"\n🎨 Creating visualizations for sample {sample_index}...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-Spectral Agricultural Data Visualization (Sample {sample_index})', fontsize=16)
    
    # RGB visualization
    if 'rgb' in processed_data:
        rgb_sample = processed_data['rgb'][sample_index].transpose(1, 2, 0)  # (H, W, 3)
        axes[0, 0].imshow(rgb_sample)
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')
        
        # RGB channel histograms
        for i, (color, channel_name) in enumerate([('red', 'Red'), ('green', 'Green'), ('blue', 'Blue')]):
            channel_data = processed_data['rgb'][:, i, :, :].flatten()
            axes[1, i].hist(channel_data, bins=50, alpha=0.7, color=color)
            axes[1, i].set_title(f'{channel_name} Channel Distribution')
            axes[1, i].set_xlabel('Value')
            axes[1, i].set_ylabel('Frequency')
    
    # NDVI visualization
    if 'ndvi' in processed_data:
        ndvi_sample = processed_data['ndvi'][sample_index]
        im1 = axes[0, 1].imshow(ndvi_sample, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0, 1].set_title('NDVI Image')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # NDVI histogram
        ndvi_data = processed_data['ndvi'].flatten()
        axes[0, 2].hist(ndvi_data, bins=50, alpha=0.7, color='green')
        axes[0, 2].set_title('NDVI Distribution')
        axes[0, 2].set_xlabel('NDVI Value')
        axes[0, 2].set_ylabel('Frequency')
    
    # Cloud mask visualization
    if 'cloud_mask' in processed_data:
        cloud_sample = processed_data['cloud_mask'][sample_index]
        axes[1, 0].imshow(cloud_sample, cmap='Blues')
        axes[1, 0].set_title('Cloud Mask')
        axes[1, 0].axis('off')
    
    # Remove unused subplots
    if 'cloud_mask' not in processed_data:
        axes[1, 0].remove()
    
    plt.tight_layout()
    plt.show()

def save_processed_data(processed_data: dict, output_dir: str = "./processed_output"):
    """
    Save processed data to disk for further analysis.
    
    Args:
        processed_data: Dictionary containing processed data
        output_dir: Directory to save processed data
    """
    print(f"\n💾 Saving processed data to: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save individual arrays
    for key, data in processed_data.items():
        output_file = output_path / f"processed_{key}.npy"
        np.save(output_file, data)
        print(f"  Saved {key}: {output_file}")
    
    # Save combined processed dataset
    combined_file = output_path / "processed_agricultural_dataset.npz"
    np.savez_compressed(combined_file, **processed_data)
    print(f"  Saved combined dataset: {combined_file}")
    
    return output_path

def main():
    """Main function demonstrating NPZ data loading and processing."""
    print("🌾 Agricultural NPZ Data Loading Example")
    print("=" * 60)
    
    # Check if we have a sample dataset, if not create one
    sample_dataset_path = "synthetic_agricultural_dataset.npz"
    
    if not os.path.exists(sample_dataset_path):
        print("📊 No sample dataset found. Creating synthetic data first...")
        from synthetic_example import generate_synthetic_agricultural_data
        output_dir = generate_synthetic_agricultural_data()
        sample_dataset_path = output_dir / "synthetic_agricultural_dataset.npz"
    
    try:
        # 1. Load the NPZ dataset
        print("\n1️⃣ Loading NPZ dataset...")
        dataset = load_agricultural_npz_dataset(str(sample_dataset_path))
        
        # 2. Process the multi-spectral data
        print("\n2️⃣ Processing multi-spectral data...")
        processed_data, statistics = process_multi_spectral_data(dataset)
        
        # 3. Display summary statistics
        print("\n3️⃣ Data Summary:")
        total_samples = len(next(iter(processed_data.values())))
        print(f"   Total samples: {total_samples}")
        print(f"   Data modalities: {list(processed_data.keys())}")
        
        # 4. Create visualizations
        print("\n4️⃣ Creating visualizations...")
        visualize_agricultural_data(processed_data)
        
        # 5. Save processed data
        print("\n5️⃣ Saving processed data...")
        output_path = save_processed_data(processed_data)
        
        print(f"\n✅ NPZ data loading example completed successfully!")
        print(f"📁 Processed data saved to: {output_path.absolute()}")
        print(f"🎯 You can now use this data for further agricultural ML analysis!")
        
    except Exception as e:
        print(f"❌ Error in NPZ data loading example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
