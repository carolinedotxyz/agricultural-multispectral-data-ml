#!/usr/bin/env python3
"""
Synthetic Agricultural Data Generator
Creates sample datasets for testing the Agricultural ML Showcase tools.
"""

import numpy as np
import os
from pathlib import Path

def generate_synthetic_agricultural_data():
    """Generate synthetic agricultural data for demonstration purposes."""
    
    print("🌾 Generating synthetic agricultural data...")
    
    # Parameters
    N, H, W = 100, 64, 64
    
    # Create output directory
    output_dir = Path("synthetic_data")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Generate RGB data (N, 3, H, W)
    print("  🔴 Generating RGB data...")
    rgb_data = np.random.rand(N, 3, H, W).astype(np.float32)
    
    # 2. Generate NDVI data (N, H, W) - simulate vegetation patterns
    print("  🟢 Generating NDVI data...")
    ndvi_data = np.random.rand(N, H, W).astype(np.float32) * 2 - 1  # Range [-1, 1]
    
    # Add some correlation between RGB green channel and NDVI
    green_channel = rgb_data[:, 1, :, :]
    ndvi_data = 0.3 * green_channel + 0.7 * ndvi_data
    ndvi_data = np.clip(ndvi_data, -1, 1)
    
    # 3. Generate cloud mask (N, H, W)
    print("  ☁️  Generating cloud mask data...")
    cloud_mask = np.random.randint(0, 2, (N, H, W), dtype=np.uint8)
    
    # 4. Generate patch IDs
    print("  🏷️  Generating patch identifiers...")
    patch_ids = np.arange(N, dtype=np.int64)
    
    # Save individual files
    print("  💾 Saving individual data files...")
    
    # RGB data
    np.save(output_dir / "synthetic_rgb.npy", rgb_data)
    
    # NDVI data
    np.save(output_dir / "synthetic_ndvi.npy", ndvi_data)
    
    # Cloud mask
    np.save(output_dir / "synthetic_cloud_mask.npy", cloud_mask)
    
    # Patch IDs
    np.save(output_dir / "synthetic_patch_ids.npy", patch_ids)
    
    # Save combined dataset
    print("  💾 Saving combined dataset...")
    combined_path = output_dir / "synthetic_agricultural_dataset.npz"
    np.savez_compressed(
        combined_path,
        rgb=rgb_data,
        ndvi=ndvi_data,
        cloud_mask=cloud_mask,
        patch_ids=patch_ids
    )
    
    # Generate metadata
    metadata = {
        "dataset_info": {
            "name": "Synthetic Agricultural Dataset",
            "description": "Synthetic multi-spectral agricultural data for testing",
            "generated_by": "Agricultural ML Showcase",
            "timestamp": "2025-01-28"
        },
        "data_specifications": {
            "total_samples": N,
            "image_height": H,
            "image_width": W,
            "rgb_shape": rgb_data.shape,
            "ndvi_shape": ndvi_data.shape,
            "cloud_mask_shape": cloud_mask.shape,
            "patch_ids_shape": patch_ids.shape
        },
        "data_characteristics": {
            "rgb_range": [float(np.min(rgb_data)), float(np.max(rgb_data))],
            "ndvi_range": [float(np.min(ndvi_data)), float(np.max(ndvi_data))],
            "cloud_coverage_percentage": float(np.mean(cloud_mask) * 100),
            "data_types": {
                "rgb": str(rgb_data.dtype),
                "ndvi": str(ndvi_data.dtype),
                "cloud_mask": str(cloud_mask.dtype),
                "patch_ids": str(patch_ids.dtype)
            }
        }
    }
    
    # Save metadata
    import json
    metadata_path = output_dir / "synthetic_dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n✅ Synthetic data generation completed!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📊 Generated {N} samples of {H}x{W} images")
    print(f"💾 Files created:")
    print(f"   - {combined_path.name} (combined dataset)")
    print(f"   - synthetic_rgb.npy (RGB data)")
    print(f"   - synthetic_ndvi.npy (NDVI data)")
    print(f"   - synthetic_cloud_mask.npy (cloud mask)")
    print(f"   - synthetic_patch_ids.npy (patch IDs)")
    print(f"   - synthetic_dataset_metadata.json (metadata)")
    
    return output_dir

def main():
    """Main function to generate synthetic data."""
    try:
        output_dir = generate_synthetic_agricultural_data()
        print(f"\n🎯 You can now use these files to test the Agricultural ML Showcase tools!")
        print(f"   Example: python ../../multi_spectral_processing/data_validation.py --dataset {output_dir}/synthetic_agricultural_dataset.npz")
        
    except Exception as e:
        print(f"❌ Error generating synthetic data: {e}")

if __name__ == "__main__":
    main()
