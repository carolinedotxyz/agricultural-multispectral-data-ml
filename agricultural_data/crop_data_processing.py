#!/usr/bin/env python3
"""
Agricultural Crop Data Processing
Professional tools for processing and managing agricultural crop datasets.
This module demonstrates agricultural data engineering best practices.
"""

import numpy as np
import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

class AgriculturalDataProcessor:
    """
    Professional agricultural data processing class.
    Demonstrates best practices for handling crop classification datasets.
    """
    
    def __init__(self, output_dir: str = "./processed_agricultural_data"):
        """
        Initialize the agricultural data processor.
        
        Args:
            output_dir: Directory to save processed datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Standard crop categories (generalized)
        self.crop_categories = {
            "crop_1": "Primary agricultural crop",
            "crop_2": "Secondary agricultural crop", 
            "crop_3": "Tertiary agricultural crop",
            "crop_4": "Quaternary agricultural crop"
        }
        
        # Processing parameters
        self.default_subsample_size = 500
        self.min_samples_per_crop = 75
        self.validation_split = 0.1
        self.test_split = 0.2
    
    def create_balanced_dataset(self, 
                               crop_data: Dict[str, np.ndarray], 
                               target_size: int = None) -> Dict[str, Any]:
        """
        Create a balanced agricultural dataset with equal representation per crop.
        
        Args:
            crop_data: Dictionary mapping crop names to data arrays
            target_size: Target number of samples per crop (default: min available)
            
        Returns:
            Dictionary containing balanced dataset and metadata
        """
        print("🌾 Creating balanced agricultural dataset...")
        
        # Determine target size per crop
        if target_size is None:
            target_size = min(len(data) for data in crop_data.values())
        
        balanced_data = {}
        metadata = {
            "crop_distribution": {},
            "total_samples": 0,
            "processing_info": {
                "method": "balanced_sampling",
                "target_size_per_crop": target_size,
                "validation_split": self.validation_split,
                "test_split": self.test_split
            }
        }
        
        for crop_name, crop_array in crop_data.items():
            print(f"  Processing {crop_name}...")
            
            # Subsample to target size
            if len(crop_array) > target_size:
                indices = random.sample(range(len(crop_array)), target_size)
                balanced_data[crop_name] = crop_array[indices]
                print(f"    ✅ Subsampled {target_size} samples from {len(crop_array)} available")
            else:
                balanced_data[crop_name] = crop_array
                print(f"    ⚠️  Only {len(crop_array)} samples available, using all")
            
            metadata["crop_distribution"][crop_name] = len(balanced_data[crop_name])
            metadata["total_samples"] += len(balanced_data[crop_name])
        
        print(f"✅ Balanced dataset created with {metadata['total_samples']} total samples")
        return {"data": balanced_data, "metadata": metadata}
    
    def create_variable_sized_dataset(self, 
                                     crop_data: Dict[str, np.ndarray],
                                     min_samples: int = 75,
                                     max_samples: int = 500) -> Dict[str, Any]:
        """
        Create a variable-sized agricultural dataset with different sample counts per crop.
        
        Args:
            crop_data: Dictionary mapping crop names to data arrays
            min_samples: Minimum samples per crop
            max_samples: Maximum samples per crop
            
        Returns:
            Dictionary containing variable-sized dataset and metadata
        """
        print("🌾 Creating variable-sized agricultural dataset...")
        
        variable_data = {}
        metadata = {
            "crop_distribution": {},
            "total_samples": 0,
            "processing_info": {
                "method": "variable_sized_sampling",
                "min_samples_per_crop": min_samples,
                "max_samples_per_crop": max_samples,
                "validation_split": self.validation_split,
                "test_split": self.test_split
            }
        }
        
        for crop_name, crop_array in crop_data.items():
            print(f"  Processing {crop_name}...")
            
            # Determine target size for this crop
            available_samples = len(crop_array)
            target_size = min(max_samples, max(min_samples, available_samples))
            
            # Subsample to target size
            if available_samples > target_size:
                indices = random.sample(range(available_samples), target_size)
                variable_data[crop_name] = crop_array[indices]
                print(f"    ✅ Subsampled {target_size} samples from {available_samples} available")
            else:
                variable_data[crop_name] = crop_array
                print(f"    ⚠️  Only {available_samples} samples available, using all")
            
            metadata["crop_distribution"][crop_name] = len(variable_data[crop_name])
            metadata["total_samples"] += len(variable_data[crop_name])
        
        print(f"✅ Variable-sized dataset created with {metadata['total_samples']} total samples")
        return {"data": variable_data, "metadata": metadata}
    
    def create_data_splits(self, 
                           dataset: Dict[str, np.ndarray],
                           random_seed: int = 42) -> Dict[str, Any]:
        """
        Create train/validation/test splits for agricultural dataset.
        
        Args:
            dataset: Dictionary containing crop data
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing split indices and metadata
        """
        print("✂️  Creating data splits...")
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        split_indices = {
            "train": [],
            "validation": [],
            "test": []
        }
        
        split_metadata = {
            "split_ratios": {
                "train": 1 - self.validation_split - self.test_split,
                "validation": self.validation_split,
                "test": self.test_split
            },
            "crop_distribution": {},
            "random_seed": random_seed
        }
        
        current_index = 0
        
        for crop_name, crop_array in dataset.items():
            crop_size = len(crop_array)
            
            # Calculate split sizes
            test_size = int(crop_size * self.test_split)
            val_size = int(crop_size * self.validation_split)
            train_size = crop_size - test_size - val_size
            
            # Create indices for this crop
            crop_indices = list(range(current_index, current_index + crop_size))
            random.shuffle(crop_indices)
            
            # Assign to splits
            split_indices["test"].extend(crop_indices[:test_size])
            split_indices["validation"].extend(crop_indices[test_size:test_size + val_size])
            split_indices["train"].extend(crop_indices[test_size + val_size:])
            
            # Track distribution
            split_metadata["crop_distribution"][crop_name] = {
                "total": crop_size,
                "train": train_size,
                "validation": val_size,
                "test": test_size
            }
            
            current_index += crop_size
        
        # Convert to numpy arrays for efficiency
        for split_name in split_indices:
            split_indices[split_name] = np.array(split_indices[split_name], dtype=np.int64)
        
        print(f"✅ Data splits created:")
        print(f"  Train: {len(split_indices['train'])} samples")
        print(f"  Validation: {len(split_indices['validation'])} samples")
        print(f"  Test: {len(split_indices['test'])} samples")
        
        return {
            "split_indices": split_indices,
            "split_metadata": split_metadata
        }
    
    def save_processed_dataset(self, 
                              dataset: Dict[str, Any], 
                              filename: str) -> str:
        """
        Save processed agricultural dataset to disk.
        
        Args:
            dataset: Dataset to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            np.savez_compressed(
                output_path,
                **dataset["data"]
            )
            
            # Save metadata separately
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(dataset["metadata"], f, indent=2)
            
            print(f"💾 Dataset saved to: {output_path}")
            print(f"📄 Metadata saved to: {metadata_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Failed to save dataset: {e}")
            return None
    
    def load_and_validate_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Load and validate a saved agricultural dataset.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Dictionary containing loaded data and validation results
        """
        print(f"🔍 Loading and validating dataset: {file_path}")
        
        try:
            # Load data
            data = np.load(file_path, allow_pickle=False)
            
            # Load metadata if available
            metadata_path = Path(file_path).with_suffix('.json')
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Basic validation
            validation_results = {
                "file_loaded": True,
                "data_keys": list(data.keys()),
                "data_shapes": {key: data[key].shape for key in data.keys()},
                "metadata_loaded": bool(metadata),
                "validation_issues": []
            }
            
            # Check for common issues
            for key, array in data.items():
                if array.size == 0:
                    validation_results["validation_issues"].append(f"Empty array for key: {key}")
                
                if np.any(np.isnan(array)):
                    validation_results["validation_issues"].append(f"NaN values found in key: {key}")
            
            print(f"✅ Dataset loaded successfully")
            print(f"  Keys: {', '.join(validation_results['data_keys'])}")
            print(f"  Metadata: {'✅' if metadata else '❌'}")
            
            if validation_results["validation_issues"]:
                print(f"⚠️  Validation issues found:")
                for issue in validation_results["validation_issues"]:
                    print(f"    - {issue}")
            
            return {
                "data": data,
                "metadata": metadata,
                "validation": validation_results
            }
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return {
                "data": None,
                "metadata": None,
                "validation": {"file_loaded": False, "error": str(e)}
            }
    
    def generate_dataset_report(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generate a comprehensive report for the agricultural dataset.
        
        Args:
            dataset_info: Dataset information and metadata
            
        Returns:
            Formatted report string
        """
        if not dataset_info.get("metadata"):
            return "❌ No metadata available for report generation"
        
        metadata = dataset_info["metadata"]
        
        report_lines = []
        report_lines.append("🌾 AGRICULTURAL DATASET REPORT")
        report_lines.append("=" * 50)
        
        # Processing information
        if "processing_info" in metadata:
            proc_info = metadata["processing_info"]
            report_lines.append("📊 PROCESSING INFORMATION")
            report_lines.append("-" * 30)
            report_lines.append(f"Method: {proc_info.get('method', 'Unknown')}")
            if 'target_size_per_crop' in proc_info:
                report_lines.append(f"Target Size per Crop: {proc_info['target_size_per_crop']}")
            report_lines.append(f"Validation Split: {proc_info.get('validation_split', 0)*100:.1f}%")
            report_lines.append(f"Test Split: {proc_info.get('test_split', 0)*100:.1f}%")
            report_lines.append("")
        
        # Crop distribution
        if "crop_distribution" in metadata:
            report_lines.append("🌱 CROP DISTRIBUTION")
            report_lines.append("-" * 30)
            total_samples = 0
            
            for crop_name, crop_info in metadata["crop_distribution"].items():
                if isinstance(crop_info, dict):
                    # Split information available
                    crop_total = crop_info.get('total', crop_info)
                    report_lines.append(f"{crop_name}: {crop_total} samples")
                    total_samples += crop_total
                else:
                    # Simple count
                    report_lines.append(f"{crop_name}: {crop_info} samples")
                    total_samples += crop_info
            
            report_lines.append(f"Total Samples: {total_samples:,}")
            report_lines.append("")
        
        # Split information
        if "split_metadata" in metadata and "crop_distribution" in metadata["split_metadata"]:
            report_lines.append("✂️  SPLIT INFORMATION")
            report_lines.append("-" * 30)
            
            split_meta = metadata["split_metadata"]
            for crop_name, crop_splits in split_meta["crop_distribution"].items():
                report_lines.append(f"{crop_name}:")
                report_lines.append(f"  Train: {crop_splits['train']}")
                report_lines.append(f"  Validation: {crop_splits['validation']}")
                report_lines.append(f"  Test: {crop_splits['test']}")
        
        return "\n".join(report_lines)

def main():
    """Example usage of the AgriculturalDataProcessor."""
    print("🌾 Agricultural Data Processing Example")
    print("=" * 50)
    
    # Initialize processor
    processor = AgriculturalDataProcessor()
    
    # Example: Create synthetic crop data for demonstration
    print("\n📊 Creating synthetic crop data for demonstration...")
    
    # Generate synthetic data (in real usage, this would load from files)
    synthetic_crops = {
        "crop_1": np.random.rand(800, 64, 64, 3),  # RGB data
        "crop_2": np.random.rand(600, 64, 64, 3),
        "crop_3": np.random.rand(400, 64, 64, 3),
        "crop_4": np.random.rand(300, 64, 64, 3)
    }
    
    # Create balanced dataset
    print("\n1️⃣ Creating balanced dataset...")
    balanced_dataset = processor.create_balanced_dataset(synthetic_crops, target_size=300)
    
    # Create variable-sized dataset
    print("\n2️⃣ Creating variable-sized dataset...")
    variable_dataset = processor.create_variable_sized_dataset(synthetic_crops, min_samples=75, max_samples=500)
    
    # Create data splits
    print("\n3️⃣ Creating data splits...")
    splits = processor.create_data_splits(balanced_dataset["data"])
    
    # Save datasets
    print("\n4️⃣ Saving processed datasets...")
    processor.save_processed_dataset(balanced_dataset, "balanced_agricultural_dataset.npz")
    processor.save_processed_dataset(variable_dataset, "variable_agricultural_dataset.npz")
    
    # Load and validate
    print("\n5️⃣ Loading and validating saved dataset...")
    loaded_dataset = processor.load_and_validate_dataset(
        str(processor.output_dir / "balanced_agricultural_dataset.npz")
    )
    
    # Generate report
    print("\n6️⃣ Generating dataset report...")
    report = processor.generate_dataset_report(loaded_dataset)
    print(f"\n{report}")
    
    print("\n✅ Agricultural data processing example completed!")

if __name__ == "__main__":
    main()
