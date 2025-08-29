#!/usr/bin/env python3
"""
Agricultural Data Quality Assurance
Professional tools for ensuring data quality in agricultural machine learning datasets.
This module demonstrates comprehensive data validation and quality control practices.
"""

import numpy as np
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

class AgriculturalDataQualityAssurance:
    """
    Professional agricultural data quality assurance system.
    Demonstrates industry-standard data validation and quality control.
    """
    
    def __init__(self, output_dir: str = "./quality_assurance_reports"):
        """
        Initialize the quality assurance system.
        
        Args:
            output_dir: Directory to save quality reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_image_size": (32, 32),
            "max_image_size": (2048, 2048),
            "min_samples_per_class": 10,
            "max_missing_data_ratio": 0.1,
            "max_corruption_ratio": 0.05,
            "min_data_integrity_score": 7.0
        }
        
        # Data type expectations
        self.expected_dtypes = {
            "rgb": ["float32", "float64", "uint8"],
            "ndvi": ["float32", "float64"],
            "cloud_mask": ["uint8", "bool", "float32"],
            "patch_ids": ["int64", "uint64", "object"]
        }
    
    def validate_dataset_structure(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate the overall structure of an agricultural dataset.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Dictionary containing structure validation results
        """
        print(f"🔍 Validating dataset structure: {dataset_path}")
        
        validation_result = {
            "dataset_path": dataset_path,
            "timestamp": datetime.now().isoformat(),
            "structure_valid": False,
            "issues": [],
            "warnings": [],
            "structure_score": 0.0,
            "details": {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(dataset_path):
                validation_result["issues"].append("Dataset file not found")
                return validation_result
            
            # Check file size
            file_size = os.path.getsize(dataset_path)
            validation_result["details"]["file_size_bytes"] = file_size
            validation_result["details"]["file_size_mb"] = file_size / (1024 * 1024)
            
            if file_size == 0:
                validation_result["issues"].append("Dataset file is empty")
                return validation_result
            
            # Load dataset
            data = np.load(dataset_path, allow_pickle=False)
            available_keys = list(data.keys())
            validation_result["details"]["available_keys"] = available_keys
            
            # Check required keys
            required_keys = ["rgb", "ndvi", "cloud_mask"]
            missing_keys = [key for key in required_keys if key not in available_keys]
            
            if missing_keys:
                validation_result["issues"].append(f"Missing required keys: {missing_keys}")
            else:
                validation_result["structure_valid"] = True
            
            # Check data types and shapes
            for key in available_keys:
                if key in data:
                    array = data[key]
                    key_validation = self._validate_array_properties(key, array)
                    validation_result["details"][key] = key_validation
                    
                    if not key_validation["valid"]:
                        validation_result["issues"].extend(key_validation["issues"])
                    if key_validation["warnings"]:
                        validation_result["warnings"].extend(key_validation["warnings"])
            
            # Calculate structure score
            validation_result["structure_score"] = self._calculate_structure_score(validation_result)
            
            data.close()
            
        except Exception as e:
            validation_result["issues"].append(f"Failed to load dataset: {e}")
        
        return validation_result
    
    def _validate_array_properties(self, key: str, array: np.ndarray) -> Dict[str, Any]:
        """
        Validate properties of a specific array in the dataset.
        
        Args:
            key: Key name for the array
            array: NumPy array to validate
            
        Returns:
            Dictionary containing validation results for this array
        """
        validation = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "properties": {}
        }
        
        # Basic properties
        validation["properties"]["shape"] = array.shape
        validation["properties"]["dtype"] = str(array.dtype)
        validation["properties"]["size"] = array.size
        validation["properties"]["ndim"] = array.ndim
        
        # Check data type expectations
        if key in self.expected_dtypes:
            expected_types = self.expected_dtypes[key]
            if str(array.dtype) not in expected_types:
                validation["warnings"].append(f"Unexpected dtype {array.dtype} for {key}, expected one of {expected_types}")
        
        # Check for empty arrays
        if array.size == 0:
            validation["valid"] = False
            validation["issues"].append(f"Array {key} is empty")
        
        # Check for NaN values
        if np.any(np.isnan(array)):
            nan_count = int(np.sum(np.isnan(array)))
            nan_ratio = nan_count / array.size
            validation["warnings"].append(f"Found {nan_count} NaN values ({nan_ratio:.2%}) in {key}")
            
            if nan_ratio > self.quality_thresholds["max_missing_data_ratio"]:
                validation["valid"] = False
                validation["issues"].append(f"Too many NaN values in {key}: {nan_ratio:.2%}")
        
        # Check for infinite values
        if np.any(np.isinf(array)):
            inf_count = int(np.sum(np.isinf(array)))
            validation["warnings"].append(f"Found {inf_count} infinite values in {key}")
        
        # Check image dimensions for spatial data
        if key in ["rgb", "ndvi", "cloud_mask"] and array.ndim >= 2:
            if array.ndim == 3:  # (N, H, W)
                H, W = array.shape[1], array.shape[2]
            elif array.ndim == 4:  # (N, C, H, W)
                H, W = array.shape[2], array.shape[3]
            else:
                H, W = array.shape[-2], array.shape[-1]
            
            min_H, min_W = self.quality_thresholds["min_image_size"]
            max_H, max_W = self.quality_thresholds["max_image_size"]
            
            if H < min_H or W < min_W:
                validation["warnings"].append(f"Image dimensions {H}x{W} are below minimum {min_H}x{min_W}")
            elif H > max_H or W > max_W:
                validation["warnings"].append(f"Image dimensions {H}x{W} are above maximum {max_H}x{max_W}")
        
        return validation
    
    def _calculate_structure_score(self, validation_result: Dict[str, Any]) -> float:
        """
        Calculate a quality score based on validation results.
        
        Args:
            validation_result: Validation results dictionary
            
        Returns:
            Quality score from 0.0 to 10.0
        """
        base_score = 10.0
        
        # Deduct points for issues
        issue_penalty = len(validation_result["issues"]) * 2.0
        warning_penalty = len(validation_result["warnings"]) * 0.5
        
        # Deduct points for missing keys
        if "details" in validation_result and "available_keys" in validation_result["details"]:
            required_keys = ["rgb", "ndvi", "cloud_mask"]
            available_keys = validation_result["details"]["available_keys"]
            missing_keys = len([key for key in required_keys if key not in available_keys])
            missing_penalty = missing_keys * 1.5
        else:
            missing_penalty = 0
        
        final_score = max(0.0, base_score - issue_penalty - warning_penalty - missing_penalty)
        return round(final_score, 1)
    
    def validate_data_integrity(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate the integrity of agricultural dataset contents.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Dictionary containing integrity validation results
        """
        print(f"🔒 Validating data integrity: {dataset_path}")
        
        integrity_result = {
            "dataset_path": dataset_path,
            "timestamp": datetime.now().isoformat(),
            "integrity_valid": False,
            "issues": [],
            "warnings": [],
            "integrity_score": 0.0,
            "details": {}
        }
        
        try:
            data = np.load(dataset_path, allow_pickle=False)
            
            # Check data consistency across modalities
            if "rgb" in data and "ndvi" in data:
                rgb_shape = data["rgb"].shape
                ndvi_shape = data["ndvi"].shape
                
                # Check if RGB and NDVI have compatible dimensions
                if rgb_shape[0] != ndvi_shape[0]:
                    integrity_result["issues"].append(f"Sample count mismatch: RGB has {rgb_shape[0]}, NDVI has {ndvi_shape[0]}")
                
                if rgb_shape[1:] != ndvi_shape[1:]:
                    integrity_result["warnings"].append(f"Dimension mismatch: RGB {rgb_shape[1:]}, NDVI {ndvi_shape[1:]}")
            
            # Check cloud mask consistency
            if "cloud_mask" in data:
                cloud_shape = data["cloud_mask"].shape
                if "rgb" in data:
                    if cloud_shape[0] != data["rgb"].shape[0]:
                        integrity_result["issues"].append("Cloud mask sample count doesn't match RGB")
                
                # Validate cloud mask values
                cloud_mask = data["cloud_mask"]
                unique_values = np.unique(cloud_mask)
                if len(unique_values) > 2:
                    integrity_result["warnings"].append(f"Cloud mask has {len(unique_values)} unique values, expected binary")
                
                # Check cloud coverage statistics
                cloud_coverage = np.mean(cloud_mask)
                if cloud_coverage > 0.9:
                    integrity_result["warnings"].append(f"Very high cloud coverage: {cloud_coverage:.1%}")
                elif cloud_coverage < 0.01:
                    integrity_result["warnings"].append(f"Very low cloud coverage: {cloud_coverage:.1%}")
            
            # Check for data corruption indicators
            corruption_indicators = self._check_corruption_indicators(data)
            if corruption_indicators["issues"]:
                integrity_result["issues"].extend(corruption_indicators["issues"])
            if corruption_indicators["warnings"]:
                integrity_result["warnings"].extend(corruption_indicators["warnings"])
            
            # Calculate integrity score
            integrity_result["integrity_score"] = self._calculate_integrity_score(integrity_result)
            integrity_result["integrity_valid"] = integrity_result["integrity_score"] >= self.quality_thresholds["min_data_integrity_score"]
            
            data.close()
            
        except Exception as e:
            integrity_result["issues"].append(f"Failed to validate data integrity: {e}")
        
        return integrity_result
    
    def _check_corruption_indicators(self, data: np.load) -> Dict[str, Any]:
        """
        Check for indicators of data corruption.
        
        Args:
            data: Loaded dataset
            
        Returns:
            Dictionary containing corruption check results
        """
        corruption_result = {
            "issues": [],
            "warnings": []
        }
        
        for key in data.keys():
            array = data[key]
            
            # Check for extreme outliers
            if array.dtype in [np.float32, np.float64]:
                # Calculate z-scores for outlier detection
                mean_val = np.mean(array)
                std_val = np.std(array)
                
                if std_val > 0:
                    z_scores = np.abs((array - mean_val) / std_val)
                    extreme_outliers = np.sum(z_scores > 5)  # 5 standard deviations
                    
                    if extreme_outliers > 0:
                        outlier_ratio = extreme_outliers / array.size
                        if outlier_ratio > self.quality_thresholds["max_corruption_ratio"]:
                            corruption_result["issues"].append(f"Too many extreme outliers in {key}: {outlier_ratio:.2%}")
                        else:
                            corruption_result["warnings"].append(f"Found {extreme_outliers} extreme outliers in {key}")
            
            # Check for uniform arrays (potential corruption)
            if array.size > 1000:  # Only check large arrays
                unique_ratio = len(np.unique(array)) / array.size
                if unique_ratio < 0.01:  # Less than 1% unique values
                    corruption_result["warnings"].append(f"Very low diversity in {key}: {unique_ratio:.2%} unique values")
        
        return corruption_result
    
    def _calculate_integrity_score(self, integrity_result: Dict[str, Any]) -> float:
        """
        Calculate an integrity score based on validation results.
        
        Args:
            integrity_result: Integrity validation results
            
        Returns:
            Integrity score from 0.0 to 10.0
        """
        base_score = 10.0
        
        # Deduct points for issues
        issue_penalty = len(integrity_result["issues"]) * 2.0
        warning_penalty = len(integrity_result["warnings"]) * 0.5
        
        final_score = max(0.0, base_score - issue_penalty - warning_penalty)
        return round(final_score, 1)
    
    def generate_quality_report(self, 
                               structure_validation: Dict[str, Any],
                               integrity_validation: Dict[str, Any]) -> str:
        """
        Generate a comprehensive quality assurance report.
        
        Args:
            structure_validation: Structure validation results
            integrity_validation: Integrity validation results
            
        Returns:
            Formatted quality report string
        """
        report_lines = []
        report_lines.append("🔍 AGRICULTURAL DATA QUALITY ASSURANCE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Dataset: {structure_validation.get('dataset_path', 'Unknown')}")
        report_lines.append(f"Timestamp: {structure_validation.get('timestamp', 'Unknown')}")
        report_lines.append("")
        
        # Overall Quality Summary
        structure_score = structure_validation.get('structure_score', 0.0)
        integrity_score = integrity_validation.get('integrity_score', 0.0)
        overall_score = (structure_score + integrity_score) / 2
        
        report_lines.append("📊 OVERALL QUALITY SUMMARY")
        report_lines.append("-" * 30)
        report_lines.append(f"Structure Score: {structure_score}/10")
        report_lines.append(f"Integrity Score: {integrity_score}/10")
        report_lines.append(f"Overall Score: {overall_score:.1f}/10")
        report_lines.append(f"Quality Status: {'✅ PASS' if overall_score >= 7.0 else '❌ FAIL'}")
        report_lines.append("")
        
        # Structure Validation Details
        report_lines.append("🏗️  STRUCTURE VALIDATION")
        report_lines.append("-" * 30)
        report_lines.append(f"Status: {'✅ Valid' if structure_validation.get('structure_valid', False) else '❌ Invalid'}")
        
        if structure_validation.get('issues'):
            report_lines.append(f"Issues: {len(structure_validation['issues'])}")
            for issue in structure_validation['issues']:
                report_lines.append(f"  ❌ {issue}")
        
        if structure_validation.get('warnings'):
            report_lines.append(f"Warnings: {len(structure_validation['warnings'])}")
            for warning in structure_validation['warnings'][:5]:  # Show first 5
                report_lines.append(f"  ⚠️  {warning}")
        
        # Integrity Validation Details
        report_lines.append("\n🔒 INTEGRITY VALIDATION")
        report_lines.append("-" * 30)
        report_lines.append(f"Status: {'✅ Valid' if integrity_validation.get('integrity_valid', False) else '❌ Invalid'}")
        
        if integrity_validation.get('issues'):
            report_lines.append(f"Issues: {len(integrity_validation['issues'])}")
            for issue in integrity_validation['issues']:
                report_lines.append(f"  ❌ {issue}")
        
        if integrity_validation.get('warnings'):
            report_lines.append(f"Warnings: {len(integrity_validation['warnings'])}")
            for warning in integrity_validation['warnings'][:5]:  # Show first 5
                report_lines.append(f"  ⚠️  {warning}")
        
        # Recommendations
        report_lines.append("\n💡 RECOMMENDATIONS")
        report_lines.append("-" * 30)
        
        if overall_score >= 9.0:
            report_lines.append("✅ Dataset quality is excellent. Ready for ML training.")
        elif overall_score >= 7.0:
            report_lines.append("✅ Dataset quality is acceptable. Minor issues should be addressed.")
        elif overall_score >= 5.0:
            report_lines.append("⚠️  Dataset quality is moderate. Several issues need attention.")
        else:
            report_lines.append("❌ Dataset quality is poor. Major issues must be resolved before use.")
        
        if structure_validation.get('issues'):
            report_lines.append("  - Fix structural issues before proceeding")
        if integrity_validation.get('issues'):
            report_lines.append("  - Address data integrity concerns")
        
        return "\n".join(report_lines)
    
    def save_quality_report(self, 
                           structure_validation: Dict[str, Any],
                           integrity_validation: Dict[str, Any],
                           filename: str = None) -> str:
        """
        Save quality assurance report to disk.
        
        Args:
            structure_validation: Structure validation results
            integrity_validation: Integrity validation results
            filename: Output filename (optional)
            
        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_assurance_report_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Generate text report
        text_report = self.generate_quality_report(structure_validation, integrity_validation)
        
        # Save text report
        text_path = output_path.with_suffix('.txt')
        with open(text_path, 'w') as f:
            f.write(text_report)
        
        # Save detailed JSON report
        detailed_report = {
            "structure_validation": structure_validation,
            "integrity_validation": integrity_validation,
            "text_report": text_report,
            "summary": {
                "overall_score": (structure_validation.get('structure_score', 0) + 
                                integrity_validation.get('integrity_score', 0)) / 2,
                "structure_score": structure_validation.get('structure_score', 0),
                "integrity_score": integrity_validation.get('integrity_score', 0),
                "total_issues": (len(structure_validation.get('issues', [])) + 
                               len(integrity_validation.get('issues', []))),
                "total_warnings": (len(structure_validation.get('warnings', [])) + 
                                 len(integrity_validation.get('warnings', [])))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        print(f"💾 Quality report saved to: {text_path}")
        print(f"📄 Detailed report saved to: {output_path}")
        
        return str(output_path)

def main():
    """Example usage of the AgriculturalDataQualityAssurance system."""
    print("🔍 Agricultural Data Quality Assurance Example")
    print("=" * 60)
    
    # Initialize quality assurance system
    qa_system = AgriculturalDataQualityAssurance()
    
    # Example: Create synthetic dataset for validation
    print("\n📊 Creating synthetic dataset for quality validation...")
    
    # Generate synthetic data
    N, H, W = 100, 64, 64
    
    # Create synthetic dataset
    synthetic_data = {
        "rgb": np.random.rand(N, 3, H, W).astype(np.float32),
        "ndvi": np.random.rand(N, H, W).astype(np.float32) * 2 - 1,
        "cloud_mask": np.random.randint(0, 2, (N, H, W), dtype=np.uint8),
        "patch_ids": np.arange(N, dtype=np.int64)
    }
    
    # Save synthetic dataset
    dataset_path = "./synthetic_agricultural_dataset.npz"
    np.savez_compressed(dataset_path, **synthetic_data)
    
    print(f"✅ Created synthetic dataset: {dataset_path}")
    
    # Run quality assurance
    print("\n1️⃣ Validating dataset structure...")
    structure_validation = qa_system.validate_dataset_structure(dataset_path)
    
    print("\n2️⃣ Validating data integrity...")
    integrity_validation = qa_system.validate_data_integrity(dataset_path)
    
    print("\n3️⃣ Generating quality report...")
    quality_report = qa_system.generate_quality_report(structure_validation, integrity_validation)
    
    # Save report
    print("\n4️⃣ Saving quality report...")
    qa_system.save_quality_report(structure_validation, integrity_validation)
    
    # Display summary
    print(f"\n📊 Quality Summary:")
    print(f"  Structure Score: {structure_validation.get('structure_score', 0)}/10")
    print(f"  Integrity Score: {integrity_validation.get('integrity_score', 0)}/10")
    print(f"  Overall Score: {(structure_validation.get('structure_score', 0) + integrity_validation.get('integrity_score', 0)) / 2:.1f}/10")
    
    # Clean up
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    
    print("\n✅ Agricultural data quality assurance example completed!")

if __name__ == "__main__":
    main()
