#!/usr/bin/env python3
"""
Multi-Spectral Data Validation
Professional validation tools for RGB and NDVI agricultural imagery datasets.
This module provides comprehensive data quality assurance for multi-spectral agricultural data.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configurable expectations for multi-spectral data
EXPECTED_RGB_SHAPE = (None, 3, None, None)   # N,3,H,W (N/H/W validated dynamically)
EXPECTED_NDVI_SHAPE = (None, None, None)     # N,H,W
EXPECTED_MASK_SHAPE = (None, None, None)     # N,H,W
EXPECTED_DTYPE_FLOAT = ("float32", "float64", "float16")
SENTINEL_VALUE = -1000.0  # Common sentinel value for cloud/no-data

def shape_str(a: np.ndarray) -> str:
    """Convert numpy array shape to readable string."""
    return f"{tuple(a.shape)}"

def within_01(arr: np.ndarray, tol: float = 1e-3) -> bool:
    """Check if array values are within [0,1] range with tolerance."""
    if arr.size == 0:
        return True
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return (mn >= -tol) and (mx <= 1.0 + tol)

def validate_rgb_data(rgb: np.ndarray) -> Dict[str, Any]:
    """Validate RGB data according to agricultural ML standards."""
    validation = {"valid": True, "issues": [], "warnings": [], "stats": {}}
    
    # Check shape and dimensions
    if rgb.ndim == 4 and rgb.shape[-1] == 3:
        validation["warnings"].append("RGB stored as (N,H,W,3); consider (N,3,H,W) for ML frameworks.")
        rgb = np.transpose(rgb, (0, 3, 1, 2))
    elif not (rgb.ndim == 4 and rgb.shape[1] == 3):
        validation["valid"] = False
        validation["issues"].append(f"Unexpected RGB shape {rgb.shape} (expected (N,3,H,W) or (N,H,W,3)).")
        return validation
    
    # Check data type
    if str(rgb.dtype) not in EXPECTED_DTYPE_FLOAT:
        validation["warnings"].append(f"RGB dtype is {rgb.dtype}; consider float32 for consistency.")
    
    # Sentinel value analysis
    sentinel_count = int(np.sum(rgb == SENTINEL_VALUE))
    validation["stats"]["sentinel_count"] = sentinel_count
    validation["stats"]["total_pixels"] = rgb.size
    
    # Range validation (after sentinel replacement)
    rgb_sanitized = np.where(rgb == SENTINEL_VALUE, 0.0, rgb)
    validation["stats"]["min_value"] = float(np.nanmin(rgb_sanitized)) if rgb_sanitized.size else None
    validation["stats"]["max_value"] = float(np.nanmax(rgb_sanitized)) if rgb_sanitized.size else None
    
    if not within_01(rgb_sanitized):
        validation["warnings"].append("RGB values not within [0,1] range (after sentinel replacement).")
    
    validation["stats"]["shape"] = shape_str(rgb)
    validation["stats"]["dtype"] = str(rgb.dtype)
    
    return validation

def validate_ndvi_data(ndvi: np.ndarray) -> Dict[str, Any]:
    """Validate NDVI data according to agricultural ML standards."""
    validation = {"valid": True, "issues": [], "warnings": [], "stats": {}}
    
    # Check shape and dimensions
    if not (ndvi.ndim == 3):
        validation["valid"] = False
        validation["issues"].append(f"NDVI must be 3D (N,H,W), got {ndvi.shape}")
        return validation
    
    # Check data type
    if str(ndvi.dtype) not in EXPECTED_DTYPE_FLOAT:
        validation["warnings"].append(f"NDVI dtype is {ndvi.dtype}; consider float32 for consistency.")
    
    # NDVI range validation (typically [-1, 1])
    ndvi_min, ndvi_max = np.nanmin(ndvi), np.nanmax(ndvi)
    validation["stats"]["min_value"] = float(ndvi_min)
    validation["stats"]["max_value"] = float(ndvi_max)
    
    if ndvi_min < -1.1 or ndvi_max > 1.1:
        validation["warnings"].append(f"NDVI values outside expected range [-1,1]: [{ndvi_min:.3f}, {ndvi_max:.3f}]")
    
    # Check for NaN values
    nan_count = int(np.sum(np.isnan(ndvi)))
    validation["stats"]["nan_count"] = nan_count
    
    if nan_count > 0:
        validation["warnings"].append(f"Found {nan_count} NaN values in NDVI data.")
    
    validation["stats"]["shape"] = shape_str(ndvi)
    validation["stats"]["dtype"] = str(ndvi.dtype)
    
    return validation

def validate_cloud_mask(cloud_mask: np.ndarray) -> Dict[str, Any]:
    """Validate cloud mask data according to agricultural ML standards."""
    validation = {"valid": True, "issues": [], "warnings": [], "stats": {}}
    
    # Check shape and dimensions
    if not (cloud_mask.ndim == 3):
        validation["valid"] = False
        validation["issues"].append(f"Cloud mask must be 3D (N,H,W), got {cloud_mask.shape}")
        return validation
    
    # Check data type
    if str(cloud_mask.dtype) not in ("uint8", "bool", "float32", "float64"):
        validation["warnings"].append(f"Cloud mask dtype {cloud_mask.dtype} may not be optimal for binary classification.")
    
    # Check for binary values (0 or 1)
    unique_values = np.unique(cloud_mask)
    validation["stats"]["unique_values"] = [float(v) for v in unique_values]
    
    if len(unique_values) > 2:
        validation["warnings"].append(f"Cloud mask has {len(unique_values)} unique values, expected binary (0,1).")
    
    # Calculate cloud coverage
    cloud_coverage = np.mean(cloud_mask)
    validation["stats"]["cloud_coverage"] = float(cloud_coverage)
    validation["stats"]["cloud_percentage"] = float(cloud_coverage * 100)
    
    if cloud_coverage > 0.8:
        validation["warnings"].append(f"High cloud coverage: {cloud_coverage*100:.1f}%")
    elif cloud_coverage < 0.1:
        validation["warnings"].append(f"Very low cloud coverage: {cloud_coverage*100:.1f}%")
    
    validation["stats"]["shape"] = shape_str(cloud_mask)
    validation["stats"]["dtype"] = str(cloud_mask.dtype)
    
    return validation

def validate_multi_spectral_dataset(npz_path: str) -> Dict[str, Any]:
    """
    Comprehensive validation of multi-spectral agricultural dataset.
    
    Args:
        npz_path: Path to .npz file containing multi-spectral data
        
    Returns:
        Dictionary containing validation results, issues, and statistics
    """
    validation_result = {
        "file_path": npz_path,
        "valid": True,
        "overall_score": 0.0,
        "issues": [],
        "warnings": [],
        "component_validation": {},
        "summary": {}
    }
    
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        validation_result["valid"] = False
        validation_result["issues"].append(f"Failed to load dataset: {e}")
        return validation_result
    
    # Check available keys
    available_keys = set(data.files)
    validation_result["summary"]["available_keys"] = sorted(list(available_keys))
    
    # Required keys for multi-spectral agricultural data
    required_keys = {"rgb", "ndvi", "cloud_mask"}
    missing_keys = [k for k in required_keys if k not in available_keys]
    
    if missing_keys:
        validation_result["valid"] = False
        validation_result["issues"].append(f"Missing required keys: {missing_keys}")
    
    # Validate each component
    if "rgb" in available_keys:
        rgb_validation = validate_rgb_data(data["rgb"])
        validation_result["component_validation"]["rgb"] = rgb_validation
        if not rgb_validation["valid"]:
            validation_result["valid"] = False
            validation_result["issues"].extend(rgb_validation["issues"])
        validation_result["warnings"].extend(rgb_validation["warnings"])
    
    if "ndvi" in available_keys:
        ndvi_validation = validate_ndvi_data(data["ndvi"])
        validation_result["component_validation"]["ndvi"] = ndvi_validation
        if not ndvi_validation["valid"]:
            validation_result["valid"] = False
            validation_result["issues"].extend(ndvi_validation["issues"])
        validation_result["warnings"].extend(ndvi_validation["warnings"])
    
    if "cloud_mask" in available_keys:
        cloud_validation = validate_cloud_mask(data["cloud_mask"])
        validation_result["component_validation"]["cloud_mask"] = cloud_validation
        if not cloud_validation["valid"]:
            validation_result["valid"] = False
            validation_result["issues"].extend(cloud_validation["issues"])
        validation_result["warnings"].extend(cloud_validation["warnings"])
    
    # Calculate overall quality score
    total_issues = len(validation_result["issues"])
    total_warnings = len(validation_result["warnings"])
    
    # Score calculation: 10 points base, -2 per issue, -0.5 per warning
    validation_result["overall_score"] = max(0.0, 10.0 - (total_issues * 2.0) - (total_warnings * 0.5))
    
    # Generate summary
    validation_result["summary"]["total_issues"] = total_issues
    validation_result["summary"]["total_warnings"] = total_warnings
    validation_result["summary"]["quality_score"] = validation_result["overall_score"]
    
    return validation_result

def main():
    """Command-line interface for dataset validation."""
    parser = argparse.ArgumentParser(description="Validate multi-spectral agricultural dataset")
    parser.add_argument("--dataset", required=True, help="Path to .npz dataset file")
    parser.add_argument("--output", help="Output JSON file for validation results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        return
    
    print(f"🔍 Validating multi-spectral dataset: {args.dataset}")
    validation_results = validate_multi_spectral_dataset(args.dataset)
    
    # Print results
    print(f"\n📊 Validation Results:")
    print(f"  Overall Valid: {'✅' if validation_results['valid'] else '❌'}")
    print(f"  Quality Score: {validation_results['overall_score']:.1f}/10")
    print(f"  Issues: {len(validation_results['issues'])}")
    print(f"  Warnings: {len(validation_results['warnings'])}")
    
    if validation_results['issues']:
        print(f"\n❌ Issues Found:")
        for issue in validation_results['issues']:
            print(f"  - {issue}")
    
    if validation_results['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
