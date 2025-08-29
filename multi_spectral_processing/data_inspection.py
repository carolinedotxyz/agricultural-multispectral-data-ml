#!/usr/bin/env python3
"""
Agricultural Data Inspection Tools
Professional utilities for exploring and analyzing multi-spectral agricultural datasets.
This module provides comprehensive data inspection capabilities for agricultural ML workflows.
"""

import numpy as np
import os
import sys
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

def inspect_npz_file(file_path: str, max_print: int = 5) -> Dict[str, Any]:
    """
    Comprehensive inspection of an .npz dataset file.
    
    Args:
        file_path: Path to the .npz file to inspect
        max_print: Maximum number of sample values to display
        
    Returns:
        Dictionary containing inspection results and metadata
    """
    inspection_results = {
        "file_path": file_path,
        "file_exists": False,
        "loadable": False,
        "keys": [],
        "data_summary": {},
        "errors": []
    }
    
    # Check if file exists
    if not os.path.exists(file_path):
        inspection_results["errors"].append(f"File not found: {file_path}")
        return inspection_results
    
    inspection_results["file_exists"] = True
    
    try:
        data = np.load(file_path, allow_pickle=True)
        inspection_results["loadable"] = True
        inspection_results["keys"] = list(data.keys())
        
        # Inspect each key
        for key in data.keys():
            arr = data[key]
            key_info = {
                "type": str(type(arr)),
                "shape": None,
                "dtype": None,
                "sample_values": [],
                "statistics": {}
            }
            
            if isinstance(arr, np.ndarray):
                key_info["shape"] = arr.shape
                key_info["dtype"] = str(arr.dtype)
                
                # Get sample values
                flat = arr.ravel()
                if flat.size > 0:
                    sample_indices = min(max_print, flat.size)
                    key_info["sample_values"] = flat[:sample_indices].tolist()
                    
                    # Basic statistics
                    if arr.size > 0:
                        key_info["statistics"] = {
                            "min": float(np.nanmin(arr)),
                            "max": float(np.nanmax(arr)),
                            "mean": float(np.nanmean(arr)),
                            "std": float(np.nanstd(arr)),
                            "nan_count": int(np.sum(np.isnan(arr))),
                            "size": int(arr.size)
                        }
                
                # Special handling for multi-dimensional arrays
                if arr.ndim == 4:
                    key_info["example_slice_shape"] = arr[0].shape
                elif arr.ndim == 3:
                    key_info["example_slice_shape"] = arr[0].shape
            else:
                key_info["value"] = str(arr)
            
            inspection_results["data_summary"][key] = key_info
        
        data.close()
        
    except Exception as e:
        inspection_results["errors"].append(f"Failed to load file: {e}")
    
    return inspection_results

def summarize_npz_file(file_path: str) -> str:
    """
    Generate a concise summary of an .npz file.
    
    Args:
        file_path: Path to the .npz file
        
    Returns:
        Formatted summary string
    """
    try:
        data = np.load(file_path, allow_pickle=False)
        
        def shape_of(key):
            return data[key].shape if key in data else None
        
        summary = f"{file_path}: "
        summary += f"rgb={shape_of('rgb')}, "
        summary += f"ndvi={shape_of('ndvi')}, "
        summary += f"mask={shape_of('cloud_mask')}, "
        summary += f"ids={shape_of('patch_ids')}"
        
        data.close()
        return summary
        
    except Exception as e:
        return f"{file_path}: Error loading file - {e}"

def batch_inspect_npz_files(directory_path: str, pattern: str = "*.npz") -> Dict[str, Any]:
    """
    Inspect multiple .npz files in a directory.
    
    Args:
        directory_path: Directory containing .npz files
        pattern: File pattern to match (default: "*.npz")
        
    Returns:
        Dictionary containing inspection results for all files
    """
    if not os.path.exists(directory_path):
        return {"error": f"Directory not found: {directory_path}"}
    
    # Find all matching files
    search_pattern = os.path.join(directory_path, pattern)
    npz_files = glob.glob(search_pattern)
    
    if not npz_files:
        return {"error": f"No .npz files found matching pattern: {pattern}"}
    
    batch_results = {
        "directory": directory_path,
        "pattern": pattern,
        "total_files": len(npz_files),
        "file_results": {},
        "summary_statistics": {}
    }
    
    # Inspect each file
    for file_path in npz_files:
        file_name = os.path.basename(file_path)
        batch_results["file_results"][file_name] = inspect_npz_file(file_path)
    
    # Generate summary statistics
    all_keys = set()
    total_size = 0
    valid_files = 0
    
    for file_result in batch_results["file_results"].values():
        if file_result["loadable"]:
            valid_files += 1
            all_keys.update(file_result["keys"])
            
            # Calculate total data size
            for key, key_info in file_result["data_summary"].items():
                if "statistics" in key_info and "size" in key_info["statistics"]:
                    total_size += key_info["statistics"]["size"]
    
    batch_results["summary_statistics"] = {
        "valid_files": valid_files,
        "total_data_elements": total_size,
        "common_keys": sorted(list(all_keys)),
        "success_rate": valid_files / len(npz_files) if npz_files else 0
    }
    
    return batch_results

def generate_inspection_report(inspection_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Generate a formatted inspection report.
    
    Args:
        inspection_results: Results from file inspection
        output_file: Optional file path to save the report
        
    Returns:
        Formatted report string
    """
    if "error" in inspection_results:
        return f"❌ Inspection failed: {inspection_results['error']}"
    
    report_lines = []
    report_lines.append("🔍 AGRICULTURAL DATA INSPECTION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Directory: {inspection_results.get('directory', 'N/A')}")
    report_lines.append(f"Pattern: {inspection_results.get('pattern', 'N/A')}")
    report_lines.append(f"Total Files: {inspection_results.get('total_files', 0)}")
    report_lines.append("")
    
    # Summary statistics
    if "summary_statistics" in inspection_results:
        stats = inspection_results["summary_statistics"]
        report_lines.append("📊 SUMMARY STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Valid Files: {stats.get('valid_files', 0)}")
        report_lines.append(f"Success Rate: {stats.get('success_rate', 0)*100:.1f}%")
        report_lines.append(f"Total Data Elements: {stats.get('total_data_elements', 0):,}")
        report_lines.append(f"Common Keys: {', '.join(stats.get('common_keys', []))}")
        report_lines.append("")
    
    # Individual file results
    if "file_results" in inspection_results:
        report_lines.append("📁 INDIVIDUAL FILE RESULTS")
        report_lines.append("-" * 30)
        
        for file_name, file_result in inspection_results["file_results"].items():
            report_lines.append(f"\nFile: {file_name}")
            
            if file_result["loadable"]:
                report_lines.append(f"  Status: ✅ Loadable")
                report_lines.append(f"  Keys: {', '.join(file_result['keys'])}")
                
                # Data summary for each key
                for key, key_info in file_result["data_summary"].items():
                    report_lines.append(f"  {key}:")
                    if "shape" in key_info and key_info["shape"]:
                        report_lines.append(f"    Shape: {key_info['shape']}")
                    if "dtype" in key_info and key_info["dtype"]:
                        report_lines.append(f"    Type: {key_info['dtype']}")
                    if "statistics" in key_info and key_info["statistics"]:
                        stats = key_info["statistics"]
                        report_lines.append(f"    Size: {stats.get('size', 'N/A'):,}")
                        if "nan_count" in stats and stats["nan_count"] > 0:
                            report_lines.append(f"    NaN Count: {stats['nan_count']}")
            else:
                report_lines.append(f"  Status: ❌ Not loadable")
                if file_result["errors"]:
                    report_lines.append(f"  Errors: {'; '.join(file_result['errors'])}")
    
    report = "\n".join(report_lines)
    
    # Save report if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
    
    return report

def main():
    """Command-line interface for data inspection."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python data_inspection.py <path_to_file.npz>")
        print("  python data_inspection.py --batch <directory_path>")
        print("  python data_inspection.py --batch <directory_path> --pattern '*.npz'")
        return
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: Directory path required for batch mode")
            return
        
        directory_path = sys.argv[2]
        pattern = sys.argv[3] if len(sys.argv) > 3 else "*.npz"
        
        print(f"🔍 Batch inspecting .npz files in: {directory_path}")
        print(f"Pattern: {pattern}")
        
        batch_results = batch_inspect_npz_files(directory_path, pattern)
        
        if "error" in batch_results:
            print(f"❌ {batch_results['error']}")
            return
        
        print(f"\n📊 Found {batch_results['total_files']} files")
        print(f"✅ Valid files: {batch_results['summary_statistics']['valid_files']}")
        print(f"📈 Success rate: {batch_results['summary_statistics']['success_rate']*100:.1f}%")
        
        # Generate and display report
        report = generate_inspection_report(batch_results)
        print(f"\n{report}")
        
    else:
        # Single file inspection
        file_path = sys.argv[1]
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"🔍 Inspecting file: {file_path}")
        inspection_results = inspect_npz_file(file_path)
        
        if not inspection_results["loadable"]:
            print(f"❌ Failed to load file: {inspection_results['errors']}")
            return
        
        print(f"✅ File loaded successfully")
        print(f"📁 Keys: {', '.join(inspection_results['keys'])}")
        
        # Display data summary
        for key, key_info in inspection_results["data_summary"].items():
            print(f"\n--- Key: {key} ---")
            print(f"  Type: {key_info['type']}")
            
            if key_info["shape"]:
                print(f"  Shape: {key_info['shape']}")
                print(f"  Dtype: {key_info['dtype']}")
                
                if key_info["sample_values"]:
                    print(f"  Sample values: {key_info['sample_values']}")
                
                if key_info["statistics"]:
                    stats = key_info["statistics"]
                    print(f"  Size: {stats['size']:,}")
                    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                    print(f"  Mean: {stats['mean']:.3f}")
                    if stats['nan_count'] > 0:
                        print(f"  NaN count: {stats['nan_count']}")
        
        print("\n✅ Inspection complete.")

if __name__ == "__main__":
    main()
