#!/usr/bin/env python3
"""
Cloud Coverage Analysis for Agricultural Imagery
Professional tools for analyzing cloud coverage in multi-spectral agricultural datasets.
This module provides comprehensive cloud analysis for agricultural remote sensing data.
"""

import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

def analyze_cloud_coverage(npz_path: str) -> Dict[str, Any]:
    """
    Analyze cloud coverage in agricultural imagery dataset.
    
    Args:
        npz_path: Path to .npz file containing cloud mask data
        
    Returns:
        Dictionary containing cloud coverage statistics and analysis
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return {
            "error": f"Failed to load dataset: {e}",
            "valid": False
        }
    
    # Check if cloud mask exists
    if "cloud_mask" not in data.files:
        return {
            "error": "No cloud_mask found in dataset",
            "valid": False
        }
    
    cloud_mask = data["cloud_mask"]
    
    # Basic statistics
    total_pixels = cloud_mask.size
    cloud_pixels = np.sum(cloud_mask)
    clear_pixels = total_pixels - cloud_pixels
    
    # Calculate coverage percentages
    cloud_coverage = cloud_pixels / total_pixels
    clear_coverage = clear_pixels / total_pixels
    
    # Calculate per-image statistics if 3D
    if cloud_mask.ndim == 3:
        N, H, W = cloud_mask.shape
        per_image_coverage = np.mean(cloud_mask, axis=(1, 2))
        
        coverage_stats = {
            "min_coverage": float(np.min(per_image_coverage)),
            "max_coverage": float(np.max(per_image_coverage)),
            "mean_coverage": float(np.mean(per_image_coverage)),
            "median_coverage": float(np.median(per_image_coverage)),
            "std_coverage": float(np.std(per_image_coverage))
        }
    else:
        coverage_stats = {}
    
    # Quality assessment
    quality_assessment = assess_cloud_quality(cloud_coverage)
    
    # Generate analysis results
    analysis_results = {
        "file_path": npz_path,
        "valid": True,
        "total_pixels": int(total_pixels),
        "cloud_pixels": int(cloud_pixels),
        "clear_pixels": int(clear_pixels),
        "cloud_coverage": float(cloud_coverage),
        "cloud_percentage": float(cloud_coverage * 100),
        "clear_coverage": float(clear_coverage),
        "clear_percentage": float(clear_coverage * 100),
        "quality_assessment": quality_assessment,
        "coverage_statistics": coverage_stats,
        "data_shape": cloud_mask.shape,
        "data_type": str(cloud_mask.dtype)
    }
    
    return analysis_results

def assess_cloud_quality(cloud_coverage: float) -> Dict[str, Any]:
    """
    Assess the quality of agricultural imagery based on cloud coverage.
    
    Args:
        cloud_coverage: Cloud coverage ratio (0.0 to 1.0)
        
    Returns:
        Dictionary containing quality assessment and recommendations
    """
    coverage_percentage = cloud_coverage * 100
    
    if coverage_percentage < 10:
        quality_level = "Excellent"
        quality_score = 10
        recommendation = "Ideal for agricultural analysis and ML training"
    elif coverage_percentage < 25:
        quality_level = "Good"
        quality_score = 8
        recommendation = "Suitable for most agricultural applications"
    elif coverage_percentage < 50:
        quality_level = "Moderate"
        quality_score = 6
        recommendation = "May require additional filtering or augmentation"
    elif coverage_percentage < 75:
        quality_level = "Poor"
        quality_score = 3
        recommendation = "Consider using cloud removal techniques or alternative data"
    else:
        quality_level = "Very Poor"
        quality_score = 1
        recommendation = "Not suitable for agricultural analysis without cloud removal"
    
    return {
        "quality_level": quality_level,
        "quality_score": quality_score,
        "coverage_percentage": coverage_percentage,
        "recommendation": recommendation
    }

def analyze_multiple_datasets(dataset_paths: List[str]) -> Dict[str, Any]:
    """
    Analyze cloud coverage across multiple agricultural datasets.
    
    Args:
        dataset_paths: List of paths to .npz dataset files
        
    Returns:
        Dictionary containing comparative analysis results
    """
    all_results = {}
    comparative_stats = {
        "total_datasets": len(dataset_paths),
        "valid_datasets": 0,
        "quality_distribution": {},
        "coverage_ranges": [],
        "recommendations": []
    }
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            result = analyze_cloud_coverage(dataset_path)
            all_results[dataset_path] = result
            
            if result.get("valid", False):
                comparative_stats["valid_datasets"] += 1
                
                # Track quality distribution
                quality_level = result["quality_assessment"]["quality_level"]
                comparative_stats["quality_distribution"][quality_level] = \
                    comparative_stats["quality_distribution"].get(quality_level, 0) + 1
                
                # Track coverage ranges
                coverage = result["cloud_percentage"]
                comparative_stats["coverage_ranges"].append(coverage)
                
                # Collect recommendations
                recommendation = result["quality_assessment"]["recommendation"]
                if recommendation not in comparative_stats["recommendations"]:
                    comparative_stats["recommendations"].append(recommendation)
    
    # Calculate comparative statistics
    if comparative_stats["coverage_ranges"]:
        comparative_stats["mean_coverage"] = float(np.mean(comparative_stats["coverage_ranges"]))
        comparative_stats["median_coverage"] = float(np.median(comparative_stats["coverage_ranges"]))
        comparative_stats["min_coverage"] = float(np.min(comparative_stats["coverage_ranges"]))
        comparative_stats["max_coverage"] = float(np.max(comparative_stats["coverage_ranges"]))
    
    return {
        "individual_results": all_results,
        "comparative_analysis": comparative_stats
    }

def generate_cloud_report(analysis_results: Dict[str, Any], output_path: str = None) -> str:
    """
    Generate a comprehensive cloud coverage report.
    
    Args:
        analysis_results: Results from cloud coverage analysis
        output_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    if not analysis_results.get("valid", False):
        return f"❌ Analysis failed: {analysis_results.get('error', 'Unknown error')}"
    
    report_lines = []
    report_lines.append("🌤️  CLOUD COVERAGE ANALYSIS REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Dataset: {analysis_results['file_path']}")
    report_lines.append(f"Data Shape: {analysis_results['data_shape']}")
    report_lines.append(f"Data Type: {analysis_results['data_type']}")
    report_lines.append("")
    
    # Coverage statistics
    report_lines.append("📊 COVERAGE STATISTICS")
    report_lines.append("-" * 30)
    report_lines.append(f"Total Pixels: {analysis_results['total_pixels']:,}")
    report_lines.append(f"Cloud Pixels: {analysis_results['cloud_pixels']:,}")
    report_lines.append(f"Clear Pixels: {analysis_results['clear_pixels']:,}")
    report_lines.append(f"Cloud Coverage: {analysis_results['cloud_percentage']:.2f}%")
    report_lines.append(f"Clear Coverage: {analysis_results['clear_percentage']:.2f}%")
    report_lines.append("")
    
    # Quality assessment
    quality = analysis_results['quality_assessment']
    report_lines.append("🎯 QUALITY ASSESSMENT")
    report_lines.append("-" * 30)
    report_lines.append(f"Quality Level: {quality['quality_level']}")
    report_lines.append(f"Quality Score: {quality['quality_score']}/10")
    report_lines.append(f"Recommendation: {quality['recommendation']}")
    report_lines.append("")
    
    # Per-image statistics if available
    if analysis_results.get('coverage_statistics'):
        stats = analysis_results['coverage_statistics']
        report_lines.append("📈 PER-IMAGE STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Min Coverage: {stats['min_coverage']*100:.2f}%")
        report_lines.append(f"Max Coverage: {stats['max_coverage']*100:.2f}%")
        report_lines.append(f"Mean Coverage: {stats['mean_coverage']*100:.2f}%")
        report_lines.append(f"Median Coverage: {stats['median_coverage']*100:.2f}%")
        report_lines.append(f"Std Coverage: {stats['std_coverage']*100:.2f}%")
    
    report = "\n".join(report_lines)
    
    # Save report if output path specified
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report

def main():
    """Command-line interface for cloud coverage analysis."""
    parser = argparse.ArgumentParser(description="Analyze cloud coverage in agricultural imagery")
    parser.add_argument("--dataset", required=True, help="Path to .npz dataset file")
    parser.add_argument("--output", help="Output JSON file for analysis results")
    parser.add_argument("--report", help="Output text file for formatted report")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        return
    
    print(f"🌤️  Analyzing cloud coverage: {args.dataset}")
    analysis_results = analyze_cloud_coverage(args.dataset)
    
    if not analysis_results.get("valid", False):
        print(f"❌ Analysis failed: {analysis_results.get('error', 'Unknown error')}")
        return
    
    # Print summary
    print(f"\n📊 Cloud Coverage Analysis Results:")
    print(f"  Cloud Coverage: {analysis_results['cloud_percentage']:.2f}%")
    print(f"  Quality Level: {analysis_results['quality_assessment']['quality_level']}")
    print(f"  Quality Score: {analysis_results['quality_assessment']['quality_score']}/10")
    print(f"  Recommendation: {analysis_results['quality_assessment']['recommendation']}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\n💾 Analysis results saved to: {args.output}")
    
    # Generate and save report if requested
    if args.report:
        report = generate_cloud_report(analysis_results, args.report)
        print(f"\n📄 Cloud coverage report saved to: {args.report}")

if __name__ == "__main__":
    main()
