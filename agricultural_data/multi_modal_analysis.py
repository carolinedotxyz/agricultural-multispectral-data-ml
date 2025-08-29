#!/usr/bin/env python3
"""
Multi-Modal Agricultural Data Analysis
Professional tools for analyzing and processing RGB + NDVI agricultural imagery.
This module demonstrates multi-spectral data fusion and analysis techniques.
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

class MultiModalAgriculturalAnalyzer:
    """
    Professional multi-modal agricultural data analyzer.
    Demonstrates RGB + NDVI data processing and fusion techniques.
    """
    
    def __init__(self, output_dir: str = "./multi_modal_analysis_results"):
        """
        Initialize the multi-modal analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis parameters
        self.rgb_channels = ['red', 'green', 'blue']
        self.ndvi_range = (-1.0, 1.0)
        self.quality_thresholds = {
            'min_ndvi': -1.0,
            'max_ndvi': 1.0,
            'min_rgb': 0.0,
            'max_rgb': 1.0
        }
    
    def analyze_rgb_data(self, rgb_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze RGB agricultural imagery data.
        
        Args:
            rgb_data: RGB data array (N, 3, H, W) or (N, H, W, 3)
            
        Returns:
            Dictionary containing RGB analysis results
        """
        print("🔴 Analyzing RGB agricultural imagery...")
        
        # Ensure correct shape (N, 3, H, W)
        if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
            rgb_data = np.transpose(rgb_data, (0, 3, 1, 2))
        elif not (rgb_data.ndim == 4 and rgb_data.shape[1] == 3):
            return {"error": f"Invalid RGB shape: {rgb_data.shape}"}
        
        N, C, H, W = rgb_data.shape
        
        # Channel-wise analysis
        channel_stats = {}
        for i, channel_name in enumerate(self.rgb_channels):
            channel_data = rgb_data[:, i, :, :]
            
            channel_stats[channel_name] = {
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "min": float(np.min(channel_data)),
                "max": float(np.max(channel_data)),
                "median": float(np.median(channel_data)),
                "q25": float(np.percentile(channel_data, 25)),
                "q75": float(np.percentile(channel_data, 75))
            }
        
        # Overall RGB statistics
        overall_stats = {
            "total_images": N,
            "image_dimensions": (H, W),
            "data_shape": rgb_data.shape,
            "data_type": str(rgb_data.dtype),
            "memory_usage_mb": rgb_data.nbytes / (1024 * 1024),
            "channel_statistics": channel_stats
        }
        
        # Quality assessment
        quality_issues = []
        for i, channel_name in enumerate(self.rgb_channels):
            channel_data = rgb_data[:, i, :, :]
            
            # Check for out-of-range values
            if np.any(channel_data < self.quality_thresholds['min_rgb']):
                quality_issues.append(f"{channel_name} channel has values below {self.quality_thresholds['min_rgb']}")
            
            if np.any(channel_data > self.quality_thresholds['max_rgb']):
                quality_issues.append(f"{channel_name} channel has values above {self.quality_thresholds['max_rgb']}")
            
            # Check for NaN values
            if np.any(np.isnan(channel_data)):
                quality_issues.append(f"{channel_name} channel contains NaN values")
        
        overall_stats["quality_issues"] = quality_issues
        overall_stats["quality_score"] = max(0, 10 - len(quality_issues))
        
        print(f"✅ RGB analysis completed for {N} images ({H}x{W})")
        return overall_stats
    
    def analyze_ndvi_data(self, ndvi_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze NDVI agricultural imagery data.
        
        Args:
            ndvi_data: NDVI data array (N, H, W)
            
        Returns:
            Dictionary containing NDVI analysis results
        """
        print("🟢 Analyzing NDVI agricultural imagery...")
        
        if not (ndvi_data.ndim == 3):
            return {"error": f"Invalid NDVI shape: {ndvi_data.shape}"}
        
        N, H, W = ndvi_data.shape
        
        # Basic statistics
        basic_stats = {
            "mean": float(np.mean(ndvi_data)),
            "std": float(np.std(ndvi_data)),
            "min": float(np.min(ndvi_data)),
            "max": float(np.max(ndvi_data)),
            "median": float(np.median(ndvi_data)),
            "q25": float(np.percentile(ndvi_data, 25)),
            "q75": float(np.percentile(ndvi_data, 75))
        }
        
        # NDVI-specific analysis
        ndvi_stats = {
            "total_images": N,
            "image_dimensions": (H, W),
            "data_shape": ndvi_data.shape,
            "data_type": str(ndvi_data.dtype),
            "memory_usage_mb": ndvi_data.nbytes / (1024 * 1024),
            "basic_statistics": basic_stats
        }
        
        # NDVI range analysis
        ndvi_range_stats = {
            "negative_values": int(np.sum(ndvi_data < 0)),
            "zero_values": int(np.sum(ndvi_data == 0)),
            "positive_values": int(np.sum(ndvi_data > 0)),
            "high_vegetation": int(np.sum(ndvi_data > 0.6)),
            "moderate_vegetation": int(np.sum((ndvi_data > 0.2) & (ndvi_data <= 0.6))),
            "low_vegetation": int(np.sum((ndvi_data > 0) & (ndvi_data <= 0.2)))
        }
        
        ndvi_stats["ndvi_range_analysis"] = ndvi_range_stats
        
        # Quality assessment
        quality_issues = []
        
        # Check for out-of-range values
        if np.any(ndvi_data < self.quality_thresholds['min_ndvi']):
            quality_issues.append(f"NDVI values below {self.quality_thresholds['min_ndvi']}")
        
        if np.any(ndvi_data > self.quality_thresholds['max_ndvi']):
            quality_issues.append(f"NDVI values above {self.quality_thresholds['max_ndvi']}")
        
        # Check for NaN values
        if np.any(np.isnan(ndvi_data)):
            quality_issues.append("NDVI data contains NaN values")
        
        # Check for extreme values
        extreme_low = np.sum(ndvi_data < -0.8)
        extreme_high = np.sum(ndvi_data > 0.9)
        
        if extreme_low > 0:
            quality_issues.append(f"Found {extreme_low} extremely low NDVI values (< -0.8)")
        
        if extreme_high > 0:
            quality_issues.append(f"Found {extreme_high} extremely high NDVI values (> 0.9)")
        
        ndvi_stats["quality_issues"] = quality_issues
        ndvi_stats["quality_score"] = max(0, 10 - len(quality_issues))
        
        print(f"✅ NDVI analysis completed for {N} images ({H}x{W})")
        return ndvi_stats
    
    def analyze_multi_modal_correlation(self, 
                                       rgb_data: np.ndarray, 
                                       ndvi_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze correlation between RGB and NDVI data.
        
        Args:
            rgb_data: RGB data array (N, 3, H, W)
            ndvi_data: NDVI data array (N, H, W)
            
        Returns:
            Dictionary containing correlation analysis results
        """
        print("🔗 Analyzing RGB-NDVI correlations...")
        
        if not (rgb_data.ndim == 4 and ndvi_data.ndim == 3):
            return {"error": "Invalid data dimensions for correlation analysis"}
        
        N, C, H, W = rgb_data.shape
        
        if ndvi_data.shape != (N, H, W):
            return {"error": "RGB and NDVI data have different dimensions"}
        
        # Flatten data for correlation analysis
        rgb_flat = rgb_data.reshape(N, C, -1)  # (N, 3, H*W)
        ndvi_flat = ndvi_data.reshape(N, -1)   # (N, H*W)
        
        # Calculate correlations for each channel
        correlations = {}
        for i, channel_name in enumerate(self.rgb_channels):
            channel_corrs = []
            
            for j in range(N):
                # Calculate correlation for each image
                corr = np.corrcoef(rgb_flat[j, i, :], ndvi_flat[j, :])[0, 1]
                if not np.isnan(corr):
                    channel_corrs.append(corr)
            
            if channel_corrs:
                correlations[channel_name] = {
                    "mean_correlation": float(np.mean(channel_corrs)),
                    "std_correlation": float(np.std(channel_corrs)),
                    "min_correlation": float(np.min(channel_corrs)),
                    "max_correlation": float(np.max(channel_corrs)),
                    "correlation_samples": len(channel_corrs)
                }
        
        # Overall correlation statistics
        all_corrs = []
        for channel_corrs in correlations.values():
            if "mean_correlation" in channel_corrs:
                all_corrs.append(channel_corrs["mean_correlation"])
        
        correlation_analysis = {
            "channel_correlations": correlations,
            "overall_correlation": {
                "mean": float(np.mean(all_corrs)) if all_corrs else 0.0,
                "std": float(np.std(all_corrs)) if all_corrs else 0.0,
                "min": float(np.min(all_corrs)) if all_corrs else 0.0,
                "max": float(np.max(all_corrs)) if all_corrs else 0.0
            },
            "analysis_metadata": {
                "total_images": N,
                "image_dimensions": (H, W),
                "correlation_method": "Pearson correlation coefficient"
            }
        }
        
        print(f"✅ Correlation analysis completed for {N} images")
        return correlation_analysis
    
    def generate_multi_modal_report(self, 
                                   rgb_analysis: Dict[str, Any],
                                   ndvi_analysis: Dict[str, Any],
                                   correlation_analysis: Dict[str, Any] = None) -> str:
        """
        Generate a comprehensive multi-modal analysis report.
        
        Args:
            rgb_analysis: Results from RGB analysis
            ndvi_analysis: Results from NDVI analysis
            correlation_analysis: Results from correlation analysis (optional)
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("🔴🟢 MULTI-MODAL AGRICULTURAL ANALYSIS REPORT")
        report_lines.append("=" * 60)
        
        # RGB Analysis Summary
        if "error" not in rgb_analysis:
            report_lines.append("🔴 RGB ANALYSIS SUMMARY")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Images: {rgb_analysis.get('total_images', 'N/A')}")
            report_lines.append(f"Image Dimensions: {rgb_analysis.get('image_dimensions', 'N/A')}")
            report_lines.append(f"Quality Score: {rgb_analysis.get('quality_score', 'N/A')}/10")
            
            if rgb_analysis.get('quality_issues'):
                report_lines.append(f"Quality Issues: {len(rgb_analysis['quality_issues'])}")
                for issue in rgb_analysis['quality_issues'][:3]:  # Show first 3
                    report_lines.append(f"  - {issue}")
            
            # Channel statistics
            if 'channel_statistics' in rgb_analysis:
                report_lines.append("\nChannel Statistics:")
                for channel, stats in rgb_analysis['channel_statistics'].items():
                    report_lines.append(f"  {channel}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        else:
            report_lines.append("🔴 RGB Analysis: ❌ Failed")
            report_lines.append(f"  Error: {rgb_analysis['error']}")
        
        report_lines.append("")
        
        # NDVI Analysis Summary
        if "error" not in ndvi_analysis:
            report_lines.append("🟢 NDVI ANALYSIS SUMMARY")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Images: {ndvi_analysis.get('total_images', 'N/A')}")
            report_lines.append(f"Image Dimensions: {ndvi_analysis.get('image_dimensions', 'N/A')}")
            report_lines.append(f"Quality Score: {ndvi_analysis.get('quality_score', 'N/A')}/10")
            
            if ndvi_analysis.get('quality_issues'):
                report_lines.append(f"Quality Issues: {len(ndvi_analysis['quality_issues'])}")
                for issue in ndvi_analysis['quality_issues'][:3]:  # Show first 3
                    report_lines.append(f"  - {issue}")
            
            # NDVI range analysis
            if 'ndvi_range_analysis' in ndvi_analysis:
                range_stats = ndvi_analysis['ndvi_range_analysis']
                report_lines.append(f"\nNDVI Range Analysis:")
                report_lines.append(f"  High Vegetation (>0.6): {range_stats.get('high_vegetation', 0):,}")
                report_lines.append(f"  Moderate Vegetation (0.2-0.6): {range_stats.get('moderate_vegetation', 0):,}")
                report_lines.append(f"  Low Vegetation (0-0.2): {range_stats.get('low_vegetation', 0):,}")
        else:
            report_lines.append("🟢 NDVI Analysis: ❌ Failed")
            report_lines.append(f"  Error: {ndvi_analysis['error']}")
        
        # Correlation Analysis Summary
        if correlation_analysis and "error" not in correlation_analysis:
            report_lines.append("\n🔗 CORRELATION ANALYSIS SUMMARY")
            report_lines.append("-" * 30)
            
            overall_corr = correlation_analysis.get('overall_correlation', {})
            report_lines.append(f"Overall RGB-NDVI Correlation:")
            report_lines.append(f"  Mean: {overall_corr.get('mean', 'N/A'):.3f}")
            report_lines.append(f"  Range: [{overall_corr.get('min', 'N/A'):.3f}, {overall_corr.get('max', 'N/A'):.3f}]")
            
            # Channel correlations
            if 'channel_correlations' in correlation_analysis:
                report_lines.append(f"\nChannel Correlations:")
                for channel, corr_stats in correlation_analysis['channel_correlations'].items():
                    mean_corr = corr_stats.get('mean_correlation', 0)
                    report_lines.append(f"  {channel}: {mean_corr:.3f}")
        
        return "\n".join(report_lines)
    
    def save_analysis_results(self, 
                             analysis_results: Dict[str, Any], 
                             filename: str) -> str:
        """
        Save multi-modal analysis results to disk.
        
        Args:
            analysis_results: Analysis results to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            print(f"💾 Analysis results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Failed to save analysis results: {e}")
            return None

def main():
    """Example usage of the MultiModalAgriculturalAnalyzer."""
    print("🔴🟢 Multi-Modal Agricultural Analysis Example")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MultiModalAgriculturalAnalyzer()
    
    # Generate synthetic data for demonstration
    print("\n📊 Generating synthetic multi-modal data for demonstration...")
    
    N, H, W = 100, 64, 64
    
    # Synthetic RGB data (N, 3, H, W)
    rgb_data = np.random.rand(N, 3, H, W).astype(np.float32)
    
    # Synthetic NDVI data (N, H, W) - simulate vegetation patterns
    ndvi_data = np.random.rand(N, H, W).astype(np.float32) * 2 - 1  # Range [-1, 1]
    
    # Add some correlation between RGB green channel and NDVI
    green_channel = rgb_data[:, 1, :, :]
    ndvi_data = 0.3 * green_channel + 0.7 * ndvi_data
    ndvi_data = np.clip(ndvi_data, -1, 1)
    
    print(f"✅ Generated synthetic data: {N} images, {H}x{W} pixels")
    
    # Analyze RGB data
    print("\n1️⃣ Analyzing RGB data...")
    rgb_analysis = analyzer.analyze_rgb_data(rgb_data)
    
    # Analyze NDVI data
    print("\n2️⃣ Analyzing NDVI data...")
    ndvi_analysis = analyzer.analyze_ndvi_data(ndvi_data)
    
    # Analyze correlations
    print("\n3️⃣ Analyzing RGB-NDVI correlations...")
    correlation_analysis = analyzer.analyze_multi_modal_correlation(rgb_data, ndvi_data)
    
    # Generate comprehensive report
    print("\n4️⃣ Generating multi-modal analysis report...")
    report = analyzer.generate_multi_modal_report(rgb_analysis, ndvi_analysis, correlation_analysis)
    
    # Save results
    print("\n5️⃣ Saving analysis results...")
    all_results = {
        "rgb_analysis": rgb_analysis,
        "ndvi_analysis": ndvi_analysis,
        "correlation_analysis": correlation_analysis,
        "analysis_metadata": {
            "timestamp": "2025-01-28",
            "data_dimensions": (N, H, W),
            "analysis_type": "multi_modal_agricultural"
        }
    }
    
    analyzer.save_analysis_results(all_results, "multi_modal_analysis_results.json")
    
    # Display report
    print(f"\n{report}")
    
    print("\n✅ Multi-modal agricultural analysis example completed!")

if __name__ == "__main__":
    main()
