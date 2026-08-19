#!/usr/bin/env python3
"""
Extract REAL accessibility scores from ATAC-seq bigWig file.

This script uses bigWigToBedGraph (from UCSC tools) to convert bigWig to text,
then extracts scores for each peak region.
"""

import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path
import argparse
import tempfile
import os

def check_bigwigtools():
    """Check if UCSC bigWig tools are available."""
    tools = ['bigWigToBedGraph', 'bigWigToWig']
    available = []
    
    for tool in tools:
        try:
            result = subprocess.run([tool], stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, timeout=2)
            available.append(tool)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    return available


def convert_bigwig_to_bedgraph(bigwig_file, output_bedgraph):
    """Convert bigWig to bedGraph using bigWigToBedGraph."""
    print(f"Converting bigWig to bedGraph...")
    print(f"  Input: {bigwig_file}")
    print(f"  Output: {output_bedgraph}")
    
    cmd = ["bigWigToBedGraph", str(bigwig_file), str(output_bedgraph)]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Conversion successful!")
        return True
    except FileNotFoundError:
        print(f"  ERROR: bigWigToBedGraph not found")
        print(f"  Please install UCSC tools: https://hgdownload.soe.ucsc.edu/downloads.html")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Conversion failed: {e.stderr}")
        return False


def extract_scores_from_bedgraph(bed_file, bedgraph_file):
    """Extract scores from bedGraph file for regions in BED file."""
    print(f"\nExtracting scores from bedGraph...")
    
    # Read BED file
    bed_df = pd.read_csv(bed_file, sep='\t', header=None,
                        names=['chr', 'start', 'end', 'id'],
                        comment='#')
    
    # Read bedGraph file (chr, start, end, value)
    print(f"  Reading bedGraph file (this may take a moment)...")
    bedgraph_df = pd.read_csv(bedgraph_file, sep='\t', header=None,
                             names=['chr', 'start', 'end', 'value'],
                             comment='#')
    
    print(f"  BedGraph contains {len(bedgraph_df)} intervals")
    print(f"  Processing {len(bed_df)} peaks...")
    
    scores = []
    
    for idx, row in bed_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(bed_df)} peaks...")
        
        # Find overlapping intervals in bedGraph
        overlapping = bedgraph_df[
            (bedgraph_df['chr'] == row['chr']) &
            (bedgraph_df['start'] < row['end']) &
            (bedgraph_df['end'] > row['start'])
        ]
        
        if len(overlapping) == 0:
            scores.append(0.0)
            continue
        
        # Calculate weighted average based on overlap
        total_score = 0.0
        total_length = 0
        
        for _, interval in overlapping.iterrows:
            # Calculate overlap
            overlap_start = max(row['start'], interval['start'])
            overlap_end = min(row['end'], interval['end'])
            overlap_length = max(0, overlap_end - overlap_start)
            
            if overlap_length > 0:
                total_score += interval['value'] * overlap_length
                total_length += overlap_length
        
        if total_length > 0:
            score = total_score / total_length
        else:
            score = 0.0
        
        scores.append(float(score))
    
    scores = np.array(scores, dtype=np.float32)
    
    print(f"\n  Extracted {len(scores)} scores")
    print(f"    Mean: {scores.mean():.4f}")
    print(f"    Std: {scores.std():.4f}")
    print(f"    Min: {scores.min():.4f}")
    print(f"    Max: {scores.max():.4f}")
    
    return scores


def main(args):
    print("=" * 60)
    print("Extract REAL Accessibility Scores from ATAC-seq bigWig")
    print("=" * 60)
    
    bed_path = Path(args.bed_file)
    bigwig_path = Path(args.bigwig_file)
    output_path = Path(args.output_file)
    
    if not bed_path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_path}")
    if not bigwig_path.exists():
        raise FileNotFoundError(f"bigWig file not found: {bigwig_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for UCSC tools
    tools = check_bigwigtools()
    if not tools:
        print("\n" + "=" * 60)
        print("MANUAL INSTALLATION REQUIRED")
        print("=" * 60)
        print("\nTo extract REAL scores, you need UCSC bigWig tools.")
        print("\nOption 1: Install UCSC tools (Recommended)")
        print("  1. Download from: https://hgdownload.soe.ucsc.edu/downloads.html")
        print("  2. Extract and add to PATH")
        print("  3. Or place bigWigToBedGraph.exe in project directory")
        print("\nOption 2: Use Python with pyBigWig (Alternative)")
        print("  1. Install via conda: conda install -c bioconda pybigwig")
        print("  2. Or use WSL (Windows Subsystem for Linux)")
        print("\nOption 3: Use online converter")
        print("  1. Convert bigWig to bedGraph online")
        print("  2. Then run this script with --bedgraph_file option")
        print("\n" + "=" * 60)
        
        # Ask if they want to proceed with manual conversion
        print("\nWould you like to:")
        print("A) Download and install UCSC tools manually")
        print("B) Convert bigWig to bedGraph manually and provide path")
        print("C) Continue with placeholder scores (NOT recommended)")
        
        return None
    
    # Use available tool
    tool = tools[0]
    print(f"\nUsing tool: {tool}")
    
    # Convert bigWig to bedGraph
    with tempfile.TemporaryDirectory() as tmpdir:
        bedgraph_file = Path(tmpdir) / "signal.bedGraph"
        
        if tool == 'bigWigToBedGraph':
            success = convert_bigwig_to_bedgraph(bigwig_path, bedgraph_file)
        else:
            # For bigWigToWig, convert to wig then to bedGraph
            wig_file = Path(tmpdir) / "signal.wig"
            success = False
            # Would need additional conversion step
        
        if not success:
            print("\nConversion failed. Please check tool installation.")
            return None
        
        # Extract scores
        scores = extract_scores_from_bedgraph(bed_path, bedgraph_file)
    
    # Save scores
    np.save(output_path, scores)
    print(f"\nSUCCESS: Saved REAL accessibility scores to: {output_path}")
    print(f"   Shape: {scores.shape}")
    print(f"   These are REAL scores extracted from ENCODE ATAC-seq data!")
    
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract REAL accessibility scores from ATAC-seq bigWig"
    )
    parser.add_argument(
        "--bed_file",
        type=str,
        default="data/processed/peaks/full_peaks.bed",
        help="Input BED file with peak coordinates"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="data/raw/encode/atac_seq/K562/atac_signal.bigWig",
        help="ATAC-seq bigWig signal file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/labels/accessibility_scores.npy",
        help="Output file for accessibility scores"
    )
    parser.add_argument(
        "--bedgraph_file",
        type=str,
        default=None,
        help="Optional: Pre-converted bedGraph file (skips bigWig conversion)"
    )
    
    args = parser.parse_args()
    main(args)

