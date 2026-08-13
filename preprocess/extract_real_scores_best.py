#!/usr/bin/env python3
"""
Extract REAL accessibility scores from ATAC-seq bigWig file using bigWigAverageOverBed.

This is the RECOMMENDED method per UCSC documentation:
https://genome.ucsc.edu/goldenpath/help/bigWig.html

bigWigAverageOverBed computes the average score of a bigWig over each BED region,
which is exactly what we need for peak accessibility scores.
"""

import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path
import argparse

def check_bigwigaverageoverbed():
    """Check if bigWigAverageOverBed is available."""
    try:
        result = subprocess.run(['bigWigAverageOverBed'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              timeout=2)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def extract_with_bigwigaverageoverbed(bed_file, bigwig_file, output_file):
    """
    Extract average scores using bigWigAverageOverBed (UCSC tool).
    
    This is the BEST method - it directly computes average scores over BED regions.
    """
    print("Using bigWigAverageOverBed (UCSC recommended method)...")
    print(f"  BED file: {bed_file}")
    print(f"  bigWig file: {bigwig_file}")
    
    # Create temporary output file
    temp_output = output_file.parent / "temp_scores.txt"
    
    # bigWigAverageOverBed command format:
    # bigWigAverageOverBed input.bigWig input.bed output.txt
    cmd = [
        "bigWigAverageOverBed",
        str(bigwig_file),
        str(bed_file),
        str(temp_output)
    ]
    
    try:
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        # Read output file
        # Format: name, size, covered, sum, mean0, mean
        # We want the 'mean' column (last column)
        print(f"  Reading results...")
        df = pd.read_csv(temp_output, sep='\t', header=None,
                        names=['name', 'size', 'covered', 'sum', 'mean0', 'mean'])
        
        scores = df['mean'].values.astype(np.float32)
        
        # Clean up temp file
        temp_output.unlink()
        
        print(f"  SUCCESS: Extracted {len(scores)} scores")
        print(f"    Mean: {scores.mean():.4f}")
        print(f"    Std: {scores.std():.4f}")
        print(f"    Min: {scores.min():.4f}")
        print(f"    Max: {scores.max():.4f}")
        
        return scores
        
    except FileNotFoundError:
        print(f"  ERROR: bigWigAverageOverBed not found")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Command failed")
        print(f"  stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main(args):
    print("=" * 60)
    print("Extract REAL Accessibility Scores - UCSC Method")
    print("=" * 60)
    print("\nUsing bigWigAverageOverBed (UCSC recommended tool)")
    print("Reference: https://genome.ucsc.edu/goldenpath/help/bigWig.html")
    print("=" * 60)
    
    bed_path = Path(args.bed_file)
    bigwig_path = Path(args.bigwig_file)
    output_path = Path(args.output_file)
    
    if not bed_path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_path}")
    if not bigwig_path.exists():
        raise FileNotFoundError(f"bigWig file not found: {bigwig_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if tool is available
    if not check_bigwigaverageoverbed():
        print("\n" + "=" * 60)
        print("MANUAL INSTALLATION REQUIRED")
        print("=" * 60)
        print("\nbigWigAverageOverBed is not found in PATH.")
        print("\nTo install:")
        print("1. Download from: https://hgdownload.soe.ucsc.edu/admin/exe/windows/")
        print("2. Download: bigWigAverageOverBed.exe")
        print("3. Save to: tools/ folder in this project")
        print("4. Or add to system PATH")
        print("\nAlternative: Use bigWigToBedGraph + Python script")
        print("=" * 60)
        return None
    
    # Extract scores
    scores = extract_with_bigwigaverageoverbed(bed_path, bigwig_path, output_path)
    
    if scores is None:
        print("\nExtraction failed. Please check tool installation.")
        return None
    
    # Save scores
    np.save(output_path, scores)
    print(f"\n" + "=" * 60)
    print(f"SUCCESS: Saved REAL accessibility scores!")
    print(f"=" * 60)
    print(f"Output file: {output_path}")
    print(f"Shape: {scores.shape}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Max: {scores.max():.4f}")
    print(f"\nThese are REAL scores extracted from ENCODE ATAC-seq data!")
    print(f"Method: bigWigAverageOverBed (UCSC recommended)")
    
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract REAL accessibility scores using bigWigAverageOverBed (UCSC tool)"
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
    
    args = parser.parse_args()
    main(args)

