#!/usr/bin/env python3
"""
Windows-Friendly Script to Extract Accessibility Scores from bigWig Files

RECOMMENDED METHOD for Windows - No compilation required!

This script works on Windows WITHOUT requiring pyBigWig compilation.
It converts bigWig to bedGraph using Windows executable, then reads with pure Python.

Usage:
    python scripts/preprocess/extract_scores_windows_friendly.py

The script will:
1. Check for bigWigToBedGraph.exe in tools/ directory
2. Convert bigWig to bedGraph automatically
3. Extract scores using pure Python (no compilation needed)
4. Save scores to data/processed/labels/accessibility_scores.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import subprocess
import sys
import os


def find_bigwig_converter(tools_dir="tools"):
    """
    Find bigWigToBedGraph executable.
    
    Checks multiple locations:
    1. tools/bigWigToBedGraph.exe (recommended)
    2. Current directory
    3. System PATH
    
    Returns:
        Path to executable if found, None otherwise
    """
    # List of possible locations
    search_paths = [
        Path(tools_dir) / "bigWigToBedGraph.exe",
        Path("tools") / "bigWigToBedGraph.exe",
        Path("bigWigToBedGraph.exe"),
        Path.cwd() / "bigWigToBedGraph.exe",
    ]
    
    # Check file paths
    for path in search_paths:
        if path.exists() and path.is_file():
            return str(path.resolve())
    
    # Check if in system PATH
    try:
        result = subprocess.run(
            ["bigWigToBedGraph.exe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2
        )
        return "bigWigToBedGraph.exe"  # Found in PATH
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def convert_bigwig_to_bedgraph_windows(bigwig_file, bedgraph_file, tools_dir="tools"):
    """
    Convert bigWig to bedGraph using Windows executable.
    
    Args:
        bigwig_file: Path to input bigWig file
        bedgraph_file: Path to output bedGraph file
        tools_dir: Directory to search for executable
    
    Returns:
        (success: bool, message: str)
    """
    # Find executable
    exe_path = find_bigwig_converter(tools_dir)
    
    if exe_path is None:
        download_url = "https://hgdownload.soe.ucsc.edu/admin/exe/windows/bigWigToBedGraph.exe"
        tools_path = Path(tools_dir) / "bigWigToBedGraph.exe"
        
        error_msg = f"""
{'='*70}
ERROR: bigWigToBedGraph.exe not found!
{'='*70}

To fix this:

1. Download the Windows executable:
   URL: {download_url}
   
2. Save it to:
   {tools_path}
   
3. Create the tools directory if it doesn't exist:
   mkdir tools
   
4. Run this script again.

Alternatively, you can convert the bigWig file manually:
   - Use online converter: https://genome.ucsc.edu/cgi-bin/hgConvert
   - Or use the downloaded executable:
     {tools_path} {bigwig_file} {bedgraph_file}
   
Then run this script with --bedgraph_file option.
{'='*70}
"""
        return False, error_msg
    
    print(f"✅ Found bigWigToBedGraph.exe: {exe_path}")
    print(f"📁 Converting: {bigwig_file.name} → {bedgraph_file.name}")
    print("   This may take a few minutes for large files...")
    
    try:
        cmd = [exe_path, str(bigwig_file), str(bedgraph_file)]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for large files
        )
        print(f"✅ Conversion successful!")
        print(f"   Output: {bedgraph_file}")
        return True, "Success"
    except subprocess.TimeoutExpired:
        return False, "Conversion timed out (file may be too large)"
    except subprocess.CalledProcessError as e:
        error_msg = f"Conversion failed: {e.stderr if e.stderr else 'Unknown error'}"
        return False, error_msg
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def read_bedgraph_file(bedgraph_file):
    """
    Read bedGraph file into pandas DataFrame.
    
    Format: chrom, start, end, value
    
    Args:
        bedgraph_file: Path to bedGraph file
    
    Returns:
        DataFrame with columns: chrom, start, end, value, or None if error
    """
    bedgraph_path = Path(bedgraph_file)
    
    if not bedgraph_path.exists():
        print(f"❌ ERROR: bedGraph file not found: {bedgraph_file}")
        return None
    
    file_size_mb = bedgraph_path.stat().st_size / (1024 * 1024)
    print(f"📖 Reading bedGraph file: {bedgraph_path.name}")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    try:
        # Read bedGraph file (tab-separated, no header)
        df = pd.read_csv(
            bedgraph_file,
            sep='\t',
            header=None,
            names=['chrom', 'start', 'end', 'value'],
            dtype={'chrom': str, 'start': int, 'end': int, 'value': float},
            comment=None,  # No comment character
            skip_blank_lines=True
        )
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if len(df) == 0:
            print(f"❌ ERROR: bedGraph file is empty or has no valid entries")
            return None
        
        print(f"✅ Read {len(df):,} bedGraph entries")
        print(f"   Chromosomes: {df['chrom'].nunique()} ({', '.join(sorted(df['chrom'].unique())[:5])}{'...' if df['chrom'].nunique() > 5 else ''})")
        print(f"   Value range: [{df['value'].min():.4f}, {df['value'].max():.4f}]")
        
        return df
    except pd.errors.EmptyDataError:
        print(f"❌ ERROR: bedGraph file is empty")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to read bedGraph file: {e}")
        print(f"   Please check that the file is a valid bedGraph format")
        return None


def extract_scores_from_bedgraph(bed_file, bedgraph_file, method='mean'):
    """
    Extract accessibility scores from bedGraph file for regions in BED file.
    
    This uses pure Python - no compilation needed!
    
    Args:
        bed_file: Path to BED file with peak coordinates
        bedgraph_file: Path to bedGraph file (signal track)
        method: 'mean', 'max', or 'sum'
    
    Returns:
        Array of accessibility scores, or None if error
    """
    print("\n" + "=" * 70)
    print("Step 2: Extracting Scores from bedGraph File")
    print("=" * 70)
    
    # Read BED file
    print(f"\n📖 Reading BED file: {bed_file.name}")
    try:
        bed_df = pd.read_csv(
            bed_file,
            sep='\t',
            header=None,
            names=['chr', 'start', 'end', 'id'],
            comment='#',
            dtype={'chr': str, 'start': int, 'end': int},
            usecols=[0, 1, 2, 3]  # Only use first 4 columns
        )
        print(f"✅ Loaded {len(bed_df)} peaks")
    except Exception as e:
        print(f"❌ ERROR: Failed to read BED file: {e}")
        return None
    
    # Read bedGraph file
    print(f"\n📖 Reading bedGraph file: {bedgraph_file.name}")
    bg_df = read_bedgraph_file(bedgraph_file)
    if bg_df is None or len(bg_df) == 0:
        print("❌ ERROR: bedGraph file is empty or invalid")
        return None
    
    # Create index by chromosome for faster lookup
    print("🔍 Indexing bedGraph data by chromosome...")
    chrom_index = {}
    for chrom in bed_df['chr'].unique():
        chrom_data = bg_df[bg_df['chrom'] == chrom].copy()
        if len(chrom_data) > 0:
            chrom_index[chrom] = chrom_data
        else:
            chrom_index[chrom] = pd.DataFrame(columns=['chrom', 'start', 'end', 'value'])
    
    # Extract scores for each peak
    print(f"\n📊 Extracting scores (method: {method})...")
    print("   This may take a few minutes...")
    scores = []
    
    for idx, peak in bed_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"   ⏳ Processed {idx + 1}/{len(bed_df)} peaks ({100*(idx+1)/len(bed_df):.1f}%)...")
        
        chrom = peak['chr']
        peak_start = peak['start']
        peak_end = peak['end']
        
        # Get chromosome data from index
        if chrom not in chrom_index:
            scores.append(0.0)
            continue
        
        chrom_data = chrom_index[chrom]
        
        if len(chrom_data) == 0:
            scores.append(0.0)
            continue
        
        # Find overlapping bedGraph entries
        overlaps = chrom_data[
            (chrom_data['start'] < peak_end) & 
            (chrom_data['end'] > peak_start)
        ].copy()
        
        if len(overlaps) == 0:
            scores.append(0.0)
            continue
        
        # Calculate weighted score (handling overlaps properly)
        overlap_values = []
        for _, overlap in overlaps.iterrows():
            # Calculate actual overlap length
            overlap_start = max(int(overlap['start']), peak_start)
            overlap_end = min(int(overlap['end']), peak_end)
            overlap_length = overlap_end - overlap_start
            
            if overlap_length > 0:
                overlap_values.append({
                    'value': float(overlap['value']),
                    'weight': overlap_length
                })
        
        if not overlap_values:
            scores.append(0.0)
            continue
        
        # Calculate score based on method
        if method == 'mean':
            # Weighted mean
            total_weight = sum(v['weight'] for v in overlap_values)
            if total_weight > 0:
                score = sum(v['value'] * v['weight'] for v in overlap_values) / total_weight
            else:
                score = 0.0
        elif method == 'max':
            score = max(v['value'] for v in overlap_values)
        elif method == 'sum':
            score = sum(v['value'] * v['weight'] for v in overlap_values)
        else:
            # Default to mean
            total_weight = sum(v['weight'] for v in overlap_values)
            score = sum(v['value'] * v['weight'] for v in overlap_values) / total_weight if total_weight > 0 else 0.0
        
        scores.append(float(score))
    
    scores = np.array(scores, dtype=np.float32)
    
    print(f"\n✅ Successfully extracted {len(scores)} scores")
    print(f"\n📈 Score Statistics:")
    print(f"   Mean:  {scores.mean():.4f}")
    print(f"   Std:   {scores.std():.4f}")
    print(f"   Min:   {scores.min():.4f}")
    print(f"   Max:   {scores.max():.4f}")
    print(f"   Non-zero: {(scores > 0).sum()} / {len(scores)} ({(scores > 0).mean()*100:.1f}%)")
    
    return scores


def main(args):
    print("\n" + "=" * 70)
    print("Windows-Friendly Accessibility Score Extraction")
    print("=" * 70)
    print("\nThis script extracts REAL accessibility scores from bigWig files")
    print("without requiring pyBigWig compilation (works on Windows!)")
    print("=" * 70)
    
    # Resolve paths
    bed_path = Path(args.bed_file).resolve()
    bigwig_path = Path(args.bigwig_file).resolve()
    output_path = Path(args.output_file).resolve()
    bedgraph_path = Path(args.bedgraph_file).resolve() if args.bedgraph_file else None
    
    # Check input files
    if not bed_path.exists():
        print(f"\n❌ ERROR: BED file not found: {bed_path}")
        print(f"   Please check the path and try again.")
        return None
    
    if not bigwig_path.exists():
        print(f"\n❌ ERROR: bigWig file not found: {bigwig_path}")
        print(f"   Please check the path and try again.")
        return None
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine bedGraph file path
    if bedgraph_path is None:
        # Create bedGraph path next to bigWig file
        bedgraph_path = bigwig_path.parent / f"{bigwig_path.stem}.bedGraph"
    
    bedgraph_path = bedgraph_path.resolve()
    
    # Step 1: Convert bigWig to bedGraph if needed
    if not bedgraph_path.exists():
        print("\n" + "=" * 70)
        print("Step 1: Convert bigWig to bedGraph")
        print("=" * 70)
        
        success, message = convert_bigwig_to_bedgraph_windows(
            bigwig_path,
            bedgraph_path,
            tools_dir=args.tools_dir
        )
        
        if not success:
            print(message)
            return None
    else:
        print(f"\n✅ Found existing bedGraph file: {bedgraph_path.name}")
        print(f"   Skipping conversion step.")
    
    # Step 2: Extract scores from bedGraph
    scores = extract_scores_from_bedgraph(bed_path, bedgraph_path, method=args.method)
    
    if scores is None:
        print("\n❌ ERROR: Failed to extract scores")
        return None
    
    # Step 3: Save scores
    print("\n" + "=" * 70)
    print("Step 3: Saving Results")
    print("=" * 70)
    
    np.save(output_path, scores)
    
    print(f"\n{'='*70}")
    print("🎉 SUCCESS! Real Accessibility Scores Extracted!")
    print("="*70)
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Shape: {scores.shape}")
    print(f"   Method: {args.method}")
    print(f"\n📊 These are REAL scores extracted from ENCODE ATAC-seq data!")
    print(f"   You can now use these for training your model.")
    print(f"\n💡 Next steps:")
    print(f"   1. Replace synthetic labels with these real scores")
    print(f"   2. Re-train your model with real data")
    print(f"   3. Compare performance improvement!")
    print("="*70)
    
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Windows-friendly extraction of accessibility scores from bigWig files"
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
        "--bedgraph_file",
        type=str,
        default=None,
        help="bedGraph file (if already converted, otherwise will convert automatically)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/labels/accessibility_scores.npy",
        help="Output file for accessibility scores"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mean",
        choices=['mean', 'max', 'sum'],
        help="Method to aggregate scores per region"
    )
    parser.add_argument(
        "--tools_dir",
        type=str,
        default="tools",
        help="Directory containing bigWigToBedGraph.exe"
    )
    
    args = parser.parse_args()
    main(args)

