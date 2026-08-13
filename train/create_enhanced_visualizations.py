#!/usr/bin/env python3
"""
Create enhanced, publication-ready visualizations for model evaluation.

Generates:
- Enhanced ROC and PR curves
- Predictions vs labels scatter plot with statistics
- Distribution plots
- Comprehensive evaluation summary figure
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import pandas as pd

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_evaluation_results(metrics_path, predictions_path=None):
    """Load evaluation metrics and optionally predictions."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    predictions = None
    labels = None
    if predictions_path and Path(predictions_path).exists():
        data = np.load(predictions_path, allow_pickle=True)
        predictions = data['predictions']
        labels = data['labels']
    
    return metrics, predictions, labels


def create_comprehensive_figure(metrics, predictions, labels, output_path):
    """Create a comprehensive 2x2 figure with all evaluation plots."""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. ROC Curve (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if predictions is not None and labels is not None:
        threshold = metrics.get('threshold', np.median(labels))
        binary_labels = (labels >= threshold).astype(int)
        try:
            fpr, tpr, _ = roc_curve(binary_labels, predictions)
            roc_auc = roc_auc_score(binary_labels, predictions)
            ax1.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
        except ValueError:
            ax1.text(0.5, 0.5, 'ROC curve not available', ha='center', va='center')
            ax1.set_title('ROC Curve')
    
    # 2. Precision-Recall Curve (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if predictions is not None and labels is not None:
        threshold = metrics.get('threshold', np.median(labels))
        binary_labels = (labels >= threshold).astype(int)
        try:
            precision, recall, _ = precision_recall_curve(binary_labels, predictions)
            pr_auc = average_precision_score(binary_labels, predictions)
            ax2.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
            baseline = binary_labels.mean()
            ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                       label=f'Baseline (AP = {baseline:.3f})')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend(loc='lower left')
            ax2.grid(True, alpha=0.3)
        except ValueError:
            ax2.text(0.5, 0.5, 'PR curve not available', ha='center', va='center')
            ax2.set_title('Precision-Recall Curve')
    
    # 3. Predictions vs Labels Scatter (bottom left)
    ax3 = fig.add_subplot(gs[1, :])
    if predictions is not None and labels is not None:
        # Scatter plot with density
        ax3.scatter(labels, predictions, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Add correlation text
        pearson_r = metrics.get('Pearson_correlation', 0)
        spearman_r = metrics.get('Spearman_correlation', 0)
        mse = metrics.get('MSE', 0)
        
        # Add diagonal line
        min_val = min(labels.min(), predictions.min())
        max_val = max(labels.max(), predictions.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect prediction', alpha=0.7)
        
        # Add regression line
        z = np.polyfit(labels, predictions, 1)
        p = np.poly1d(z)
        ax3.plot(labels, p(labels), "b--", alpha=0.5, linewidth=1.5, label='Linear fit')
        
        ax3.set_xlabel('True Accessibility Scores', fontweight='bold')
        ax3.set_ylabel('Predicted Accessibility Scores', fontweight='bold')
        ax3.set_title('Predictions vs True Labels', fontweight='bold')
        
        # Add statistics text box
        stats_text = f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}\nMSE = {mse:.4f}'
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
    
    # 4. Distribution comparison (bottom right)
    ax4 = fig.add_subplot(gs[2, 0])
    if predictions is not None and labels is not None:
        ax4.hist(labels, bins=20, alpha=0.6, label='True Labels', color='blue', edgecolor='black')
        ax4.hist(predictions, bins=20, alpha=0.6, label='Predictions', color='red', edgecolor='black')
        ax4.set_xlabel('Accessibility Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Metrics summary table (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Create metrics table
    metric_names = [
        'MSE',
        'Pearson Correlation',
        'Spearman Correlation',
        'ROC-AUC',
        'PR-AUC'
    ]
    
    metric_values = [
        f"{metrics.get('MSE', 0):.4f}",
        f"{metrics.get('Pearson_correlation', 0):.4f}",
        f"{metrics.get('Spearman_correlation', 0):.4f}",
        f"{metrics.get('ROC_AUC', 0):.4f}" if metrics.get('ROC_AUC') is not None else 'N/A',
        f"{metrics.get('PR_AUC', 0):.4f}" if metrics.get('PR_AUC') is not None else 'N/A'
    ]
    
    table_data = list(zip(metric_names, metric_values))
    table = ax5.table(cellText=table_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(metric_names) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax5.set_title('Evaluation Metrics Summary', fontweight='bold', pad=20)
    
    plt.suptitle('EpiBERT Model Evaluation Results', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"  Saved comprehensive figure: {output_path}")
    plt.close()


def create_individual_plots(metrics, predictions, labels, output_dir):
    """Create individual high-quality plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if predictions is None or labels is None:
        print("  Warning: Predictions/labels not available for individual plots")
        return
    
    threshold = metrics.get('threshold', np.median(labels))
    binary_labels = (labels >= threshold).astype(int)
    
    # 1. Enhanced scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(labels, predictions, alpha=0.7, s=50, edgecolors='black', linewidth=0.5, c=labels, cmap='viridis')
    
    # Add diagonal
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Add regression line
    z = np.polyfit(labels, predictions, 1)
    p = np.poly1d(z)
    ax.plot(labels, p(labels), "b--", alpha=0.7, linewidth=2, label='Linear fit')
    
    pearson_r = metrics.get('Pearson_correlation', 0)
    spearman_r = metrics.get('Spearman_correlation', 0)
    mse = metrics.get('MSE', 0)
    
    ax.set_xlabel('True Accessibility Scores', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Accessibility Scores', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Predictions vs True Labels\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}, MSE={mse:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='True Score')
    plt.tight_layout()
    plt.savefig(output_dir / "enhanced_scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved enhanced scatter plot: {output_dir / 'enhanced_scatter_plot.png'}")


def main():
    """Main function to create all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create enhanced evaluation visualizations")
    parser.add_argument("--metrics_file", type=str, 
                       default="results/evaluation/evaluation_metrics.json",
                       help="Path to evaluation metrics JSON file")
    parser.add_argument("--predictions_file", type=str, 
                       default="results/evaluation/predictions_and_labels.npz",
                       help="Path to predictions numpy file (optional)")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Creating Enhanced Visualizations")
    print("=" * 60)
    
    # Load metrics
    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        return
    
    metrics, predictions, labels = load_evaluation_results(metrics_path, args.predictions_file)
    
    print(f"\nLoaded metrics from: {metrics_path}")
    print(f"  MSE: {metrics.get('MSE', 0):.4f}")
    print(f"  Pearson: {metrics.get('Pearson_correlation', 0):.4f}")
    print(f"  Spearman: {metrics.get('Spearman_correlation', 0):.4f}")
    print(f"  ROC-AUC: {metrics.get('ROC_AUC', 'N/A')}")
    print(f"  PR-AUC: {metrics.get('PR_AUC', 'N/A')}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive figure
    print(f"\nGenerating visualizations...")
    create_comprehensive_figure(metrics, predictions, labels, 
                               output_dir / "comprehensive_evaluation.png")
    
    # Create individual plots if predictions available
    if predictions is not None and labels is not None:
        create_individual_plots(metrics, predictions, labels, output_dir)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

