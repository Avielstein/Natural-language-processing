#!/usr/bin/env python3
"""
Generate visualizations for stop word analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_visualizations():
    """Create all visualizations for stop word analysis."""
    
    # Data from our analysis
    corpora = ['Brown', 'Reuters', 'Webtext']
    vocab_sizes = [10000, 10000, 10000]
    stopword_counts = [320, 350, 415]
    percentages = [3.20, 3.50, 4.15]
    coverage_counts = [145, 128, 143]
    coverage_percentages = [81.5, 71.9, 80.3]
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'stopword_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Figure 1: Vocabulary Allocation (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(corpora, stopword_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Stop Word Tokens', fontsize=12, fontweight='bold')
    ax.set_xlabel('Corpus', fontsize=12, fontweight='bold')
    ax.set_title('Stop Word Token Count by Corpus', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 500)
    
    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, stopword_counts, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.2f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stopword_token_counts.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'stopword_token_counts.png'}")
    plt.close()
    
    # Figure 2: Coverage Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(corpora, coverage_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=178, color='red', linestyle='--', linewidth=2, label='Total Reference Stop Words (178)', alpha=0.7)
    ax.set_ylabel('Number of Unique Stop Words Covered', fontsize=12, fontweight='bold')
    ax.set_xlabel('Corpus', fontsize=12, fontweight='bold')
    ax.set_title('Stop Word Coverage: Unique Forms Captured', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 200)
    
    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, coverage_counts, coverage_percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}/178\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stopword_coverage.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'stopword_coverage.png'}")
    plt.close()
    
    # Figure 3: Dual Comparison (Side by Side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Percentage of Vocab
    bars1 = ax1.bar(corpora, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Percentage of Vocabulary (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Corpus', fontsize=12, fontweight='bold')
    ax1.set_title('Stop Words as % of Vocabulary', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 5)
    ax1.axhline(y=3.62, color='orange', linestyle='--', linewidth=2, label='Average (3.62%)', alpha=0.7)
    
    for bar, pct in zip(bars1, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Right: Coverage Percentage
    bars2 = ax2.bar(corpora, coverage_percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Coverage of Reference Set (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Corpus', fontsize=12, fontweight='bold')
    ax2.set_title('Coverage of 178 Reference Stop Words', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=77.9, color='orange', linestyle='--', linewidth=2, label='Average (77.9%)', alpha=0.7)
    
    for bar, pct in zip(bars2, coverage_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.suptitle('Stop Word Analysis: Allocation vs Coverage', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'stopword_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'stopword_comparison.png'}")
    plt.close()
    
    # Figure 4: Missing Stop Words (Pie Chart showing gaps)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (corpus, covered, cov_pct) in enumerate(zip(corpora, coverage_counts, coverage_percentages)):
        missing = 178 - covered
        
        sizes = [covered, missing]
        labels = [f'Covered\n({covered})', f'Missing\n({missing})']
        colors_pie = [colors[idx], '#95a5a6']
        explode = (0.05, 0)
        
        axes[idx].pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                     shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[idx].set_title(f'{corpus}\n{cov_pct:.1f}% Coverage', fontsize=12, fontweight='bold')
    
    plt.suptitle('Stop Word Coverage Distribution by Corpus', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'stopword_coverage_pie.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'stopword_coverage_pie.png'}")
    plt.close()
    
    # Figure 5: Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Token counts
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(corpora, stopword_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_title('Stop Word Tokens in Vocabulary', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Token Count', fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (count, pct) in enumerate(zip(stopword_counts, percentages)):
        ax1.text(i, count, f'{count}\n{pct:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Coverage counts
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(corpora, coverage_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=178, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_title('Unique Stop Words Covered', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count (out of 178)', fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (count, pct) in enumerate(zip(coverage_counts, coverage_percentages)):
        ax2.text(i, count, f'{count}\n{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Allocation percentage
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(corpora, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=3.62, color='orange', linestyle='--', linewidth=1.5, label='Avg: 3.62%', alpha=0.7)
    ax3.set_title('Vocabulary Space Allocation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('% of Vocabulary', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for i, pct in enumerate(percentages):
        ax3.text(i, pct, f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 4: Coverage percentage
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(corpora, coverage_percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=77.9, color='orange', linestyle='--', linewidth=1.5, label='Avg: 77.9%', alpha=0.7)
    ax4.set_title('Reference Set Coverage', fontsize=12, fontweight='bold')
    ax4.set_ylabel('% Covered', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for i, pct in enumerate(coverage_percentages):
        ax4.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 5: Combined ranking (scatter)
    ax5 = fig.add_subplot(gs[2, :])
    x_pos = np.arange(len(corpora))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, percentages, width, label='% of Vocab', 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax5_right = ax5.twinx()
    bars2 = ax5_right.bar(x_pos + width/2, coverage_percentages, width, label='% Coverage',
                          color=['#27ae60', '#2980b9', '#c0392b'], alpha=0.7, 
                          edgecolor='black', linewidth=1.5)
    
    ax5.set_xlabel('Corpus', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Allocation (% of Vocabulary)', fontsize=10, fontweight='bold', color='#2c3e50')
    ax5_right.set_ylabel('Coverage (% of Reference Set)', fontsize=10, fontweight='bold', color='#16a085')
    ax5.set_title('Allocation vs Coverage Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(corpora, fontsize=10)
    ax5.legend(loc='upper left', fontsize=9)
    ax5_right.legend(loc='upper right', fontsize=9)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Stop Word Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'stopword_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'stopword_dashboard.png'}")
    plt.close()
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    return output_dir

if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING STOP WORD ANALYSIS VISUALIZATIONS")
    print("=" * 70)
    print()
    output_dir = create_visualizations()
    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
