"""
Visualizer - Create plots and charts for vocabulary comparison.
"""

import os
import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class VocabVisualizer:
    """Create visualizations for vocabulary comparisons."""
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Visualizer initialized. Outputs will be saved to: {output_dir}")
    
    def plot_vocab_sizes(self, comparison: Dict, filename: str = 'vocab_sizes.png') -> str:
        """
        Plot vocabulary sizes as bar chart.
        
        Args:
            comparison: Comparison dictionary from VocabComparator
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 6))
        
        names = list(comparison['vocab_sizes'].keys())
        sizes = list(comparison['vocab_sizes'].values())
        
        bars = plt.bar(names, sizes, color=sns.color_palette('husl', len(names)))
        plt.xlabel('Corpus', fontsize=12, fontweight='bold')
        plt.ylabel('Vocabulary Size', fontsize=12, fontweight='bold')
        plt.title('Vocabulary Sizes Across Corpora', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved vocabulary sizes plot: {filepath}")
        return filepath
    
    def plot_overlap_venn(self, comparison: Dict, filename: str = 'overlap_venn.png') -> str:
        """
        Plot Venn diagram showing token overlap (for 2 corpora).
        
        Args:
            comparison: Comparison dictionary
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        if comparison['num_models'] != 2:
            logger.warning("Venn diagram only supports 2 models")
            return None
        
        try:
            from matplotlib_venn import venn2
        except ImportError:
            logger.error("matplotlib-venn not installed. Install with: pip install matplotlib-venn")
            return None
        
        plt.figure(figsize=(10, 8))
        
        names = comparison['model_names']
        unique_1 = len(comparison['unique_tokens'][names[0]])
        unique_2 = len(comparison['unique_tokens'][names[1]])
        shared = len(comparison['shared_tokens'])
        
        venn2(subsets=(unique_1, unique_2, shared), 
              set_labels=names,
              set_colors=('skyblue', 'lightcoral'),
              alpha=0.7)
        
        plt.title('Token Overlap Between Corpora', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Venn diagram: {filepath}")
        return filepath
    
    def plot_token_lengths(self, comparison: Dict, filename: str = 'token_lengths.png') -> str:
        """
        Plot token length distributions.
        
        Args:
            comparison: Comparison dictionary
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        stats = comparison['token_length_stats']
        names = list(stats.keys())
        
        # Box plot
        data = []
        labels = []
        for name in names:
            dist = stats[name]['distribution']
            # Expand distribution to list of values
            values = []
            for length, count in dist.items():
                values.extend([length] * count)
            data.append(values)
            labels.append(name)
        
        ax1.boxplot(data, labels=labels)
        ax1.set_ylabel('Token Length (characters)', fontsize=11, fontweight='bold')
        ax1.set_title('Token Length Distribution', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Bar chart of means
        means = [stats[name]['mean'] for name in names]
        colors = sns.color_palette('husl', len(names))
        bars = ax2.bar(names, means, color=colors)
        ax2.set_ylabel('Mean Token Length', fontsize=11, fontweight='bold')
        ax2.set_title('Average Token Length', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved token length plots: {filepath}")
        return filepath
    
    def plot_jaccard_similarity(self, comparison: Dict, filename: str = 'jaccard_similarity.png') -> str:
        """
        Plot Jaccard similarity matrix.
        
        Args:
            comparison: Comparison dictionary
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        names = comparison['model_names']
        n = len(names)
        
        # Create similarity matrix
        similarity_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                pair_name = f"{names[i]} vs {names[j]}"
                if pair_name in comparison['pairwise_overlaps']:
                    similarity = comparison['pairwise_overlaps'][pair_name]['jaccard_similarity']
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, fmt='.3f', 
                   xticklabels=names, yticklabels=names,
                   cmap='YlOrRd', vmin=0, vmax=1,
                   cbar_kws={'label': 'Jaccard Similarity'})
        plt.title('Vocabulary Overlap (Jaccard Similarity)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Jaccard similarity matrix: {filepath}")
        return filepath
    
    def plot_unique_token_counts(self, comparison: Dict, filename: str = 'unique_tokens.png') -> str:
        """
        Plot counts of unique vs shared tokens.
        
        Args:
            comparison: Comparison dictionary
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 6))
        
        names = comparison['model_names']
        unique_counts = [len(comparison['unique_tokens'][name]) for name in names]
        shared_count = len(comparison['shared_tokens'])
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, unique_counts, width, label='Unique', 
                       color='steelblue', alpha=0.8)
        bars2 = plt.bar(x + width/2, [shared_count] * len(names), width, 
                       label='Shared', color='coral', alpha=0.8)
        
        plt.xlabel('Corpus', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Tokens', fontsize=12, fontweight='bold')
        plt.title('Unique vs Shared Tokens', fontsize=14, fontweight='bold')
        plt.xticks(x, names)
        plt.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved unique token counts: {filepath}")
        return filepath
    
    def plot_compression_ratios(self, compression_results: Dict, 
                               filename: str = 'compression_ratios.png') -> str:
        """
        Plot compression ratios on different test texts.
        
        Args:
            compression_results: Results from VocabComparator.get_compression_ratios()
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        text_names = list(compression_results.keys())
        model_names = list(next(iter(compression_results.values())).keys())
        
        x = np.arange(len(text_names))
        width = 0.8 / len(model_names)
        
        colors = sns.color_palette('husl', len(model_names))
        
        for i, model_name in enumerate(model_names):
            ratios = [compression_results[text][model_name]['compression_ratio'] 
                     for text in text_names]
            offset = width * (i - len(model_names)/2 + 0.5)
            bars = ax.bar(x + offset, ratios, width, label=model_name, 
                         color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Test Text', fontsize=12, fontweight='bold')
        ax.set_ylabel('Compression Ratio (chars/token)', fontsize=12, fontweight='bold')
        ax.set_title('Compression Efficiency on Different Texts', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(text_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved compression ratios: {filepath}")
        return filepath
    
    def create_full_report(self, comparison: Dict, 
                          compression_results: Optional[Dict] = None) -> List[str]:
        """
        Create all visualizations.
        
        Args:
            comparison: Comparison dictionary
            compression_results: Optional compression results
        
        Returns:
            List of paths to created plots
        """
        logger.info("Creating full visualization report...")
        
        plots = []
        
        # Core plots
        plots.append(self.plot_vocab_sizes(comparison))
        plots.append(self.plot_token_lengths(comparison))
        plots.append(self.plot_unique_token_counts(comparison))
        plots.append(self.plot_jaccard_similarity(comparison))
        
        # Venn diagram for 2 models
        if comparison['num_models'] == 2:
            venn_path = self.plot_overlap_venn(comparison)
            if venn_path:
                plots.append(venn_path)
        
        # Compression ratios if provided
        if compression_results:
            plots.append(self.plot_compression_ratios(compression_results))
        
        logger.info(f"Created {len(plots)} visualization plots")
        return [p for p in plots if p is not None]


if __name__ == '__main__':
    print("VocabVisualizer module - use with comparison results")
    print("Example:")
    print("  visualizer = VocabVisualizer(output_dir='results')")
    print("  visualizer.create_full_report(comparison)")
