"""
Visualization and analysis functions for DNA sequence analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict


class DNAVisualizer:
    """Visualization utilities for DNA sequence analysis."""
    
    def __init__(self, output_dir: str = 'results/visualizations'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_pattern_length_distribution(self, patterns: List[str], save: bool = True):
        """
        Plot distribution of learned pattern lengths.
        
        Args:
            patterns: List of learned DNA patterns
            save: Whether to save the plot
        """
        lengths = [len(p) for p in patterns]
        length_dist = Counter(lengths)
        
        plt.figure(figsize=(10, 6))
        plt.bar(length_dist.keys(), length_dist.values(), color='steelblue', alpha=0.7)
        plt.xlabel('Pattern Length (bases)', fontsize=12)
        plt.ylabel('Number of Patterns', fontsize=12)
        plt.title('Distribution of Learned DNA Pattern Lengths\n(SentencePiece/BPE discovers variable-length patterns)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        if save:
            plt.savefig(self.output_dir / 'pattern_length_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print(f"Average pattern length: {np.mean(lengths):.2f} bases")
        print(f"Longest pattern: {max(lengths)} bases")
        print(f"Shortest pattern: {min(lengths)} bases")
    
    def plot_confusion_matrices(self, results_dict: Dict[str, Dict], save: bool = True):
        """
        Plot confusion matrices for model comparison.
        
        Args:
            results_dict: Dictionary mapping model names to result dictionaries
            save: Whether to save the plot
        """
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        colors = ['Blues', 'Greens', 'Oranges', 'Purples']
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            cm = results['confusion_matrix']
            accuracy = results['accuracy']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap=colors[idx % len(colors)], ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.4f}', fontsize=14)
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: np.ndarray, 
                               sp_extractor, top_n: int = 20, save: bool = True):
        """
        Plot top important features.
        
        Args:
            feature_importance: Feature importance scores
            sp_extractor: SentencePiece extractor to get pattern names
            top_n: Number of top features to display
            save: Whether to save the plot
        """
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        
        important_patterns = []
        for idx in top_indices:
            pattern = sp_extractor.sp.id_to_piece(idx).replace(' ', '')
            importance = feature_importance[idx]
            if pattern and pattern not in ['<unk>', '<s>', '</s>', '<pad>']:
                important_patterns.append((pattern, importance))
        
        if not important_patterns:
            print("No valid patterns found for visualization")
            return
        
        patterns, importances = zip(*important_patterns)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(patterns)), importances, color='forestgreen', alpha=0.7)
        plt.yticks(range(len(patterns)), patterns, fontsize=11)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {len(patterns)} Most Important DNA Patterns for Classification\n(Discovered by SentencePiece/BPE)', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'top_patterns_importance.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return important_patterns
    
    def plot_compression_comparison(self, compression_data: Dict[str, List[float]], save: bool = True):
        """
        Plot compression efficiency comparison.
        
        Args:
            compression_data: Dictionary with 'original', 'kmer', 'sentencepiece' token counts
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(3)
        values = [
            np.mean(compression_data['original']),
            np.mean(compression_data['kmer']),
            np.mean(compression_data['sentencepiece'])
        ]
        labels = ['Original\nSequence', 'K-mer\nTokens', 'SentencePiece\nTokens']
        colors = ['lightgray', 'steelblue', 'forestgreen']
        
        bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Average Token Count', fontsize=12)
        ax.set_title('Compression Efficiency Comparison\n(Lower is better - fewer tokens needed)', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'compression_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print compression statistics
        kmer_ratio = values[0] / values[1]
        sp_ratio = values[0] / values[2]
        reduction = (1 - values[2] / values[1]) * 100
        
        print(f"\nCompression ratios:")
        print(f"  K-mers: {kmer_ratio:.2f}x")
        print(f"  SentencePiece: {sp_ratio:.2f}x")
        print(f"\nSentencePiece achieves {reduction:.1f}% fewer tokens than k-mers")


def analyze_pattern_frequency(data: pd.DataFrame, sp_extractor, top_n: int = 20) -> Counter:
    """
    Analyze pattern frequency across all sequences.
    
    Args:
        data: DataFrame with DNA sequences
        sp_extractor: SentencePiece extractor
        top_n: Number of top patterns to display
        
    Returns:
        Counter with pattern frequencies
    """
    print("Finding most frequent patterns across all sequences...")
    pattern_counts = Counter()
    
    for seq in data['sequence']:
        tokens = sp_extractor.encode_sequence(seq, out_type=str)
        clean_tokens = [t.replace(' ', '') for t in tokens if t.replace(' ', '') and len(t.replace(' ', '')) > 1]
        pattern_counts.update(clean_tokens)
    
    print(f"\nTop {top_n} most frequent learned patterns:")
    for pattern, count in pattern_counts.most_common(top_n):
        print(f"  {pattern:15s} - appears {count:,} times")
    
    return pattern_counts


def analyze_patterns_by_species(data: pd.DataFrame, sp_extractor, top_n: int = 10) -> Dict:
    """
    Analyze pattern usage by species.
    
    Args:
        data: DataFrame with DNA sequences and species labels
        sp_extractor: SentencePiece extractor
        top_n: Number of top patterns per species
        
    Returns:
        Dictionary mapping species to pattern counters
    """
    print("Analyzing pattern usage by species...\n")
    
    species_patterns = {}
    for species in data['species'].unique():
        species_data = data[data['species'] == species]
        species_counter = Counter()
        
        for seq in species_data['sequence']:
            tokens = sp_extractor.encode_sequence(seq, out_type=str)
            clean_tokens = [t.replace(' ', '') for t in tokens if t.replace(' ', '') and len(t.replace(' ', '')) > 1]
            species_counter.update(clean_tokens)
        
        species_patterns[species] = species_counter
        
        print(f"{species.capitalize()} - Top {top_n} patterns:")
        for pattern, count in species_counter.most_common(top_n):
            print(f"  {pattern:15s} - {count:,} times")
        print()
    
    return species_patterns
