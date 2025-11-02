"""
Motif Analysis and Visualization for BPE-discovered DNA patterns.

This module provides tools to analyze and visualize the biological motifs
discovered by Byte Pair Encoding, including:
- Pattern length distributions
- Feature importance for classification
- Comparison with known biological motifs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple
import sentencepiece as spm


class MotifAnalyzer:
    """Analyze and visualize BPE-discovered DNA motifs."""
    
    def __init__(self, sp_model_path: str):
        """
        Initialize motif analyzer with trained SentencePiece model.
        
        Args:
            sp_model_path: Path to trained .model file
        """
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.vocab_size = self.sp.get_piece_size()
        
    def extract_patterns(self) -> List[str]:
        """
        Extract multi-character patterns from vocabulary.
        
        Returns:
            List of learned patterns (excluding special tokens)
        """
        patterns = []
        for i in range(self.vocab_size):
            piece = self.sp.id_to_piece(i)
            # Remove special tokens and single bases (except informative ones)
            if piece not in ['<unk>', '<s>', '</s>', '<pad>'] and '▁' not in piece:
                clean_piece = piece.replace(' ', '')
                if clean_piece and len(clean_piece) >= 2:  # Multi-character patterns
                    patterns.append(clean_piece)
        return patterns
    
    def analyze_pattern_lengths(self, patterns: List[str]) -> pd.DataFrame:
        """
        Analyze distribution of pattern lengths.
        
        Args:
            patterns: List of BPE patterns
            
        Returns:
            DataFrame with length statistics
        """
        lengths = [len(p) for p in patterns]
        length_counts = Counter(lengths)
        
        df = pd.DataFrame([
            {'Length': length, 'Count': count, 'Percentage': (count/len(patterns))*100}
            for length, count in sorted(length_counts.items())
        ])
        
        return df
    
    def identify_biological_motifs(self, patterns: List[str]) -> Dict[str, List[str]]:
        """
        Classify patterns into known biological motif categories.
        
        Args:
            patterns: List of BPE patterns
            
        Returns:
            Dictionary mapping motif types to examples
        """
        motifs = {
            'CpG Islands': [],
            'TATA-box Variants': [],
            'Poly-A Regions': [],
            'Poly-T Regions': [],
            'GC-rich Regions': [],
            'AT-rich Regions': [],
            'Other': []
        }
        
        for pattern in patterns:
            pattern_upper = pattern.upper()
            
            # CpG islands: CG-rich patterns
            if pattern_upper.count('CG') >= 2 or 'CGCG' in pattern_upper:
                motifs['CpG Islands'].append(pattern)
            
            # TATA-box variants
            elif 'TATA' in pattern_upper or pattern_upper.count('TA') >= 2:
                motifs['TATA-box Variants'].append(pattern)
            
            # Poly-A regions
            elif pattern_upper.count('A') / len(pattern_upper) >= 0.75:
                motifs['Poly-A Regions'].append(pattern)
            
            # Poly-T regions
            elif pattern_upper.count('T') / len(pattern_upper) >= 0.75:
                motifs['Poly-T Regions'].append(pattern)
            
            # GC-rich regions
            elif (pattern_upper.count('G') + pattern_upper.count('C')) / len(pattern_upper) >= 0.7:
                motifs['GC-rich Regions'].append(pattern)
            
            # AT-rich regions
            elif (pattern_upper.count('A') + pattern_upper.count('T')) / len(pattern_upper) >= 0.7:
                motifs['AT-rich Regions'].append(pattern)
            
            # Other patterns
            else:
                motifs['Other'].append(pattern)
        
        return motifs
    
    def plot_length_distribution(self, save_path: str = None):
        """
        Create visualization of pattern length distribution.
        
        Args:
            save_path: Optional path to save figure
        """
        patterns = self.extract_patterns()
        df = self.analyze_pattern_lengths(patterns)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar plot of counts
        ax1.bar(df['Length'], df['Count'], color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Pattern Length (bp)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Patterns', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of BPE Pattern Lengths', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, row in df.iterrows():
            ax1.text(row['Length'], row['Count'], str(int(row['Count'])), 
                    ha='center', va='bottom', fontweight='bold')
        
        # Pie chart of percentages
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        ax2.pie(df['Count'], labels=[f"{int(l)} bp" for l in df['Length']], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Pattern Length Composition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PATTERN LENGTH STATISTICS")
        print("="*60)
        print(f"\nTotal patterns discovered: {len(patterns)}")
        print(f"Shortest pattern: {df['Length'].min()} bp")
        print(f"Longest pattern: {df['Length'].max()} bp")
        print(f"Most common length: {df.loc[df['Count'].idxmax(), 'Length']:.0f} bp " +
              f"({df.loc[df['Count'].idxmax(), 'Count']:.0f} patterns, " +
              f"{df.loc[df['Count'].idxmax(), 'Percentage']:.1f}%)")
        print("\n" + df.to_string(index=False))
        
        return df
    
    def plot_motif_categories(self, save_path: str = None):
        """
        Visualize discovered biological motif categories.
        
        Args:
            save_path: Optional path to save figure
        """
        patterns = self.extract_patterns()
        motifs = self.identify_biological_motifs(patterns)
        
        # Count patterns in each category
        categories = []
        counts = []
        for category, pattern_list in motifs.items():
            if len(pattern_list) > 0:  # Only include non-empty categories
                categories.append(category)
                counts.append(len(pattern_list))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#95a5a6']
        bars = ax1.barh(categories, counts, color=colors[:len(categories)], alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Number of Patterns', fontsize=12, fontweight='bold')
        ax1.set_title('Discovered Biological Motif Categories', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(count, i, f' {count}', va='center', fontweight='bold')
        
        # Pie chart
        ax2.pie(counts, labels=categories, autopct='%1.1f%%', 
                colors=colors[:len(categories)], startangle=45)
        ax2.set_title('Motif Category Composition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        # Print examples
        print("\n" + "="*60)
        print("DISCOVERED BIOLOGICAL MOTIFS")
        print("="*60)
        
        for category, pattern_list in motifs.items():
            if len(pattern_list) > 0:
                print(f"\n{category}: {len(pattern_list)} patterns")
                print(f"Examples: {', '.join(pattern_list[:10])}")
                if len(pattern_list) > 10:
                    print(f"... and {len(pattern_list) - 10} more")
        
        return motifs
    
    def plot_feature_importance(self, feature_importance: np.ndarray, 
                                top_n: int = 20, save_path: str = None):
        """
        Visualize most important patterns for classification.
        
        Args:
            feature_importance: Array of feature importance scores (from RF)
            top_n: Number of top features to display
            save_path: Optional path to save figure
        """
        patterns = []
        for i in range(min(self.vocab_size, len(feature_importance))):
            piece = self.sp.id_to_piece(i)
            if piece not in ['<unk>', '<s>', '</s>', '<pad>']:
                patterns.append(piece)
            else:
                patterns.append(f"[{piece}]")
        
        # Get top N features
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        top_patterns = [patterns[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        # Classify patterns biologically
        pattern_categories = []
        for pattern in top_patterns:
            if 'CG' in pattern.upper() and pattern.count('CG') >= 2:
                pattern_categories.append('CpG Island')
            elif 'TATA' in pattern.upper():
                pattern_categories.append('TATA-box')
            elif pattern.upper().count('A') / len(pattern) >= 0.75:
                pattern_categories.append('Poly-A')
            elif pattern.upper().count('T') / len(pattern) >= 0.75:
                pattern_categories.append('Poly-T')
            elif (pattern.upper().count('G') + pattern.upper().count('C')) / len(pattern) >= 0.7:
                pattern_categories.append('GC-rich')
            else:
                pattern_categories.append('Other')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by category
        category_colors = {
            'CpG Island': '#e74c3c',
            'TATA-box': '#3498db',
            'Poly-A': '#2ecc71',
            'Poly-T': '#f39c12',
            'GC-rich': '#9b59b6',
            'Other': '#95a5a6'
        }
        colors = [category_colors[cat] for cat in pattern_categories]
        
        bars = ax.barh(range(top_n), top_importance, color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_patterns, fontsize=10, family='monospace')
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Discriminative DNA Patterns', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat, alpha=0.8) 
                          for cat, color in category_colors.items()]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_importance)):
            ax.text(importance, i, f' {importance:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        # Print table
        print("\n" + "="*80)
        print("TOP DISCRIMINATIVE PATTERNS FOR CLASSIFICATION")
        print("="*80)
        print(f"\n{'Rank':<6}{'Pattern':<15}{'Importance':<12}{'Category':<15}{'Interpretation'}")
        print("-"*80)
        
        interpretations = {
            'CpG Island': 'Gene regulation, promoter region',
            'TATA-box': 'Transcription initiation site',
            'Poly-A': 'mRNA stability, structural element',
            'Poly-T': 'DNA flexibility, structural element',
            'GC-rich': 'Genomic stability, gene-dense region',
            'Other': 'Potential novel motif'
        }
        
        for i, (pattern, importance, category) in enumerate(zip(top_patterns, top_importance, pattern_categories), 1):
            print(f"{i:<6}{pattern:<15}{importance:<12.4f}{category:<15}{interpretations[category]}")
        
        return top_patterns, top_importance, pattern_categories
    
    def generate_report(self, feature_importance: np.ndarray = None, 
                       output_dir: str = 'results'):
        """
        Generate comprehensive motif analysis report with all visualizations.
        
        Args:
            feature_importance: Optional array of feature importance from classifier
            output_dir: Directory to save figures
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MOTIF ANALYSIS REPORT")
        print("="*80)
        
        # 1. Pattern length distribution
        print("\n[1/3] Analyzing pattern length distribution...")
        self.plot_length_distribution(save_path=f'{output_dir}/pattern_lengths.png')
        
        # 2. Biological motif categories
        print("\n[2/3] Classifying biological motifs...")
        self.plot_motif_categories(save_path=f'{output_dir}/motif_categories.png')
        
        # 3. Feature importance (if provided)
        if feature_importance is not None:
            print("\n[3/3] Analyzing feature importance for classification...")
            self.plot_feature_importance(feature_importance, top_n=20, 
                                        save_path=f'{output_dir}/feature_importance.png')
        else:
            print("\n[3/3] Skipping feature importance (no classifier provided)")
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80)
        print(f"\nFigures saved to: {output_dir}/")
        print("- pattern_lengths.png")
        print("- motif_categories.png")
        if feature_importance is not None:
            print("- feature_importance.png")


def compare_with_known_motifs():
    """
    Compare discovered patterns with well-known biological motifs from literature.
    """
    known_motifs = {
        'TATA Box': {
            'consensus': 'TATAAA',
            'variants': ['TATAA', 'TATAAAT', 'TATA'],
            'function': 'Transcription initiation',
            'reference': 'Breathnach & Chambon (1981)'
        },
        'CAAT Box': {
            'consensus': 'GGCCAATCT',
            'variants': ['CAAT', 'CCAAT'],
            'function': 'Transcription enhancement',
            'reference': 'Breathnach & Chambon (1981)'
        },
        'GC Box': {
            'consensus': 'GGGCGG',
            'variants': ['GGGCGG', 'GGGGCGGGG'],
            'function': 'Promoter element',
            'reference': 'Kadonaga et al. (1986)'
        },
        'CpG Island': {
            'consensus': 'CG-rich region',
            'variants': ['CG', 'CGCG', 'CGCGCG'],
            'function': 'Gene regulation, methylation',
            'reference': 'Bird (1986)'
        },
        'Poly-A Signal': {
            'consensus': 'AATAAA',
            'variants': ['AATAAA', 'ATTAAA', 'AAAA'],
            'function': 'mRNA polyadenylation',
            'reference': 'Proudfoot (2011)'
        }
    }
    
    print("\n" + "="*80)
    print("COMPARISON WITH KNOWN BIOLOGICAL MOTIFS")
    print("="*80)
    
    for motif_name, info in known_motifs.items():
        print(f"\n{motif_name}:")
        print(f"  Consensus: {info['consensus']}")
        print(f"  Function: {info['function']}")
        print(f"  Reference: {info['reference']}")
        print(f"  Known variants: {', '.join(info['variants'])}")
    
    print("\n" + "="*80)
    print("Note: Run MotifAnalyzer.generate_report() to see which patterns were discovered")
    print("="*80)
    
    return known_motifs


if __name__ == "__main__":
    # Example usage
    print("Motif Analyzer Module")
    print("=====================")
    print("\nTo use this module:")
    print("1. Train your BPE model")
    print("2. Create analyzer: analyzer = MotifAnalyzer('models/dna_sp.model')")
    print("3. Generate report: analyzer.generate_report(feature_importance)")
    print("\nOr run from demo script for complete analysis.")
