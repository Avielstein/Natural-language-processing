"""
Vocabulary Comparator - Compare vocabularies from different BPE models.
"""

import logging
from typing import Dict, Set, List, Tuple
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class VocabComparator:
    """Compare vocabularies from multiple trained models."""
    
    def __init__(self, *models):
        """
        Initialize comparator with trained models.
        
        Args:
            *models: Variable number of TrainedModel objects
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models to compare")
        
        self.models = list(models)
        self.model_names = [f"Model_{i+1}" for i in range(len(models))]
        
        logger.info(f"Initialized comparator with {len(models)} models")
    
    def set_model_names(self, names: List[str]) -> None:
        """
        Set custom names for models.
        
        Args:
            names: List of model names
        """
        if len(names) != len(self.models):
            raise ValueError(f"Need {len(self.models)} names, got {len(names)}")
        self.model_names = names
    
    def get_unique_tokens(self, model_idx: int) -> Set[str]:
        """
        Get tokens unique to a specific model.
        
        Args:
            model_idx: Index of model
        
        Returns:
            Set of unique tokens
        """
        model_tokens = set(self.models[model_idx].get_tokens())
        other_tokens = set()
        
        for i, model in enumerate(self.models):
            if i != model_idx:
                other_tokens.update(model.get_tokens())
        
        return model_tokens - other_tokens
    
    def get_shared_tokens(self) -> Set[str]:
        """
        Get tokens shared across all models.
        
        Returns:
            Set of shared tokens
        """
        shared = set(self.models[0].get_tokens())
        for model in self.models[1:]:
            shared &= set(model.get_tokens())
        return shared
    
    def get_pairwise_overlap(self, idx1: int, idx2: int) -> Dict:
        """
        Calculate overlap between two models.
        
        Args:
            idx1: Index of first model
            idx2: Index of second model
        
        Returns:
            Dictionary with overlap statistics
        """
        tokens1 = set(self.models[idx1].get_tokens())
        tokens2 = set(self.models[idx2].get_tokens())
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        jaccard = len(intersection) / len(union) if union else 0
        
        return {
            'model1_size': len(tokens1),
            'model2_size': len(tokens2),
            'intersection_size': len(intersection),
            'union_size': len(union),
            'jaccard_similarity': jaccard,
            'model1_unique': len(tokens1 - tokens2),
            'model2_unique': len(tokens2 - tokens1),
        }
    
    def get_token_length_stats(self) -> Dict[str, Dict]:
        """
        Compare token length distributions.
        
        Returns:
            Dictionary mapping model names to length statistics
        """
        stats = {}
        
        for name, model in zip(self.model_names, self.models):
            tokens = [t for t in model.get_tokens() 
                     if not t.startswith('<') and not t == '▁']
            lengths = [len(t) for t in tokens]
            
            stats[name] = {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'distribution': Counter(lengths),
            }
        
        return stats
    
    def compare_all(self) -> Dict:
        """
        Generate comprehensive comparison report.
        
        Returns:
            Dictionary containing all comparison metrics
        """
        logger.info("Generating comprehensive comparison...")
        
        comparison = {
            'num_models': len(self.models),
            'model_names': self.model_names,
            'vocab_sizes': {name: len(model.get_tokens()) 
                           for name, model in zip(self.model_names, self.models)},
            'shared_tokens': list(self.get_shared_tokens()),
            'unique_tokens': {},
            'pairwise_overlaps': {},
            'token_length_stats': self.get_token_length_stats(),
        }
        
        # Get unique tokens for each model
        for i, name in enumerate(self.model_names):
            comparison['unique_tokens'][name] = list(self.get_unique_tokens(i))
        
        # Get pairwise overlaps
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                pair_name = f"{self.model_names[i]} vs {self.model_names[j]}"
                comparison['pairwise_overlaps'][pair_name] = self.get_pairwise_overlap(i, j)
        
        return comparison
    
    def print_summary(self, comparison: Dict = None) -> None:
        """
        Print human-readable comparison summary.
        
        Args:
            comparison: Comparison dictionary (will generate if None)
        """
        if comparison is None:
            comparison = self.compare_all()
        
        print("\n" + "="*70)
        print("VOCABULARY COMPARISON SUMMARY")
        print("="*70)
        
        print(f"\nNumber of models: {comparison['num_models']}")
        print(f"Model names: {', '.join(comparison['model_names'])}")
        
        print("\n" + "-"*70)
        print("VOCABULARY SIZES")
        print("-"*70)
        for name, size in comparison['vocab_sizes'].items():
            print(f"  {name}: {size:,} tokens")
        
        print("\n" + "-"*70)
        print("SHARED TOKENS")
        print("-"*70)
        shared = comparison['shared_tokens']
        print(f"  Total shared: {len(shared):,} tokens")
        if len(shared) < 20:
            print(f"  Tokens: {shared}")
        else:
            print(f"  Sample: {shared[:10]} ...")
        
        print("\n" + "-"*70)
        print("UNIQUE TOKENS")
        print("-"*70)
        for name, tokens in comparison['unique_tokens'].items():
            print(f"\n  {name}: {len(tokens):,} unique tokens")
            if tokens:
                sample = tokens[:10] if len(tokens) > 10 else tokens
                print(f"    Sample: {sample}")
        
        print("\n" + "-"*70)
        print("PAIRWISE OVERLAPS")
        print("-"*70)
        for pair_name, stats in comparison['pairwise_overlaps'].items():
            print(f"\n  {pair_name}:")
            print(f"    Jaccard Similarity: {stats['jaccard_similarity']:.3f}")
            print(f"    Intersection: {stats['intersection_size']:,} tokens")
            print(f"    Model 1 unique: {stats['model1_unique']:,} tokens")
            print(f"    Model 2 unique: {stats['model2_unique']:,} tokens")
        
        print("\n" + "-"*70)
        print("TOKEN LENGTH STATISTICS")
        print("-"*70)
        for name, stats in comparison['token_length_stats'].items():
            print(f"\n  {name}:")
            print(f"    Mean length: {stats['mean']:.2f} chars")
            print(f"    Median length: {stats['median']:.1f} chars")
            print(f"    Range: {stats['min']}-{stats['max']} chars")
            print(f"    Std dev: {stats['std']:.2f}")
        
        print("\n" + "="*70)
    
    def get_compression_ratios(self, test_texts: Dict[str, str]) -> Dict:
        """
        Compare compression efficiency on test texts.
        
        Args:
            test_texts: Dictionary mapping text names to text samples
        
        Returns:
            Dictionary with compression statistics
        """
        results = {}
        
        for text_name, text in test_texts.items():
            results[text_name] = {}
            original_length = len(text)
            
            for model_name, model in zip(self.model_names, self.models):
                tokens = model.encode(text, out_type='int')
                compression_ratio = original_length / len(tokens) if tokens else 0
                
                results[text_name][model_name] = {
                    'num_tokens': len(tokens),
                    'compression_ratio': compression_ratio,
                    'chars_per_token': compression_ratio,
                }
        
        return results
    
    def analyze_domain_specificity(self, sample_size: int = 100) -> Dict:
        """
        Analyze how domain-specific each vocabulary is.
        
        Args:
            sample_size: Number of tokens to analyze per model
        
        Returns:
            Dictionary with domain analysis
        """
        analysis = {}
        
        for i, (name, model) in enumerate(zip(self.model_names, self.models)):
            tokens = model.get_tokens()
            
            # Get unique tokens for this model
            unique = self.get_unique_tokens(i)
            
            # Sample analysis
            sample_unique = list(unique)[:sample_size]
            
            # Categorize tokens
            categories = self._categorize_tokens(sample_unique)
            
            analysis[name] = {
                'total_unique': len(unique),
                'unique_percentage': len(unique) / len(tokens) * 100,
                'categories': categories,
                'sample_unique_tokens': sample_unique[:20],
            }
        
        return analysis
    
    def _categorize_tokens(self, tokens: List[str]) -> Dict[str, int]:
        """Categorize tokens by type."""
        categories = {
            'alphabetic': 0,
            'numeric': 0,
            'alphanumeric': 0,
            'special_chars': 0,
            'punctuation': 0,
            'mixed': 0,
        }
        
        for token in tokens:
            # Remove space marker
            clean_token = token.replace('▁', '')
            
            if clean_token.isalpha():
                categories['alphabetic'] += 1
            elif clean_token.isdigit():
                categories['numeric'] += 1
            elif clean_token.isalnum():
                categories['alphanumeric'] += 1
            elif len(clean_token) == 1 and not clean_token.isalnum():
                categories['punctuation'] += 1
            elif any(c in clean_token for c in '!@#$%^&*()[]{}'):
                categories['special_chars'] += 1
            else:
                categories['mixed'] += 1
        
        return categories


if __name__ == '__main__':
    print("VocabComparator module - use with trained models")
    print("Example:")
    print("  comparator = VocabComparator(model1, model2)")
    print("  comparator.set_model_names(['NLP', 'Scientific'])")
    print("  comparison = comparator.compare_all()")
    print("  comparator.print_summary(comparison)")
