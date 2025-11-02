"""
DNA SentencePiece Features - A package for DNA sequence analysis using BPE.
"""

from .data_loader import DNADataLoader
from .feature_extractor import KmerFeatureExtractor, SentencePieceFeatureExtractor
from .classifier import DNAClassifier, split_data, compare_models
from .visualizer import DNAVisualizer, analyze_pattern_frequency, analyze_patterns_by_species
from . import utils

__version__ = "1.0.0"
__all__ = [
    'DNADataLoader',
    'KmerFeatureExtractor',
    'SentencePieceFeatureExtractor',
    'DNAClassifier',
    'DNAVisualizer',
    'split_data',
    'compare_models',
    'analyze_pattern_frequency',
    'analyze_patterns_by_species',
    'utils'
]
