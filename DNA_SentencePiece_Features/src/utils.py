"""
Utility functions for DNA sequence analysis.
"""
import numpy as np
from typing import List, Tuple


def get_kmers(sequence: str, size: int = 6) -> List[str]:
    """
    Extract k-mers from a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        size: Length of k-mers (default: 6)
        
    Returns:
        List of k-mer strings
    """
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


def prepare_sequence_for_sp(sequence: str) -> str:
    """
    Prepare DNA sequence for SentencePiece by adding spaces between characters.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Space-separated DNA sequence
    """
    return ' '.join(list(sequence))


def calculate_compression_ratio(original_length: float, token_count: float) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original_length: Original sequence length
        token_count: Number of tokens after encoding
        
    Returns:
        Compression ratio
    """
    return original_length / token_count if token_count > 0 else 0
