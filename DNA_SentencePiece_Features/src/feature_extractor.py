"""
Feature extraction from DNA sequences using k-mers and SentencePiece.
"""
import numpy as np
import pandas as pd
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple
from .utils import get_kmers, prepare_sequence_for_sp


class KmerFeatureExtractor:
    """Extract k-mer features from DNA sequences."""
    
    def __init__(self, kmer_size: int = 6, max_features: int = 1500):
        """
        Initialize k-mer feature extractor.
        
        Args:
            kmer_size: Length of k-mers
            max_features: Maximum number of features to use
        """
        self.kmer_size = kmer_size
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=max_features)
    
    def extract_kmers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract k-mers from all sequences.
        
        Args:
            data: DataFrame with 'sequence' column
            
        Returns:
            DataFrame with added k-mer columns
        """
        print(f"Extracting {self.kmer_size}-mers...")
        data['kmers'] = data['sequence'].apply(lambda x: get_kmers(x, size=self.kmer_size))
        data['kmer_text'] = data['kmers'].apply(lambda x: ' '.join(x))
        print(f"Done! Each sequence now has {self.kmer_size}-mers extracted.")
        return data
    
    def fit_transform(self, kmer_texts: pd.Series) -> np.ndarray:
        """
        Fit vectorizer and transform k-mer texts to feature matrix.
        
        Args:
            kmer_texts: Series of k-mer text strings
            
        Returns:
            Feature matrix
        """
        print("Creating k-mer feature matrix...")
        X = self.vectorizer.fit_transform(kmer_texts).toarray()
        print(f"K-mer feature matrix shape: {X.shape}")
        print(f"Using top {X.shape[1]} most frequent {self.kmer_size}-mers as features")
        return X
    
    def get_feature_names(self):
        """Get k-mer feature names."""
        return self.vectorizer.get_feature_names_out()


class SentencePieceFeatureExtractor:
    """Extract features using SentencePiece/BPE."""
    
    def __init__(self, vocab_size: int = 1500, model_prefix: str = 'models/dna_sp'):
        """
        Initialize SentencePiece feature extractor.
        
        Args:
            vocab_size: Vocabulary size for SentencePiece
            model_prefix: Prefix for saved model files
        """
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp = None
    
    def train(self, input_file: str = 'data/dna_sequences.txt'):
        """
        Train SentencePiece model using BPE.
        
        Args:
            input_file: Path to training data file
        """
        print(f"Training SentencePiece BPE model with vocabulary size {self.vocab_size}...")
        
        # Train BPE model on raw DNA sequences
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',  # Byte Pair Encoding
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            num_threads=1,
            max_sentencepiece_length=16,  # Allow longer patterns
            split_by_whitespace=False,  # Don't split by whitespace
            split_by_unicode_script=False,
            split_by_number=False,
            byte_fallback=False,
            normalization_rule_name='identity',  # No normalization
            add_dummy_prefix=False  # Don't add ▁ prefix for DNA sequences
        )
        print("SentencePiece BPE model trained successfully!")
    
    def load(self, model_file: str = None):
        """
        Load trained SentencePiece model.
        
        Args:
            model_file: Path to model file (default: uses model_prefix)
        """
        if model_file is None:
            model_file = f"{self.model_prefix}.model"
        
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        print(f"Loaded SentencePiece model with {self.sp.get_piece_size()} pieces")
    
    def encode_sequence(self, sequence: str, out_type=int) -> list:
        """
        Encode a single sequence using BPE.
        
        Args:
            sequence: DNA sequence string (raw, no spacing needed)
            out_type: Output type (int or str)
            
        Returns:
            List of token IDs or strings
        """
        if self.sp is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Encode raw DNA sequence directly - BPE will split it
        return self.sp.encode(sequence, out_type=out_type)
    
    def encode_to_features(self, sequence: str) -> np.ndarray:
        """
        Convert a DNA sequence to a feature vector.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Feature vector (token counts)
        """
        token_ids = self.encode_sequence(sequence, out_type=int)
        
        # Create count vector
        feature_vector = np.zeros(self.vocab_size)
        for token_id in token_ids:
            if token_id < self.vocab_size:
                feature_vector[token_id] += 1
        
        return feature_vector
    
    def extract_features(self, sequences: pd.Series) -> np.ndarray:
        """
        Extract SentencePiece features for all sequences.
        
        Args:
            sequences: Series of DNA sequences
            
        Returns:
            Feature matrix
        """
        print("Extracting SentencePiece features...")
        X = np.array([self.encode_to_features(seq) for seq in sequences])
        print(f"SentencePiece feature matrix shape: {X.shape}")
        return X
    
    def get_learned_patterns(self):
        """
        Extract learned multi-character patterns from vocabulary.
        
        Returns:
            List of learned patterns
        """
        if self.sp is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        patterns = []
        for i in range(self.sp.get_piece_size()):
            piece = self.sp.id_to_piece(i)
            # Remove special tokens and spaces
            if piece not in ['<unk>', '<s>', '</s>', '<pad>'] and '▁' not in piece:
                clean_piece = piece.replace(' ', '')
                if clean_piece and len(clean_piece) > 1:
                    patterns.append(clean_piece)
        
        return patterns
