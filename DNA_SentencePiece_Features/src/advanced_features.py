"""
Advanced NLP feature extraction for BPE tokens.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from typing import List, Tuple


class TokenNgramExtractor:
    """Extract n-gram features from BPE token sequences."""
    
    def __init__(self, ngram_range: Tuple[int, int] = (1, 3), max_features: int = 2000):
        """
        Initialize n-gram extractor.
        
        Args:
            ngram_range: Range of n-grams (min_n, max_n)
            max_features: Maximum number of features
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=r'(?u)\b\w+\b'  # Match tokens
        )
    
    def extract_token_sequences(self, sequences: pd.Series, sp_model) -> List[str]:
        """
        Convert DNA sequences to token strings.
        
        Args:
            sequences: DNA sequences
            sp_model: Trained SentencePiece model
            
        Returns:
            List of token strings (space-separated)
        """
        token_sequences = []
        for seq in sequences:
            tokens = sp_model.encode(seq, out_type=str)
            token_sequences.append(' '.join(tokens))
        return token_sequences
    
    def fit_transform(self, token_sequences: List[str]) -> np.ndarray:
        """
        Extract n-gram features.
        
        Args:
            token_sequences: List of token strings
            
        Returns:
            Feature matrix
        """
        X = self.vectorizer.fit_transform(token_sequences).toarray()
        print(f"Token n-gram features: {X.shape}")
        print(f"N-gram range: {self.ngram_range}")
        return X
    
    def get_feature_names(self):
        """Get n-gram feature names."""
        return self.vectorizer.get_feature_names_out()


class TfidfTokenExtractor:
    """TF-IDF weighting for BPE tokens."""
    
    def __init__(self, max_features: int = 1500):
        """
        Initialize TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r'(?u)\b\w+\b'
        )
    
    def fit_transform(self, token_sequences: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features.
        
        Args:
            token_sequences: List of token strings
            
        Returns:
            Feature matrix
        """
        X = self.vectorizer.fit_transform(token_sequences).toarray()
        print(f"TF-IDF features: {X.shape}")
        return X
    
    def get_feature_names(self):
        """Get feature names."""
        return self.vectorizer.get_feature_names_out()


class TokenEmbeddingExtractor:
    """Learn embeddings for BPE tokens (Word2Vec style)."""
    
    def __init__(self, embedding_dim: int = 50, window: int = 5):
        """
        Initialize embedding extractor.
        
        Args:
            embedding_dim: Dimension of embeddings
            window: Context window size
        """
        self.embedding_dim = embedding_dim
        self.window = window
        self.embeddings = None
        self.token_to_id = None
    
    def train(self, token_sequences: List[List[str]]):
        """
        Train token embeddings using Word2Vec approach.
        
        Args:
            token_sequences: List of token lists
        """
        from gensim.models import Word2Vec
        
        print(f"Training token embeddings (dim={self.embedding_dim})...")
        model = Word2Vec(
            sentences=token_sequences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=2,
            workers=4,
            epochs=20
        )
        
        # Store embeddings
        self.token_to_id = {token: i for i, token in enumerate(model.wv.index_to_key)}
        self.embeddings = model.wv.vectors
        
        print(f"Learned embeddings for {len(self.token_to_id)} tokens")
    
    def extract_features(self, token_sequences: List[List[str]]) -> np.ndarray:
        """
        Convert sequences to average embedding vectors.
        
        Args:
            token_sequences: List of token lists
            
        Returns:
            Feature matrix (avg embeddings per sequence)
        """
        X = []
        for tokens in token_sequences:
            # Get embeddings for tokens in sequence
            token_embeds = []
            for token in tokens:
                if token in self.token_to_id:
                    idx = self.token_to_id[token]
                    token_embeds.append(self.embeddings[idx])
            
            # Average embeddings
            if token_embeds:
                avg_embed = np.mean(token_embeds, axis=0)
            else:
                avg_embed = np.zeros(self.embedding_dim)
            
            X.append(avg_embed)
        
        X = np.array(X)
        print(f"Token embedding features: {X.shape}")
        return X


class SequenceCNNFeatures:
    """Extract features using 1D CNN on token sequences."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, max_seq_len: int = 50):
        """
        Initialize CNN feature extractor.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.model = None
    
    def build_model(self, num_classes: int):
        """
        Build CNN model.
        
        Args:
            num_classes: Number of output classes
        """
        from tensorflow import keras
        from tensorflow.keras import layers
        
        inputs = layers.Input(shape=(self.max_seq_len,))
        
        # Embedding layer
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        
        # Multiple CNN layers with different kernel sizes
        conv_blocks = []
        for kernel_size in [3, 4, 5]:
            conv = layers.Conv1D(128, kernel_size, activation='relu')(x)
            conv = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        
        # Concatenate
        x = layers.Concatenate()(conv_blocks)
        
        # Dense layers
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output for feature extraction (before final classification)
        features = layers.Dense(64, activation='relu', name='features')(x)
        
        # Classification output
        outputs = layers.Dense(num_classes, activation='softmax')(features)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.feature_model = keras.Model(inputs=inputs, outputs=features)
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_sequences(self, token_sequences: List[List[int]]) -> np.ndarray:
        """
        Pad/truncate sequences to fixed length.
        
        Args:
            token_sequences: List of token ID sequences
            
        Returns:
            Padded array
        """
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return pad_sequences(
            token_sequences,
            maxlen=self.max_seq_len,
            padding='post',
            truncating='post'
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, epochs: int = 10):
        """
        Train CNN model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
        """
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract CNN features (before classification layer).
        
        Args:
            X: Input sequences
            
        Returns:
            Feature vectors
        """
        return self.feature_model.predict(X, verbose=0)
