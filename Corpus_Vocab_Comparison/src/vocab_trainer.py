"""
Vocabulary Trainer - Train SentencePiece BPE models on different corpora.
"""

import os
import tempfile
import logging
from typing import Optional, Dict
import sentencepiece as spm

logger = logging.getLogger(__name__)


class VocabTrainer:
    """Train and manage SentencePiece BPE models for different corpora."""
    
    def __init__(self, 
                 vocab_size: int = 5000,
                 model_type: str = 'bpe',
                 character_coverage: float = 1.0,
                 max_sentence_length: int = 16384):
        """
        Initialize vocabulary trainer.
        
        Args:
            vocab_size: Target vocabulary size
            model_type: 'bpe', 'unigram', 'char', or 'word'
            character_coverage: Character coverage (1.0 = all chars)
            max_sentence_length: Maximum sentence length to process
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.max_sentence_length = max_sentence_length
        
        logger.info(f"Initialized VocabTrainer with vocab_size={vocab_size}, "
                   f"model_type={model_type}")
    
    def train(self, 
              text: str, 
              model_prefix: str,
              user_defined_symbols: Optional[list] = None) -> 'TrainedModel':
        """
        Train a SentencePiece model on text.
        
        Args:
            text: Training text corpus
            model_prefix: Prefix for saved model files (e.g., 'models/nlp')
            user_defined_symbols: Additional symbols to preserve
        
        Returns:
            TrainedModel object
        """
        # Create temporary file for training
        # Split text into lines to avoid max_sentence_length issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                        delete=False, encoding='utf-8') as f:
            # Split long text into chunks (sentences/lines)
            # Use multiple strategies to find natural breaks
            import re
            
            # First try splitting by existing newlines
            lines = text.split('\n')
            
            # If lines are too long, split further by sentence boundaries
            processed_lines = []
            for line in lines:
                if len(line) <= 10000:  # Reasonable line length
                    if line.strip():
                        processed_lines.append(line)
                else:
                    # Split long lines by sentence boundaries
                    sentences = re.split(r'(?<=[.!?])\s+', line)
                    for sent in sentences:
                        if sent.strip():
                            processed_lines.append(sent)
            
            # Write lines
            f.write('\n'.join(processed_lines))
            temp_file = f.name
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(model_prefix)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Training parameters
            train_params = {
                'input': temp_file,
                'model_prefix': model_prefix,
                'model_type': self.model_type,
                'vocab_size': self.vocab_size,
                'character_coverage': self.character_coverage,
                'max_sentence_length': self.max_sentence_length,
                'pad_id': 0,
                'unk_id': 1,
                'bos_id': 2,
                'eos_id': 3,
            }
            
            if user_defined_symbols:
                train_params['user_defined_symbols'] = user_defined_symbols
            
            logger.info(f"Training {self.model_type} model with vocab_size={self.vocab_size}...")
            logger.info(f"Output: {model_prefix}.model and {model_prefix}.vocab")
            
            # Train the model
            spm.SentencePieceTrainer.train(**train_params)
            
            logger.info("Training complete!")
            
            # Return trained model object
            return TrainedModel(f"{model_prefix}.model", f"{model_prefix}.vocab")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def train_multiple(self, 
                      corpus_dict: Dict[str, str],
                      output_dir: str = 'models') -> Dict[str, 'TrainedModel']:
        """
        Train multiple models on different corpora.
        
        Args:
            corpus_dict: Dictionary mapping corpus names to text
            output_dir: Directory to save models
        
        Returns:
            Dictionary mapping corpus names to TrainedModel objects
        """
        models = {}
        
        for corpus_name, text in corpus_dict.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model for: {corpus_name}")
            logger.info(f"{'='*60}")
            
            model_prefix = os.path.join(output_dir, corpus_name)
            model = self.train(text, model_prefix)
            models[corpus_name] = model
        
        logger.info(f"\nTrained {len(models)} models successfully!")
        return models


class TrainedModel:
    """Wrapper for a trained SentencePiece model."""
    
    def __init__(self, model_path: str, vocab_path: str):
        """
        Initialize trained model.
        
        Args:
            model_path: Path to .model file
            vocab_path: Path to .vocab file
        """
        self.model_path = model_path
        self.vocab_path = vocab_path
        
        # Load the model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Load vocabulary
        self._load_vocab()
        
        logger.info(f"Loaded model: {model_path}")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
    
    def _load_vocab(self):
        """Load vocabulary from .vocab file."""
        self.vocab = {}
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    token, score = parts
                    self.vocab[token] = float(score)
    
    def encode(self, text: str, out_type: str = 'str') -> list:
        """
        Encode text to tokens.
        
        Args:
            text: Input text
            out_type: 'str' for token strings, 'int' for token IDs
        
        Returns:
            List of tokens
        """
        if out_type == 'str':
            return self.sp.encode(text, out_type=str)
        else:
            return self.sp.encode(text, out_type=int)
    
    def decode(self, tokens: list) -> str:
        """
        Decode tokens back to text.
        
        Args:
            tokens: List of token IDs or strings
        
        Returns:
            Decoded text
        """
        if isinstance(tokens[0], str):
            # Convert string tokens to IDs first
            tokens = [self.sp.piece_to_id(t) for t in tokens]
        return self.sp.decode(tokens)
    
    def get_vocab_dict(self) -> Dict[str, float]:
        """
        Get vocabulary dictionary.
        
        Returns:
            Dictionary mapping tokens to scores
        """
        return self.vocab.copy()
    
    def get_tokens(self) -> list:
        """
        Get list of all tokens.
        
        Returns:
            List of token strings
        """
        return list(self.vocab.keys())
    
    def get_stats(self) -> Dict:
        """
        Get model statistics.
        
        Returns:
            Dictionary of statistics
        """
        tokens = self.get_tokens()
        
        # Calculate token length distribution
        lengths = [len(t) for t in tokens if not t.startswith('▁')]
        
        # Count special tokens
        special_tokens = [t for t in tokens if t.startswith('<') or t == '▁']
        
        return {
            'vocab_size': len(self.vocab),
            'num_tokens': len(tokens),
            'num_special_tokens': len(special_tokens),
            'avg_token_length': sum(lengths) / len(lengths) if lengths else 0,
            'max_token_length': max(lengths) if lengths else 0,
            'min_token_length': min(lengths) if lengths else 0,
        }
    
    def sample_encode(self, text: str, num_samples: int = 5) -> None:
        """
        Print sample encodings.
        
        Args:
            text: Text to encode
            num_samples: Number of sample sentences to show
        """
        sentences = text.split('\n')[:num_samples]
        
        print(f"\n{'='*60}")
        print(f"Sample Encodings from {os.path.basename(self.model_path)}")
        print(f"{'='*60}")
        
        for i, sent in enumerate(sentences, 1):
            if sent.strip():
                tokens = self.encode(sent[:100], out_type='str')
                print(f"\n{i}. Original: {sent[:100]}...")
                print(f"   Tokens ({len(tokens)}): {tokens[:10]}...")


if __name__ == '__main__':
    # Quick test
    print("Testing VocabTrainer...")
    
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Natural language processing is fascinating.
    Machine learning models can learn patterns from data.
    """
    
    trainer = VocabTrainer(vocab_size=100)
    model = trainer.train(sample_text, 'test_model')
    
    print("\nModel stats:")
    for key, value in model.get_stats().items():
        print(f"  {key}: {value}")
    
    print("\nSample tokens:", model.get_tokens()[:20])
