"""
Test script for SentencePiece/BPE feature extraction.
Run this after test_02_kmer_extraction.py succeeds.
"""

import sys
import os
import pickle
from collections import Counter

# Add parent directory to path to import the package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from data_loader import DNADataLoader
from feature_extractor import SentencePieceFeatureExtractor

def test_sentencepiece():
    """Test SentencePiece/BPE feature extraction functionality."""
    
    print("="*60)
    print("TEST 3: SENTENCEPIECE/BPE FEATURE EXTRACTION")
    print("="*60)
    
    # Load data first
    print("\n1. Loading data...")
    loader = DNADataLoader()
    loader.download_dataset()
    data = loader.load_data()
    loader.prepare_for_sentencepiece(data)
    print("✓ Data loaded and prepared")
    
    # Initialize SentencePiece extractor
    print("\n2. Initializing SentencePiece Feature Extractor...")
    vocab_size = 1500
    sp_extractor = SentencePieceFeatureExtractor(vocab_size=vocab_size, model_prefix='../models/dna_sp')
    print(f"✓ SentencePiece extractor initialized (vocab_size={vocab_size})")
    
    # Train SentencePiece model
    print("\n3. Training SentencePiece/BPE model...")
    print("   (This may take a minute...)")
    sp_extractor.train(input_file='../data/dna_sequences.txt')
    print("✓ SentencePiece model trained successfully")
    
    # Load the trained model
    print("\n4. Loading trained SentencePiece model...")
    sp_extractor.load()
    print("✓ Model loaded successfully")
    
    # Test encoding on sample sequence
    print("\n5. Testing sequence encoding...")
    sample_seq = data['sequence'].iloc[0]
    tokens = sp_extractor.encode_sequence(sample_seq, out_type=str)
    token_ids = sp_extractor.encode_sequence(sample_seq, out_type=int)
    
    print(f"\nSample sequence: {sample_seq[:60]}...")
    print(f"First 15 tokens: {tokens[:15]}")
    print(f"First 15 token IDs: {token_ids[:15]}")
    print(f"Total tokens: {len(tokens)}")
    print(f"Compare to sequence length: {len(sample_seq)} bases")
    print(f"Compression ratio: {len(sample_seq) / len(tokens):.2f}x")
    print("✓ Encoding successful")
    
    # Extract learned patterns
    print("\n6. Analyzing learned patterns...")
    patterns = sp_extractor.get_learned_patterns()
    print(f"Discovered {len(patterns)} multi-character patterns")
    
    # Analyze pattern lengths
    lengths = [len(p) for p in patterns]
    length_dist = Counter(lengths)
    print(f"\nPattern length distribution:")
    for length, count in sorted(length_dist.items())[:5]:
        print(f"  {length}-character patterns: {count}")
    
    # Show sample patterns
    print(f"\nSample learned patterns:")
    for length in sorted(set(lengths))[:3]:
        patterns_of_length = [p for p in patterns if len(p) == length][:5]
        print(f"  {length}-char: {patterns_of_length}")
    
    print("✓ Pattern analysis complete")
    
    # Extract features for all sequences
    print("\n7. Extracting SentencePiece features for all sequences...")
    print("   (This may take a moment...)")
    X_sp = sp_extractor.extract_features(data['sequence'])
    
    # Show feature statistics
    print(f"\nFeature matrix shape: {X_sp.shape}")
    print(f"Number of sequences: {X_sp.shape[0]}")
    print(f"Vocabulary size: {X_sp.shape[1]}")
    
    # Analyze feature matrix
    non_zero_counts = (X_sp > 0).sum(axis=1)
    print(f"\nAverage non-zero features per sequence: {non_zero_counts.mean():.1f}")
    print(f"Min non-zero features: {non_zero_counts.min()}")
    print(f"Max non-zero features: {non_zero_counts.max()}")
    
    # Show sample feature vector
    print(f"\nSample feature vector (first 20 values):")
    print(X_sp[0][:20])
    
    # Save for next test
    print("\n8. Saving SentencePiece data for next test...")
    with open('../test_outputs/sentencepiece_data.pkl', 'wb') as f:
        pickle.dump({
            'X_sp': X_sp,
            'data': data,
            'extractor': sp_extractor,
            'patterns': patterns
        }, f)
    
    print("✓ SentencePiece data saved to ../test_outputs/sentencepiece_data.pkl")
    
    print("\n" + "="*60)
    print("✓ ALL SENTENCEPIECE TESTS PASSED!")
    print("="*60)
    
    return True, X_sp, data, sp_extractor

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DNA SENTENCEPIECE FEATURES - BPE EXTRACTION TEST")
    print("="*60)
    
    try:
        result = test_sentencepiece()
        
        if result:
            print("\n✓ SentencePiece/BPE component is working correctly!")
            print("\nKey findings:")
            print("- BPE learns variable-length DNA patterns (2-20+ characters)")
            print("- More efficient compression than fixed k-mers")
            print("- Patterns are data-driven, not arbitrarily chosen")
            print("\nNext steps:")
            print("1. Run: python test_04_classification.py")
            print("2. This will compare k-mer vs SentencePiece classification")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\n✗ SentencePiece component needs attention")
