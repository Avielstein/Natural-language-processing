"""
Test script for k-mer feature extraction.
Run this after test_01_data_loader.py succeeds.
"""

import sys
import os
import pickle

# Add parent directory to path to import the package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from data_loader import DNADataLoader
from feature_extractor import KmerFeatureExtractor

def test_kmer_extraction():
    """Test k-mer feature extraction functionality."""
    
    print("="*60)
    print("TEST 2: K-MER FEATURE EXTRACTION")
    print("="*60)
    
    # Load data first
    print("\n1. Loading data...")
    loader = DNADataLoader()
    loader.download_dataset()
    data = loader.load_data()
    print("✓ Data loaded")
    
    # Initialize k-mer extractor
    print("\n2. Initializing K-mer Feature Extractor...")
    kmer_size = 6
    max_features = 1500
    kmer_extractor = KmerFeatureExtractor(kmer_size=kmer_size, max_features=max_features)
    print(f"✓ K-mer extractor initialized (k={kmer_size}, max_features={max_features})")
    
    # Extract k-mers from sequences
    print("\n3. Extracting k-mers from sequences...")
    data = kmer_extractor.extract_kmers(data)
    
    # Show sample k-mers
    sample_kmers = data['kmers'].iloc[0]
    print(f"\nSample sequence: {data['sequence'].iloc[0][:60]}...")
    print(f"First 10 k-mers: {sample_kmers[:10]}")
    print(f"Total k-mers in sequence: {len(sample_kmers)}")
    print("✓ K-mers extracted successfully")
    
    # Create feature matrix
    print("\n4. Creating k-mer feature matrix...")
    X_kmer = kmer_extractor.fit_transform(data['kmer_text'])
    
    # Show feature statistics
    print(f"\nFeature matrix shape: {X_kmer.shape}")
    print(f"Number of sequences: {X_kmer.shape[0]}")
    print(f"Number of k-mer features: {X_kmer.shape[1]}")
    
    # Show sample feature names
    feature_names = kmer_extractor.get_feature_names()
    print(f"\nSample k-mer features: {list(feature_names[:10])}")
    
    # Analyze feature matrix
    print("\n5. Analyzing feature matrix...")
    non_zero_counts = (X_kmer > 0).sum(axis=1)
    print(f"Average non-zero features per sequence: {non_zero_counts.mean():.1f}")
    print(f"Min non-zero features: {non_zero_counts.min()}")
    print(f"Max non-zero features: {non_zero_counts.max()}")
    
    # Show sample feature vector
    print(f"\nSample feature vector (first 20 values):")
    print(X_kmer[0][:20])
    
    # Save for next test
    print("\n6. Saving k-mer data for next test...")
    os.makedirs('../test_outputs', exist_ok=True)
    
    with open('../test_outputs/kmer_data.pkl', 'wb') as f:
        pickle.dump({
            'X_kmer': X_kmer,
            'data': data,
            'extractor': kmer_extractor
        }, f)
    
    print("✓ K-mer data saved to ../test_outputs/kmer_data.pkl")
    
    print("\n" + "="*60)
    print("✓ ALL K-MER EXTRACTION TESTS PASSED!")
    print("="*60)
    
    return True, X_kmer, data

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DNA SENTENCEPIECE FEATURES - K-MER EXTRACTION TEST")
    print("="*60)
    
    try:
        result = test_kmer_extraction()
        
        if result:
            print("\n✓ K-mer extraction component is working correctly!")
            print("\nNext steps:")
            print("1. Run: python test_03_sentencepiece.py")
            print("2. This will test SentencePiece/BPE training and feature extraction")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\n✗ K-mer extraction component needs attention")
