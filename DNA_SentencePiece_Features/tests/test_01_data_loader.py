"""
Test script for data loading functionality.
Run this first to validate data loading works correctly.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from data_loader import DNADataLoader

def test_data_loader():
    """Test the data loader functionality."""
    
    print("="*60)
    print("TEST 1: DATA LOADER")
    print("="*60)
    
    # Initialize data loader
    print("\n1. Initializing DNADataLoader...")
    loader = DNADataLoader()
    print("✓ DNADataLoader initialized successfully")
    
    # Download dataset (you'll need to enter Kaggle credentials)
    print("\n2. Downloading dataset from Kaggle...")
    print("Note: You will be prompted to enter your Kaggle credentials")
    print("If you don't have a Kaggle account, create one at https://www.kaggle.com")
    
    try:
        loader.download_dataset()
        print("✓ Dataset downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nTo fix this:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to Account settings and create an API token")
        print("3. Run this script again")
        return False
    
    # Load data
    print("\n3. Loading DNA sequences...")
    try:
        data = loader.load_data()
        print("✓ Data loaded successfully")
        
        # Validate data
        print("\n4. Validating loaded data...")
        assert 'sequence' in data.columns, "Missing 'sequence' column"
        assert 'class' in data.columns, "Missing 'class' column"
        assert 'species' in data.columns, "Missing 'species' column"
        assert len(data) > 0, "No data loaded"
        print("✓ Data validation passed")
        
        # Display sample
        print("\n5. Sample data:")
        print("-" * 60)
        print(f"First sequence: {data['sequence'].iloc[0][:80]}...")
        print(f"Sequence length: {len(data['sequence'].iloc[0])} bases")
        print(f"Class: {data['class'].iloc[0]}")
        print(f"Species: {data['species'].iloc[0]}")
        
        # Prepare for SentencePiece
        print("\n6. Preparing sequences for SentencePiece training...")
        loader.prepare_for_sentencepiece(data)
        
        # Check if file was created
        if os.path.exists('data/dna_sequences.txt'):
            print("✓ SentencePiece training file created successfully")
            
            # Show sample from file
            with open('data/dna_sequences.txt', 'r') as f:
                first_line = f.readline().strip()
                print(f"\nSample from training file (first 100 chars):")
                print(first_line[:100] + "...")
        else:
            print("✗ Training file not created")
            return False
        
        print("\n" + "="*60)
        print("✓ ALL DATA LOADER TESTS PASSED!")
        print("="*60)
        
        return True, data
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DNA SENTENCEPIECE FEATURES - DATA LOADER TEST")
    print("="*60)
    
    result = test_data_loader()
    
    if result:
        print("\n✓ Data loading component is working correctly!")
        print("\nNext steps:")
        print("1. Run: python test_02_kmer_extraction.py")
        print("2. This will test k-mer feature extraction")
    else:
        print("\n✗ Data loading component needs attention")
        print("Please fix the issues above before proceeding")
