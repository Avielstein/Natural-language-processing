#!/usr/bin/env python3
"""
Quick test to verify installation and basic functionality.
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    errors = []
    
    try:
        import sentencepiece
        print("✓ sentencepiece")
    except ImportError as e:
        errors.append(f"✗ sentencepiece: {e}")
    
    try:
        import nltk
        print("✓ nltk")
    except ImportError as e:
        errors.append(f"✗ nltk: {e}")
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        errors.append(f"✗ matplotlib: {e}")
    
    try:
        import seaborn
        print("✓ seaborn")
    except ImportError as e:
        errors.append(f"✗ seaborn: {e}")
    
    try:
        import pdfplumber
        print("✓ pdfplumber")
    except ImportError as e:
        print("⚠ pdfplumber (optional)")
    
    try:
        import PyPDF2
        print("✓ PyPDF2")
    except ImportError as e:
        print("⚠ PyPDF2 (optional)")
    
    return errors

def test_modules():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    errors = []
    
    try:
        from src.corpus_loaders import NLPCorpusLoader, PDFCorpusLoader
        print("✓ corpus_loaders")
    except ImportError as e:
        errors.append(f"✗ corpus_loaders: {e}")
    
    try:
        from src.vocab_trainer import VocabTrainer
        print("✓ vocab_trainer")
    except ImportError as e:
        errors.append(f"✗ vocab_trainer: {e}")
    
    try:
        from src.vocab_comparator import VocabComparator
        print("✓ vocab_comparator")
    except ImportError as e:
        errors.append(f"✗ vocab_comparator: {e}")
    
    try:
        from src.visualizer import VocabVisualizer
        print("✓ visualizer")
    except ImportError as e:
        errors.append(f"✗ visualizer: {e}")
    
    return errors

def main():
    print("="*60)
    print("CORPUS VOCABULARY COMPARISON - INSTALLATION TEST")
    print("="*60)
    
    # Test imports
    import_errors = test_imports()
    
    # Test modules
    module_errors = test_modules()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if not import_errors and not module_errors:
        print("\n✓ All tests passed!")
        print("\nYou're ready to run:")
        print("  cd demos")
        print("  python compare_corpora.py")
        return 0
    else:
        print("\n✗ Some tests failed:")
        for error in import_errors + module_errors:
            print(f"  {error}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    sys.exit(main())
