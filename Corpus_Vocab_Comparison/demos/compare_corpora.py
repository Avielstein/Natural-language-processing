#!/usr/bin/env python3
"""
Main Demo: Compare Vocabularies Across Different Corpora

This script demonstrates the complete workflow:
1. Load corpora (NLP datasets and scientific PDFs)
2. Train BPE models on each corpus
3. Compare the learned vocabularies
4. Generate visualizations
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.corpus_loaders import NLPCorpusLoader, PDFCorpusLoader, clean_text
from src.vocab_trainer import VocabTrainer
from src.vocab_comparator import VocabComparator
from src.visualizer import VocabVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete vocabulary comparison workflow."""
    
    print("\n" + "="*70)
    print("CORPUS VOCABULARY COMPARISON DEMO")
    print("="*70)
    
    # Configuration
    VOCAB_SIZE = 10000
    
    # Step 1: Load NLP Corpora
    print("\n" + "-"*70)
    print("STEP 1: Loading Three NLP Library Corpora")
    print("-"*70)
    
    corpora = {}
    nlp_loader = NLPCorpusLoader()
    
    # Load Reuters (business/finance news)
    logger.info("Loading Reuters corpus...")
    try:
        reuters_text = nlp_loader.load_reuters(max_docs=500)
        reuters_text = clean_text(reuters_text)
        corpora['Reuters'] = reuters_text
        print(f"✓ Reuters (News): {len(reuters_text):,} characters")
    except Exception as e:
        logger.error(f"Failed to load Reuters: {e}")
        print(f"✗ Failed to load Reuters: {e}")
    
    # Load Brown Corpus (diverse genres)
    logger.info("Loading Brown corpus...")
    try:
        brown_text = nlp_loader.load_brown()
        brown_text = clean_text(brown_text)
        corpora['Brown'] = brown_text
        print(f"✓ Brown (General): {len(brown_text):,} characters")
    except Exception as e:
        logger.error(f"Failed to load Brown: {e}")
        print(f"✗ Failed to load Brown: {e}")
    
    # Load Webtext (informal web text)
    logger.info("Loading Webtext corpus...")
    try:
        webtext_text = nlp_loader.load_webtext()
        webtext_text = clean_text(webtext_text)
        corpora['Webtext'] = webtext_text
        print(f"✓ Webtext (Informal): {len(webtext_text):,} characters")
    except Exception as e:
        logger.error(f"Failed to load Webtext: {e}")
        print(f"✗ Failed to load Webtext: {e}")
    
    if len(corpora) < 2:
        print("\n⚠ Need at least 2 corpora for comparison!")
        print("  Please ensure corpora are loaded successfully.")
        print("\nTroubleshooting:")
        print("  Run: python -c 'import nltk; nltk.download(\"reuters\"); nltk.download(\"brown\"); nltk.download(\"webtext\")'")
        return
    
    print(f"\n✓ Successfully loaded {len(corpora)} corpora for comparison")
    
    # Step 2: Train BPE Models
    print("\n" + "-"*70)
    print("STEP 2: Training BPE Models")
    print("-"*70)
    
    trainer = VocabTrainer(vocab_size=VOCAB_SIZE, model_type='bpe')
    models = trainer.train_multiple(corpora, output_dir='models')
    
    print(f"\n✓ Trained {len(models)} models successfully!")
    
    # Step 3: Compare Vocabularies
    print("\n" + "-"*70)
    print("STEP 3: Comparing Vocabularies")
    print("-"*70)
    
    model_list = list(models.values())
    comparator = VocabComparator(*model_list)
    comparator.set_model_names(list(corpora.keys()))
    
    comparison = comparator.compare_all()
    comparator.print_summary(comparison)
    
    # Step 4: Analyze Compression
    print("\n" + "-"*70)
    print("STEP 4: Analyzing Compression Efficiency")
    print("-"*70)
    
    # Get sample texts for compression analysis
    test_texts = {}
    for corpus_name, text in corpora.items():
        # Take first 1000 characters as sample
        sample = text[:1000]
        test_texts[f"{corpus_name}_sample"] = sample
    
    compression_results = comparator.get_compression_ratios(test_texts)
    
    print("\nCompression Ratios (chars per token):")
    for text_name, results in compression_results.items():
        print(f"\n  {text_name}:")
        for model_name, stats in results.items():
            print(f"    {model_name}: {stats['compression_ratio']:.2f} "
                  f"({stats['num_tokens']} tokens)")
    
    # Step 5: Domain Specificity Analysis
    print("\n" + "-"*70)
    print("STEP 5: Analyzing Domain Specificity")
    print("-"*70)
    
    domain_analysis = comparator.analyze_domain_specificity(sample_size=100)
    
    for corpus_name, analysis in domain_analysis.items():
        print(f"\n  {corpus_name}:")
        print(f"    Unique tokens: {analysis['total_unique']:,} "
              f"({analysis['unique_percentage']:.1f}% of vocabulary)")
        print(f"    Token categories:")
        for category, count in analysis['categories'].items():
            if count > 0:
                print(f"      {category}: {count}")
        if analysis['sample_unique_tokens']:
            sample = analysis['sample_unique_tokens'][:10]
            print(f"    Sample unique tokens: {sample}")
    
    # Step 6: Create Visualizations
    print("\n" + "-"*70)
    print("STEP 6: Creating Visualizations")
    print("-"*70)
    
    visualizer = VocabVisualizer(output_dir='results')
    plots = visualizer.create_full_report(comparison, compression_results)
    
    print(f"\n✓ Created {len(plots)} visualization plots:")
    for plot in plots:
        print(f"  • {os.path.basename(plot)}")
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📊 Results saved to:")
    print(f"  • Models: models/")
    print(f"  • Visualizations: results/")
    print(f"\n📈 Key Findings:")
    print(f"  • Trained {len(models)} BPE models with vocab_size={VOCAB_SIZE}")
    print(f"  • Shared tokens: {len(comparison['shared_tokens']):,}")
    
    for name in corpora.keys():
        unique = len(comparison['unique_tokens'][name])
        print(f"  • {name} unique tokens: {unique:,}")
    
    if len(models) == 2:
        names = list(corpora.keys())
        pair_key = f"{names[0]} vs {names[1]}"
        jaccard = comparison['pairwise_overlaps'][pair_key]['jaccard_similarity']
        print(f"  • Vocabulary overlap (Jaccard): {jaccard:.3f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        sys.exit(1)
