#!/usr/bin/env python3
"""
Analyze the percentage of stop words in the vocabularies.
"""

import os
from pathlib import Path

# Common English stop words
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 
    'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', 
    "wouldn't"
}

def load_vocab(vocab_path):
    """Load vocabulary from a SentencePiece vocab file."""
    vocab_tokens = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Format: token<tab>score
                parts = line.strip().split('\t')
                if parts:
                    token = parts[0]
                    vocab_tokens.append(token)
    return vocab_tokens

def clean_token(token):
    """Clean a SentencePiece token for comparison with stop words."""
    # Remove the leading underscore (▁) that represents a space in SentencePiece
    cleaned = token.replace('▁', '').lower()
    # Remove any remaining special characters
    cleaned = cleaned.strip()
    return cleaned

def normalize_stopword(word):
    """Normalize stop word for better matching (handle apostrophes)."""
    # Replace various apostrophe forms with standard apostrophe
    word = word.replace("'", "'").replace("'", "'").replace("`", "'")
    return word.lower()

def analyze_stopwords(vocab_tokens):
    """Analyze what percentage of vocab are stop words."""
    total_tokens = len(vocab_tokens)
    stopword_count = 0
    found_stopwords = []
    matched_reference_stopwords = set()
    
    # Create normalized version of reference stop words
    normalized_stopwords = {normalize_stopword(sw): sw for sw in STOP_WORDS}
    
    for token in vocab_tokens:
        cleaned = clean_token(token)
        normalized_cleaned = normalize_stopword(cleaned)
        
        if normalized_cleaned in normalized_stopwords:
            stopword_count += 1
            found_stopwords.append(token)
            matched_reference_stopwords.add(normalized_stopwords[normalized_cleaned])
    
    percentage = (stopword_count / total_tokens * 100) if total_tokens > 0 else 0
    coverage = (len(matched_reference_stopwords) / len(STOP_WORDS) * 100) if STOP_WORDS else 0
    
    return {
        'total_tokens': total_tokens,
        'stopword_count': stopword_count,
        'percentage': percentage,
        'stopwords': found_stopwords[:20],  # First 20 for display
        'coverage': coverage,
        'matched_count': len(matched_reference_stopwords),
        'matched_list': sorted(matched_reference_stopwords)
    }

def main():
    """Main function to analyze stop words in all vocabularies."""
    # Get the models directory
    models_dir = Path(__file__).parent / 'models'
    
    print("=" * 70)
    print("STOP WORD ANALYSIS")
    print("=" * 70)
    print(f"\nTotal stop words in reference list: {len(STOP_WORDS)}\n")
    
    # Analyze each vocab file
    vocab_files = ['Brown.vocab', 'Reuters.vocab', 'Webtext.vocab']
    
    results = {}
    for vocab_file in vocab_files:
        vocab_path = models_dir / vocab_file
        if vocab_path.exists():
            corpus_name = vocab_file.replace('.vocab', '')
            print(f"\n{'─' * 70}")
            print(f"Analyzing: {corpus_name}")
            print('─' * 70)
            
            vocab_tokens = load_vocab(vocab_path)
            analysis = analyze_stopwords(vocab_tokens)
            results[corpus_name] = analysis
            
            print(f"Total vocabulary size: {analysis['total_tokens']:,}")
            print(f"Stop words found: {analysis['stopword_count']:,}")
            print(f"Percentage of vocab: {analysis['percentage']:.2f}%")
            print(f"\nCoverage: {analysis['matched_count']}/{len(STOP_WORDS)} "
                  f"({analysis['coverage']:.1f}%) of reference stop words")
            
            if analysis['stopwords']:
                print(f"\nFirst 20 stop words found:")
                for i, word in enumerate(analysis['stopwords'], 1):
                    print(f"  {i:2d}. {word}")
        else:
            print(f"\nWarning: {vocab_path} not found")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Corpus':<15} {'Vocab Size':<12} {'Stop Words':<12} "
          f"{'% of Vocab':<12} {'Coverage':<15}")
    print("-" * 70)
    
    for corpus_name, analysis in sorted(results.items()):
        print(f"{corpus_name:<15} {analysis['total_tokens']:<12,} "
              f"{analysis['stopword_count']:<12,} {analysis['percentage']:<12.2f}% "
              f"{analysis['matched_count']}/{len(STOP_WORDS)} ({analysis['coverage']:.1f}%)")
    
    # Additional insights
    print("\n" + "=" * 70)
    print("INSIGHTS")
    print("=" * 70)
    
    if results:
        avg_percentage = sum(r['percentage'] for r in results.values()) / len(results)
        avg_coverage = sum(r['coverage'] for r in results.values()) / len(results)
        
        print(f"\nAverage stop word percentage across all corpora: {avg_percentage:.2f}%")
        print(f"Average coverage of reference stop words: {avg_coverage:.1f}%")
        
        max_corpus = max(results.items(), key=lambda x: x[1]['percentage'])
        min_corpus = min(results.items(), key=lambda x: x[1]['percentage'])
        
        print(f"\nHighest vocab %: {max_corpus[0]} ({max_corpus[1]['percentage']:.2f}%)")
        print(f"Lowest vocab %: {min_corpus[0]} ({min_corpus[1]['percentage']:.2f}%)")
        
        max_coverage = max(results.items(), key=lambda x: x[1]['coverage'])
        min_coverage = min(results.items(), key=lambda x: x[1]['coverage'])
        
        print(f"\nBest coverage: {max_coverage[0]} ({max_coverage[1]['matched_count']}/{len(STOP_WORDS)} = {max_coverage[1]['coverage']:.1f}%)")
        print(f"Lowest coverage: {min_coverage[0]} ({min_coverage[1]['matched_count']}/{len(STOP_WORDS)} = {min_coverage[1]['coverage']:.1f}%)")
        
        print("\n" + "─" * 70)
        print("NOTES:")
        print("─" * 70)
        print("• '% of Vocab' = how many vocab tokens are stop words")
        print("• 'Coverage' = how many of the 178 reference stop words appear")
        print("• Apostrophes are normalized (don't, don't, don`t all match)")
        print("• SentencePiece tokenization may split words into subword units")

if __name__ == '__main__':
    main()
