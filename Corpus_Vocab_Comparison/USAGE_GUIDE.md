# Corpus Vocabulary Comparison - Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
cd Corpus_Vocab_Comparison
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
cd demos
python compare_corpora.py
```

This will:
- Load the Reuters corpus (standard NLP dataset)
- Load scientific PDFs from `../system 2 security/` directory
- Train BPE models on each corpus (vocab_size=5000)
- Compare the learned vocabularies
- Generate visualizations in `results/` directory

## What You'll Get

### Trained Models
- `models/Reuters.model` and `models/Reuters.vocab`
- `models/Scientific.model` and `models/Scientific.vocab`

### Visualizations
- `vocab_sizes.png` - Compare vocabulary sizes
- `token_lengths.png` - Token length distributions
- `unique_tokens.png` - Unique vs shared tokens
- `jaccard_similarity.png` - Vocabulary overlap heatmap
- `overlap_venn.png` - Venn diagram (for 2 corpora)
- `compression_ratios.png` - Compression efficiency

### Analysis Output
- Comprehensive comparison statistics
- Domain specificity analysis
- Compression ratio comparisons

## Customization

### Use Different Corpora

Edit `demos/compare_corpora.py`:

```python
# Use different NLP corpora
brown_text = nlp_loader.load_brown(['news', 'fiction'])
gutenberg_text = nlp_loader.load_gutenberg()

# Load from different PDF directory
pdf_text = pdf_loader.load_from_directory('/path/to/your/pdfs')

# Add to corpora dict
corpora = {
    'News': brown_text,
    'Classic_Literature': gutenberg_text,
    'Scientific': pdf_text
}
```

### Adjust Vocabulary Size

```python
trainer = VocabTrainer(vocab_size=10000)  # Default is 5000
```

### Change Model Type

```python
# Options: 'bpe', 'unigram', 'char', 'word'
trainer = VocabTrainer(vocab_size=5000, model_type='unigram')
```

## Python API Usage

### Basic Usage

```python
from src.corpus_loaders import NLPCorpusLoader, PDFCorpusLoader
from src.vocab_trainer import VocabTrainer
from src.vocab_comparator import VocabComparator
from src.visualizer import VocabVisualizer

# Load corpora
nlp_loader = NLPCorpusLoader()
text1 = nlp_loader.load_reuters(max_docs=500)

pdf_loader = PDFCorpusLoader()
text2 = pdf_loader.load_from_directory('path/to/pdfs')

# Train models
trainer = VocabTrainer(vocab_size=5000)
model1 = trainer.train(text1, 'models/corpus1')
model2 = trainer.train(text2, 'models/corpus2')

# Compare
comparator = VocabComparator(model1, model2)
comparator.set_model_names(['Corpus1', 'Corpus2'])
comparison = comparator.compare_all()
comparator.print_summary()

# Visualize
visualizer = VocabVisualizer(output_dir='results')
visualizer.create_full_report(comparison)
```

### Advanced Analysis

```python
# Get unique tokens for each corpus
unique_1 = comparator.get_unique_tokens(0)
unique_2 = comparator.get_unique_tokens(1)

# Get shared tokens
shared = comparator.get_shared_tokens()

# Analyze compression on test texts
test_texts = {
    'sample1': 'Some text here...',
    'sample2': 'More text...'
}
compression = comparator.get_compression_ratios(test_texts)

# Domain specificity analysis
domain_analysis = comparator.analyze_domain_specificity(sample_size=100)
```

## Available Corpora

### NLTK Datasets
- **Reuters**: Business/finance news articles
- **Brown**: Diverse genres (news, fiction, etc.)
- **Gutenberg**: Classic literature
- **Webtext**: Informal web text

### Custom Sources
- **PDF files**: Scientific papers, technical documents
- **Text files**: Any plain text corpus
- **Custom loaders**: Easily extend for new sources

## Interpretation Guide

### Jaccard Similarity
- 1.0 = Identical vocabularies
- 0.5 = 50% overlap
- 0.0 = No overlap

High overlap suggests:
- Similar text domains
- Generic vocabulary
- Limited domain specialization

Low overlap suggests:
- Different text domains
- Specialized vocabularies
- Strong domain-specific terms

### Token Length
- Longer tokens = Better compression for that domain
- Scientific text often has longer tokens (technical terms)
- General text has shorter, more common tokens

### Compression Ratio
- Higher ratio = More efficient tokenization
- Model trained on similar domain compresses better
- Cross-domain compression reveals vocabulary mismatch

## Tips

1. **Start Small**: Use `max_docs` parameter to test with subset
2. **Clean Text**: Use `clean_text()` to remove noise
3. **Vocab Size**: 5000 is good default, increase for larger corpora
4. **Multiple Corpora**: Compare 3+ corpora for richer analysis
5. **Save Results**: Models and vocabs can be reused

## Troubleshooting

### NLTK Download Errors
```python
import nltk
nltk.download('reuters')
nltk.download('brown')
```

### PDF Extraction Issues
- Install pdfplumber: `pip install pdfplumber`
- For scanned PDFs, may need OCR (pytesseract)

### Memory Issues
- Reduce corpus size with `max_docs` parameter
- Train models separately
- Use smaller vocab_size

## Next Steps

1. Experiment with different corpora combinations
2. Try different vocabulary sizes
3. Analyze domain-specific token patterns
4. Use models for downstream NLP tasks
5. Compare with your DNA BPE models!

---

*Part of the Natural-language-processing toolkit*
