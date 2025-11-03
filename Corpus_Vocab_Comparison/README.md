# Corpus Vocabulary Comparison

**Comparing BPE/SentencePiece Vocabularies Across Different Text Corpora**

---

## Overview

This project provides tools to train BPE (Byte Pair Encoding) models on different text corpora and compare the learned vocabularies. By comparing tokens learned from standard NLP datasets versus domain-specific text (e.g., scientific PDFs), we can understand how vocabulary patterns differ across domains.

## Features

- **Multiple Corpus Loaders**: Support for well-known NLP datasets (Reuters, Brown, etc.)
- **PDF Text Extraction**: Extract and preprocess text from scientific PDFs
- **BPE Training**: Train separate SentencePiece models on each corpus
- **Vocabulary Comparison**: Analyze differences in learned tokens
- **Visualization**: Generate comparative plots and statistics

## Use Cases

1. **Domain Adaptation**: Understand vocabulary differences when moving from general to domain-specific text
2. **Tokenizer Analysis**: Compare compression patterns across different text types
3. **Transfer Learning**: Identify shared vs. unique tokens for multi-domain models
4. **Text Characteristics**: Quantify linguistic differences between corpora

## Project Structure

```
Corpus_Vocab_Comparison/
├── README.md
├── src/
│   ├── __init__.py
│   ├── corpus_loaders.py      # Load NLP datasets and PDFs
│   ├── vocab_trainer.py        # Train BPE models
│   ├── vocab_comparator.py     # Compare vocabularies
│   └── visualizer.py           # Generate plots
├── demos/
│   └── compare_corpora.py      # Main demo script
├── data/
│   ├── nlp_datasets/           # Downloaded NLP corpora
│   └── pdfs/                   # Scientific PDFs
├── models/
│   ├── nlp_model/              # Trained on NLP datasets
│   └── scientific_model/       # Trained on PDFs
└── results/
    └── comparison_plots/       # Visualization outputs
```

## Quick Start

```python
from src.corpus_loaders import NLPCorpusLoader, PDFCorpusLoader
from src.vocab_trainer import VocabTrainer
from src.vocab_comparator import VocabComparator

# Load corpora
nlp_loader = NLPCorpusLoader()
nlp_text = nlp_loader.load_reuters()

pdf_loader = PDFCorpusLoader()
pdf_text = pdf_loader.load_from_directory('data/pdfs/')

# Train BPE models
trainer = VocabTrainer(vocab_size=5000)
nlp_model = trainer.train(nlp_text, 'models/nlp_model')
pdf_model = trainer.train(pdf_text, 'models/scientific_model')

# Compare vocabularies
comparator = VocabComparator(nlp_model, pdf_model)
comparison = comparator.compare()
comparator.visualize(output_dir='results/')
```

## Installation

```bash
# Required packages
pip install sentencepiece nltk PyPDF2 pdfplumber matplotlib seaborn pandas numpy
```

## Comparison Metrics

- **Unique Tokens**: Tokens appearing in only one corpus
- **Shared Tokens**: Common tokens across both corpora
- **Token Length Distribution**: Character length patterns
- **Frequency Differences**: How token frequencies differ
- **Semantic Clustering**: Group similar tokens using embeddings
- **Compression Ratios**: How efficiently each corpus is tokenized

---

*Part of the Natural-language-processing toolkit*
