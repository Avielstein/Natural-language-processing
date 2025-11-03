# Stop Word Coverage in SentencePiece Vocabularies: A Comparative Analysis

**Investigating how different text corpora affect stop word representation in learned vocabularies**

---

## Summary

This study investigates the representation and coverage of English stop words across three distinct SentencePiece-trained vocabularies derived from diverse text corpora: Brown (balanced multi-genre), Reuters (financial news), and Webtext (informal web content). Using a reference set of 178 common English stop words, we analyzed both the proportion of vocabulary space allocated to stop words and the coverage of unique stop word forms.

### Key Findings

| Metric | Brown | Reuters | Webtext | Average |
|--------|-------|---------|---------|---------|
| Vocabulary Allocation | 3.20% | 3.50% | 4.15% | 3.62% |
| Stop Word Coverage | 81.5% | 71.9% | 80.3% | 77.9% |
| Unique Stop Words | 145/178 | 128/178 | 143/178 | 139/178 |
| Coverage Rank | 1st | 3rd | 2nd | - |

**Main Finding**: While stop words comprise only 3-4% of vocabulary tokens across all corpora, coverage varies significantly (71.9%-81.5%), reflecting the linguistic diversity of each source corpus. Balanced corpora (Brown) capture 10% more stop words than domain-specific corpora (Reuters), despite identical vocabulary sizes.

---

## Research Questions

1. What percentage of vocabulary space is dedicated to stop word tokens?
2. How many unique stop words from a standard reference set appear in each vocabulary?
3. How do different source corpora (balanced, news, web) affect stop word representation?

---

## Methodology

### Data Sources

Three SentencePiece vocabularies, each with 10,000 tokens:

| Corpus | Type | Size | Characteristics |
|--------|------|------|-----------------|
| Brown | Balanced | ~1M words | Fiction, news, academic, diverse genres |
| Reuters | News | ~10M words | Financial news, formal business language |
| Webtext | Web | Variable | Blogs, forums, conversational style |

All vocabularies were trained using SentencePiece with unigram language model.

### Stop Word Reference Set

Reference set of 178 English stop words including:
- Function words: articles (a, an, the), prepositions (of, in, at)
- Pronouns: personal (I, you, he), possessive (my, your)
- Auxiliary verbs: be-forms, have-forms, modals (can, will)
- Contractions: negative (don't, can't), auxiliary (I'm, you're)
- Common adverbs: very, just, now, then

### Normalization Procedures

- Removed SentencePiece space marker (▁)
- Case normalization to lowercase
- Apostrophe normalization: (', ', `) → standard apostrophe
- Subword matching for fragmented tokens

### Metrics

```
Vocabulary Percentage = (stop_word_tokens / total_vocab_size) × 100
Coverage = (unique_matched_stopwords / 178) × 100
```

---

## Results

### Vocabulary Allocation

Despite corpus differences, stop words consistently occupy 3-4% of vocabulary space:

| Corpus | Stop Word Tokens | Percentage |
|--------|------------------|------------|
| Brown | 320 | 3.20% |
| Reuters | 350 | 3.50% |
| Webtext | 415 | 4.15% |

This stability suggests stop words form a functional layer independent of domain.

### Coverage Analysis

Coverage varied significantly (10% range) based on corpus characteristics:

| Corpus | Matched | Coverage | Interpretation |
|--------|---------|----------|----------------|
| Brown | 145/178 | 81.5% | Diverse genres → broad coverage |
| Reuters | 128/178 | 71.9% | Formal news → missing contractions |
| Webtext | 143/178 | 80.3% | Informal → high but gaps remain |

### Visualizations

#### Stop Word Coverage Comparison

![Coverage Comparison](results/stopword_analysis/stopword_coverage.png)

Brown corpus captures 81.5% of reference stop words, significantly outperforming specialized corpora.

#### Allocation vs Coverage Analysis

![Allocation vs Coverage](results/stopword_analysis/stopword_comparison.png)

Left panel shows similar allocation (~3-4%) across corpora. Right panel shows significant coverage differences (72-82%).

#### Coverage Distribution

![Coverage Distribution](results/stopword_analysis/stopword_coverage_pie.png)

Visual breakdown showing percentage of captured vs missing stop words per corpus.

#### Complete Analysis Dashboard

![Dashboard](results/stopword_analysis/stopword_dashboard.png)

Comprehensive view of all metrics across all three corpora.

### Missing Stop Words

Uncovered stop words (21-28% depending on corpus) follow patterns:

- Rare contractions: mightn't, shan't, needn't (archaic or formal)
- Negative forms: wasn't, weren't, hasn't, hadn't (fragmented into subwords)
- Possessive contractions: you've, I'd, they're (less common in formal text)
- Emphatic forms: that'll, should've (primarily spoken language)

### Universal Stop Words

128 stop words appeared in all three vocabularies:
- Core articles: a, an, the
- Common prepositions: of, in, to, for, with, at, from, by
- Basic pronouns: I, you, he, she, it, we, they
- Essential verbs: is, are, was, were, be, been, have, has, had, do, does, did
- Frequent conjunctions: and, but, or, if

---

## Discussion

### Brown Corpus: Highest Coverage (81.5%)

The Brown corpus's balanced design exposes vocabulary to:
- Multiple genres with diverse linguistic contexts
- Fiction includes conversational language with contractions
- Academic texts include formal pronouns and auxiliary verbs
- News sections bridge formal and informal registers
- Captures rare but valid stop words across genres

### Reuters Corpus: Lowest Coverage (71.9%)

Financial news has restricted register:
- Limited use of contractions in professional writing
- Reduced first/second person pronouns (objective reporting style)
- Domain-specific terminology displaces low-frequency stop words
- Formal style avoids colloquialisms

Missing from Reuters: can't, won't, shouldn't, you've, I'm, etc.

### Webtext Corpus: Intermediate Coverage (80.3%)

Web content shows high coverage due to:
- Conversational style with frequent contractions
- First-person narratives (blogs, forums, social media)
- Informal register approaching spoken language
- High token count from repeated common stop words

Despite informality, 20% of stop words still missing, possibly due to:
- Training data sampling variations
- Regional/dialectal variations
- Archaic forms excluded from modern web text

### SentencePiece Tokenization Effects

Subword tokenization creates patterns:

1. Stop word fragmentation: Common words like "the" remain whole (▁the), rare forms split (should + n't)
2. Multiple representations: Inflates token count while maintaining coverage
3. Frequency-driven splitting: High-frequency stop words treated as atomic units

---

## Implications

### Vocabulary Design

The consistent 3-4% allocation suggests:
- Reserving 300-400 tokens for stop words is sufficient for 10k vocabularies
- Stop words are adequately represented without dominating vocabulary
- Models trained on different corpora share similar functional word distributions

### Corpus Selection

- Domain-specific corpora may lack conversational stop words needed for general tasks
- Balanced corpora recommended for general-purpose models
- Domain-specific models may require stop word augmentation
- High token count doesn't guarantee coverage of unique forms (415 tokens in Webtext < 145 unique forms in Brown)

### NLP Applications

- Stop word removal: Modern approaches should account for fragmented representations
- Named entity recognition: Lower coverage in Reuters may affect entity boundary detection
- Sentiment analysis: Missing contractions (can't, won't) could impact polarity detection
- Vocabulary size: 10k tokens provide ~78% stop word coverage; larger vocabularies (20k-50k) may approach 90%+

---

## Reproducibility

### Running the Analysis

```bash
cd Corpus_Vocab_Comparison

# Analyze stop words
python demos/analyze_stopwords.py

# Generate visualizations  
python demos/visualize_stopwords.py
```

### System Requirements

- Python 3.7 or higher
- Dependencies: sentencepiece, matplotlib, numpy
- Runtime: < 1 minute for analysis + visualization

### Expected Output

```
======================================================================
STOP WORD ANALYSIS
======================================================================

Total stop words in reference list: 178

──────────────────────────────────────────────────────────────────────
Analyzing: Brown
──────────────────────────────────────────────────────────────────────
Total vocabulary size: 10,000
Stop words found: 320
Percentage of vocab: 3.20%
Coverage: 145/178 (81.5%) of reference stop words
...
```

---

## Project Structure

```
Corpus_Vocab_Comparison/
├── README.md                          # Main documentation
├── STOPWORD_ANALYSIS.md              # Full scientific paper
├── USAGE_GUIDE.md                    # Detailed usage instructions
├── demos/
│   ├── analyze_stopwords.py          # Stop word analysis script
│   ├── visualize_stopwords.py        # Generate visualizations
│   └── models/
│       ├── Brown.vocab               # Brown corpus vocabulary
│       ├── Reuters.vocab             # Reuters corpus vocabulary
│       └── Webtext.vocab             # Webtext corpus vocabulary
├── results/
│   └── stopword_analysis/            # Generated visualizations
└── src/
    ├── corpus_loaders.py             # Load various corpora
    ├── vocab_trainer.py              # Train SentencePiece models
    ├── vocab_comparator.py           # Compare vocabularies
    └── visualizer.py                 # Visualization utilities
```

---

## Limitations

### Reference Set Constraints

- Limited to 178 stop words; larger lists exist (300-400 words)
- Based on written English; spoken language stop words underrepresented
- No distinction between content-bearing and purely functional usage

### Corpus Characteristics

- Single snapshot of each corpus type
- Training data size variations not controlled
- Temporal effects not examined (language change over time)

### Tokenization Parameters

- Fixed vocabulary size (10k) may not reflect optimal settings
- Unigram LM only; BPE or WordPiece might show different patterns
- Character coverage and byte fallback effects not analyzed

---

## Conclusions

Key findings:

1. **Stable allocation**: Stop words consistently occupy 3-4% of vocabulary space across diverse corpora

2. **Variable coverage**: Coverage ranges from 72-82%, with balanced corpora (Brown) significantly outperforming domain-specific corpora (Reuters)

3. **Corpus effects dominate**: Source corpus characteristics predict stop word coverage better than tokenization parameters

4. **Practical implications**: 
   - Select training corpora based on target application's linguistic requirements
   - Balanced corpora recommended for general-purpose models
   - Domain-specific models may require stop word augmentation

---

## Future Research Directions

- Vocabulary size scaling: How does coverage change with 5k, 20k, 50k vocabularies?
- Multilingual analysis: Stop word patterns in other languages
- Frequency weighting: Analyzing stop word usage frequency, not just presence
- Downstream task impact: Correlation between coverage and task performance
- Temporal dynamics: How stop word usage evolves over time in web corpora
- Different tokenization algorithms: BPE, WordPiece comparison

---

## References

1. Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent approach to subword tokenization. *Proceedings of EMNLP 2018*, 66-71.

2. Kudo, T. (2018). Subword regularization: Improving neural network translation models with multiple subword candidates. *Proceedings of ACL 2018*, 66-75.

3. Francis, W. N., & Kučera, H. (1979). *Brown Corpus Manual*. Brown University.

4. Lewis, D. D., Yang, Y., Rose, T. G., & Li, F. (2004). RCV1: A new benchmark collection for text categorization research. *Journal of Machine Learning Research*, 5, 361-397.

5. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL 2016*, 1715-1725.

---

## Citation

```bibtex
@article{stopword_coverage_2025,
  title={Stop Word Coverage in SentencePiece Vocabularies: A Comparative Analysis},
  author={Corpus Vocabulary Comparison Project},
  year={2025},
  url={https://github.com/Avielstein/Natural-language-processing}
}
```

---

**Additional Documentation**: [Full Scientific Paper](STOPWORD_ANALYSIS.md) | [Usage Guide](USAGE_GUIDE.md)
