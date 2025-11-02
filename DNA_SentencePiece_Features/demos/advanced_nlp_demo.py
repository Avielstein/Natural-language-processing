"""
Advanced NLP Techniques Demo: Compare different approaches for BPE token features
"""
import sys
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import from src directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from advanced_features import (
    TokenNgramExtractor,
    TfidfTokenExtractor,
    TokenEmbeddingExtractor,
    SequenceCNNFeatures
)

print("="*80)
print("ADVANCED NLP FOR DNA CLASSIFICATION: COMPREHENSIVE COMPARISON")
print("="*80)

# Load data
print("\n[1] Loading pre-extracted data...")
with open('../test_outputs/sentencepiece_data.pkl', 'rb') as f:
    bpe_data = pickle.load(f)

data = bpe_data['data']
sp_extractor = bpe_data['extractor']
X_baseline = bpe_data['X_sp']
y = data['class']

print(f"    Dataset: {len(data)} sequences, {y.nunique()} classes")
print(f"    Baseline BPE features: {X_baseline.shape}")

# Encode label strings to integers for some models
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train_base, X_test_base, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X_baseline, y, y_encoded, test_size=0.2, random_state=42, stratify=y
)

print(f"    Train: {len(y_train)} | Test: {len(y_test)}")

# Store results
results = {}

# ============================================================================
# Baseline: Bag-of-Tokens (already extracted)
# ============================================================================
print("\n" + "="*80)
print("[BASELINE] Bag-of-Tokens (Token Counts)")
print("="*80)

rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_base, y_train)
y_pred_baseline = rf_baseline.predict(X_test_base)
acc_baseline = accuracy_score(y_test, y_pred_baseline)

print(f"\n✓ Baseline Accuracy: {acc_baseline:.4f} ({acc_baseline*100:.2f}%)")
results['Baseline (Bag-of-Tokens)'] = acc_baseline

# ============================================================================
# Method 1: Token N-grams
# ============================================================================
print("\n" + "="*80)
print("[METHOD 1] Token N-grams (1-3 grams)")
print("="*80)

ngram_extractor = TokenNgramExtractor(ngram_range=(1, 3), max_features=2000)

# Convert sequences to token strings
print("\nConverting sequences to token strings...")
token_sequences = ngram_extractor.extract_token_sequences(data['sequence'], sp_extractor.sp)

# Split using same indices as baseline
train_indices = X_train_base.shape[0]
token_train_seq = token_sequences[:train_indices]
token_test_seq = token_sequences[train_indices:]

X_train_ngram = ngram_extractor.fit_transform(token_train_seq)
X_test_ngram = ngram_extractor.vectorizer.transform(token_test_seq).toarray()

print("\nTraining Random Forest...")
rf_ngram = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_ngram.fit(X_train_ngram, y_train)
y_pred_ngram = rf_ngram.predict(X_test_ngram)
acc_ngram = accuracy_score(y_test, y_pred_ngram)

print(f"\n✓ N-gram Accuracy: {acc_ngram:.4f} ({acc_ngram*100:.2f}%)")
improvement = ((acc_ngram - acc_baseline) / acc_baseline) * 100
print(f"  Improvement over baseline: {improvement:+.1f}%")
results['Token N-grams (1-3)'] = acc_ngram

# Show sample n-grams
print("\nSample learned n-grams:")
feature_names = ngram_extractor.get_feature_names()
for i, name in enumerate(feature_names[:10]):
    print(f"  {i+1}. {name}")

# ============================================================================
# Method 2: TF-IDF Weighting
# ============================================================================
print("\n" + "="*80)
print("[METHOD 2] TF-IDF Weighting")
print("="*80)

tfidf_extractor = TfidfTokenExtractor(max_features=1500)

X_train_tfidf = tfidf_extractor.fit_transform(token_train_seq)
X_test_tfidf = tfidf_extractor.vectorizer.transform(token_test_seq).toarray()

print("\nTraining Random Forest...")
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = rf_tfidf.predict(X_test_tfidf)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)

print(f"\n✓ TF-IDF Accuracy: {acc_tfidf:.4f} ({acc_tfidf*100:.2f}%)")
improvement = ((acc_tfidf - acc_baseline) / acc_baseline) * 100
print(f"  Improvement over baseline: {improvement:+.1f}%")
results['TF-IDF Weighting'] = acc_tfidf

# ============================================================================
# Method 3: Token Embeddings (Word2Vec style)
# ============================================================================
print("\n" + "="*80)
print("[METHOD 3] Token Embeddings (Word2Vec)")
print("="*80)

# Check if gensim is available
try:
    embedding_extractor = TokenEmbeddingExtractor(embedding_dim=50, window=5)
    
    # Convert to token lists
    token_lists = [sp_extractor.sp.encode(seq, out_type=str) for seq in data['sequence']]
    
    # Train embeddings
    embedding_extractor.train(token_lists)
    
    # Extract features
    X_embeddings = embedding_extractor.extract_features(token_lists)
    X_train_emb, X_test_emb = train_test_split(
        X_embeddings, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining Random Forest...")
    rf_emb = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_emb.fit(X_train_emb, y_train)
    y_pred_emb = rf_emb.predict(X_test_emb)
    acc_emb = accuracy_score(y_test, y_pred_emb)
    
    print(f"\n✓ Embedding Accuracy: {acc_emb:.4f} ({acc_emb*100:.2f}%)")
    improvement = ((acc_emb - acc_baseline) / acc_baseline) * 100
    print(f"  Improvement over baseline: {improvement:+.1f}%")
    results['Token Embeddings'] = acc_emb
    
except ImportError:
    print("\n⚠️  Gensim not installed. Skipping token embeddings.")
    print("   Install with: pip install gensim")
    results['Token Embeddings'] = None

# ============================================================================
# Method 4: 1D CNN on Token Sequences
# ============================================================================
print("\n" + "="*80)
print("[METHOD 4] 1D CNN on Token Sequences")
print("="*80)

try:
    import tensorflow as tf
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    
    cnn_extractor = SequenceCNNFeatures(
        vocab_size=sp_extractor.vocab_size,
        embedding_dim=64,
        max_seq_len=50
    )
    
    # Convert to token ID sequences
    token_id_sequences = [sp_extractor.sp.encode(seq, out_type=int) for seq in data['sequence']]
    X_cnn = cnn_extractor.prepare_sequences(token_id_sequences)
    
    # Split
    X_train_cnn, X_test_cnn = train_test_split(
        X_cnn, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train for validation
    X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
        X_train_cnn, y_train_enc, test_size=0.2, random_state=42
    )
    
    # Build and train CNN
    print("\nBuilding CNN model...")
    cnn_extractor.build_model(num_classes=y.nunique())
    
    print("\nTraining CNN (this may take a minute)...")
    cnn_extractor.train(X_train_cnn, y_train_cnn, X_val_cnn, y_val_cnn, epochs=10)
    
    # Get predictions
    y_pred_cnn_probs = cnn_extractor.model.predict(X_test_cnn, verbose=0)
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
    y_pred_cnn = le.inverse_transform(y_pred_cnn)
    
    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    
    print(f"\n✓ CNN Accuracy: {acc_cnn:.4f} ({acc_cnn*100:.2f}%)")
    improvement = ((acc_cnn - acc_baseline) / acc_baseline) * 100
    print(f"  Improvement over baseline: {improvement:+.1f}%")
    results['1D CNN'] = acc_cnn
    
except ImportError:
    print("\n⚠️  TensorFlow not installed. Skipping CNN.")
    print("   Install with: pip install tensorflow")
    results['1D CNN'] = None

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS COMPARISON")
print("="*80)

print("\nAccuracy Summary:")
print("-" * 80)
sorted_results = sorted(results.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)

for method, acc in sorted_results:
    if acc is not None:
        improvement = ((acc - acc_baseline) / acc_baseline) * 100
        status = "🏆" if acc == max([a for a in results.values() if a is not None]) else "  "
        print(f"{status} {method:30s}: {acc*100:6.2f}% ({improvement:+6.1f}%)")
    else:
        print(f"   {method:30s}: SKIPPED (dependency missing)")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

best_method = sorted_results[0][0]
best_acc = sorted_results[0][1]

if best_acc is not None and best_acc > acc_baseline:
    improvement = ((best_acc - acc_baseline) / acc_baseline) * 100
    print(f"\n✓ Best method: {best_method}")
    print(f"✓ Improvement: {improvement:.1f}% over baseline")
    print(f"✓ This shows that treating BPE tokens as 'words' helps!")
else:
    print("\n→ Advanced methods performed similarly to baseline")
    print("→ May need more data or different approaches")

print("\nWhy these techniques help:")
print("  • N-grams: Capture sequential token patterns")
print("  • TF-IDF: Emphasize discriminative tokens")
print("  • Embeddings: Learn token relationships")  
print("  • CNN: Model local sequence patterns")
