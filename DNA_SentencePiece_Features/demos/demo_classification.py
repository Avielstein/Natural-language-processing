"""
Quick demo: Compare k-mer vs BPE features for DNA classification
"""
import sys
import os
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("="*70)
print("DNA CLASSIFICATION: K-MER vs BPE COMPARISON")
print("="*70)

# Load pre-extracted features
print("\n1. Loading k-mer features...")
with open('../test_outputs/kmer_data.pkl', 'rb') as f:
    kmer_data = pickle.load(f)
X_kmer = kmer_data['X_kmer']
data = kmer_data['data']
y = data['class']

print(f"   K-mer features: {X_kmer.shape}")
print(f"   Classes: {y.nunique()} ({', '.join(y.unique())})")

print("\n2. Loading BPE features...")
with open('../test_outputs/sentencepiece_data.pkl', 'rb') as f:
    bpe_data = pickle.load(f)
X_bpe = bpe_data['X_sp']

print(f"   BPE features: {X_bpe.shape}")

# Split data
print("\n3. Splitting data (80% train, 20% test)...")
X_kmer_train, X_kmer_test, y_train, y_test = train_test_split(
    X_kmer, y, test_size=0.2, random_state=42, stratify=y
)
X_bpe_train, X_bpe_test, _, _ = train_test_split(
    X_bpe, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(y_train)}")
print(f"   Test samples: {len(y_test)}")

# Train k-mer classifier
print("\n4. Training Random Forest with K-mer features...")
rf_kmer = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_kmer.fit(X_kmer_train, y_train)
y_pred_kmer = rf_kmer.predict(X_kmer_test)
acc_kmer = accuracy_score(y_test, y_pred_kmer)

print(f"   K-mer Accuracy: {acc_kmer:.4f} ({acc_kmer*100:.2f}%)")

# Train BPE classifier
print("\n5. Training Random Forest with BPE features...")
rf_bpe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_bpe.fit(X_bpe_train, y_train)
y_pred_bpe = rf_bpe.predict(X_bpe_test)
acc_bpe = accuracy_score(y_test, y_pred_bpe)

print(f"   BPE Accuracy: {acc_bpe:.4f} ({acc_bpe*100:.2f}%)")

# Compare
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print(f"\nK-mer (fixed 6-mers):  {acc_kmer*100:.2f}%")
print(f"BPE (learned patterns): {acc_bpe*100:.2f}%")

if acc_bpe > acc_kmer:
    improvement = ((acc_bpe - acc_kmer) / acc_kmer) * 100
    print(f"\n✓ BPE is {improvement:.1f}% better than k-mers!")
elif acc_kmer > acc_bpe:
    improvement = ((acc_kmer - acc_bpe) / acc_bpe) * 100
    print(f"\n✓ K-mers are {improvement:.1f}% better than BPE!")
else:
    print("\n→ Both approaches perform equally!")

print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORTS")
print("="*70)

print("\nK-mer Classification Report:")
print("-" * 70)
print(classification_report(y_test, y_pred_kmer, zero_division=0))

print("\nBPE Classification Report:")
print("-" * 70)
print(classification_report(y_test, y_pred_bpe, zero_division=0))

# Feature importance analysis
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*70)

# K-mer top features
kmer_importance = rf_kmer.feature_importances_
kmer_feature_names = kmer_data['extractor'].get_feature_names()
top_kmer_idx = kmer_importance.argsort()[-10:][::-1]

print("\nTop K-mer Features:")
for i, idx in enumerate(top_kmer_idx, 1):
    print(f"  {i}. {kmer_feature_names[idx]:8s} (importance: {kmer_importance[idx]:.4f})")

# BPE top features
bpe_importance = rf_bpe.feature_importances_
bpe_patterns = bpe_data['patterns']
top_bpe_idx = bpe_importance.argsort()[-10:][::-1]

print("\nTop BPE Patterns:")
for i, idx in enumerate(top_bpe_idx, 1):
    # Find the actual pattern for this feature index
    if idx < len(bpe_patterns):
        pattern = bpe_patterns[idx]
    else:
        pattern = f"feature_{idx}"
    print(f"  {i}. {pattern:10s} (importance: {bpe_importance[idx]:.4f})")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("\n✓ Both k-mer and BPE features work for DNA classification")
print("✓ BPE learns data-driven patterns (2-6+ chars)")
print("✓ K-mers use fixed-length patterns (6 chars)")
print("✓ Feature importance reveals discriminative sequences")
print("✓ Both approaches are valid - choice depends on use case")
