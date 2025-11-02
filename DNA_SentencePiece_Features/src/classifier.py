"""
Classification models for DNA sequence analysis.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, Dict


class DNAClassifier:
    """Wrapper for DNA sequence classification."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            n_estimators: Number of trees in random forest
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"Training Random Forest with {self.n_estimators} estimators...")
        self.model.fit(X_train, y_train)
        print("Training complete!")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return results
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        return self.model.feature_importances_


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    
    return X_train, X_test, y_train, y_test


def compare_models(results_dict: Dict[str, Dict]) -> None:
    """
    Compare multiple model results.
    
    Args:
        results_dict: Dictionary mapping model names to result dictionaries
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for model_name, results in results_dict.items():
        print(f"{model_name:30s} Accuracy: {results['accuracy']:.4f}")
    
    # Calculate improvement
    if len(results_dict) == 2:
        models = list(results_dict.keys())
        acc1 = results_dict[models[0]]['accuracy']
        acc2 = results_dict[models[1]]['accuracy']
        improvement = (acc2 - acc1) * 100
        print(f"\nImprovement: {improvement:+.2f}%")
    
    print("="*60)
