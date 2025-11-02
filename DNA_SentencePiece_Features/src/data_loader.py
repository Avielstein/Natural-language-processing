"""
Data loading and preprocessing for DNA sequences.
"""
import pandas as pd
import kagglehub
from pathlib import Path
from typing import Tuple


class DNADataLoader:
    """Class for loading and preparing DNA sequence datasets."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing the dataset (optional, will be set after download)
        """
        self.data_dir = Path(data_dir) if data_dir else None
    
    def download_dataset(self, dataset_id: str = 'miadul/dna-classification-dataset'):
        """
        Download dataset from Kaggle using kagglehub.
        
        Args:
            dataset_id: Kaggle dataset ID (format: 'username/dataset-name')
        
        Returns:
            Path to downloaded dataset
        """
        print(f"Downloading dataset: {dataset_id}...")
        path = kagglehub.dataset_download(dataset_id)
        self.data_dir = Path(path)
        print(f"Dataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        return path
    
    def load_data(self) -> pd.DataFrame:
        """
        Load DNA sequence data from CSV file.
        
        Returns:
            DataFrame with sequences and labels
        """
        # Load the CSV dataset
        csv_file = self.data_dir / 'synthetic_dna_dataset.csv'
        all_data = pd.read_csv(csv_file)
        
        # Rename columns to match expected format
        all_data = all_data.rename(columns={
            'Sequence': 'sequence',
            'Class_Label': 'class',
            'Disease_Risk': 'disease_risk'
        })
        
        # Add species column (same as class for this dataset)
        all_data['species'] = all_data['class'].str.lower()
        
        # Keep only essential columns
        all_data = all_data[['sequence', 'class', 'species', 'disease_risk']]
        
        print(f"Loaded {len(all_data)} sequences")
        print(f"\nClass distribution:")
        print(all_data['class'].value_counts())
        print(f"\nDisease risk distribution:")
        print(all_data['disease_risk'].value_counts())
        print(f"\nFirst sequence sample: {all_data['sequence'].iloc[0][:80]}...")
        print(f"Sequence length: {len(all_data['sequence'].iloc[0])} bases")
        
        return all_data
    
    def prepare_for_sentencepiece(self, data: pd.DataFrame, output_file: str = 'data/dna_sequences.txt'):
        """
        Prepare sequences for SentencePiece training.
        
        Args:
            data: DataFrame containing sequences
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for seq in data['sequence']:
                # Write raw DNA sequences without spacing
                # BPE will learn to split these into meaningful substrings
                f.write(seq + '\n')
        
        print(f"Prepared {len(data)} sequences for SentencePiece training")
        print(f"Saved to: {output_path}")
