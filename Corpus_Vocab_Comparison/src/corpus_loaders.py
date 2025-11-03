"""
Corpus Loaders - Load text from various sources for vocabulary comparison.

Supports:
- Standard NLP datasets (Reuters, Brown, Gutenberg, etc.)
- PDF documents (scientific papers)
- Custom text files
"""

import os
import re
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPCorpusLoader:
    """Load standard NLP datasets using NLTK."""
    
    def __init__(self):
        """Initialize NLP corpus loader."""
        try:
            import nltk
            self.nltk = nltk
        except ImportError:
            raise ImportError("NLTK is required. Install with: pip install nltk")
    
    def _download_if_needed(self, corpus_name: str) -> None:
        """Download NLTK corpus if not already present."""
        try:
            self.nltk.data.find(f'corpora/{corpus_name}')
        except LookupError:
            logger.info(f"Downloading {corpus_name} corpus...")
            self.nltk.download(corpus_name, quiet=True)
    
    def load_reuters(self, max_docs: Optional[int] = None) -> str:
        """
        Load Reuters-21578 news corpus.
        
        Args:
            max_docs: Maximum number of documents to load (None = all)
        
        Returns:
            Combined text from all documents
        """
        self._download_if_needed('reuters')
        from nltk.corpus import reuters
        
        fileids = reuters.fileids()[:max_docs] if max_docs else reuters.fileids()
        logger.info(f"Loading {len(fileids)} Reuters documents...")
        
        texts = [reuters.raw(fileid) for fileid in fileids]
        return '\n\n'.join(texts)
    
    def load_brown(self, categories: Optional[List[str]] = None) -> str:
        """
        Load Brown Corpus (diverse text genres).
        
        Args:
            categories: Specific categories to load (None = all)
        
        Returns:
            Combined text from selected categories
        """
        self._download_if_needed('brown')
        from nltk.corpus import brown
        
        if categories:
            logger.info(f"Loading Brown corpus categories: {categories}")
            texts = [brown.raw(fileid) for cat in categories 
                    for fileid in brown.fileids(categories=cat)]
        else:
            logger.info("Loading entire Brown corpus...")
            texts = [brown.raw(fileid) for fileid in brown.fileids()]
        
        return '\n\n'.join(texts)
    
    def load_gutenberg(self, fileids: Optional[List[str]] = None) -> str:
        """
        Load Project Gutenberg texts (classic literature).
        
        Args:
            fileids: Specific texts to load (None = all)
        
        Returns:
            Combined text from selected books
        """
        self._download_if_needed('gutenberg')
        from nltk.corpus import gutenberg
        
        if fileids:
            logger.info(f"Loading Gutenberg texts: {fileids}")
            texts = [gutenberg.raw(fid) for fid in fileids]
        else:
            logger.info("Loading all Gutenberg texts...")
            texts = [gutenberg.raw(fid) for fid in gutenberg.fileids()]
        
        return '\n\n'.join(texts)
    
    def load_webtext(self) -> str:
        """
        Load web text corpus (informal internet text).
        
        Returns:
            Combined web text
        """
        self._download_if_needed('webtext')
        from nltk.corpus import webtext
        
        logger.info("Loading webtext corpus...")
        texts = [webtext.raw(fileid) for fileid in webtext.fileids()]
        return '\n\n'.join(texts)
    
    def get_corpus_info(self) -> Dict[str, str]:
        """
        Get information about available corpora.
        
        Returns:
            Dictionary of corpus names and descriptions
        """
        return {
            'reuters': 'Reuters-21578 news articles (business/finance)',
            'brown': 'Brown Corpus - 500 samples from diverse genres (1961)',
            'gutenberg': 'Project Gutenberg classic literature',
            'webtext': 'Informal web text (forums, reviews)',
        }


class PDFCorpusLoader:
    """Load and extract text from PDF documents."""
    
    def __init__(self, use_pdfplumber: bool = True):
        """
        Initialize PDF loader.
        
        Args:
            use_pdfplumber: Use pdfplumber (better) vs PyPDF2 (fallback)
        """
        self.use_pdfplumber = use_pdfplumber
        
        if use_pdfplumber:
            try:
                import pdfplumber
                self.pdf_lib = pdfplumber
                self.lib_name = 'pdfplumber'
            except ImportError:
                logger.warning("pdfplumber not found, falling back to PyPDF2")
                self._init_pypdf2()
        else:
            self._init_pypdf2()
    
    def _init_pypdf2(self):
        """Initialize PyPDF2 as fallback."""
        try:
            import PyPDF2
            self.pdf_lib = PyPDF2
            self.lib_name = 'PyPDF2'
        except ImportError:
            raise ImportError(
                "PDF library required. Install with: "
                "pip install pdfplumber (recommended) or pip install PyPDF2"
            )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")
        
        try:
            if self.lib_name == 'pdfplumber':
                return self._extract_pdfplumber(pdf_path)
            else:
                return self._extract_pypdf2(pdf_path)
        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")
            return ""
    
    def _extract_pdfplumber(self, pdf_path: str) -> str:
        """Extract using pdfplumber (better quality)."""
        text_parts = []
        with self.pdf_lib.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return '\n\n'.join(text_parts)
    
    def _extract_pypdf2(self, pdf_path: str) -> str:
        """Extract using PyPDF2 (fallback)."""
        text_parts = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = self.pdf_lib.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return '\n\n'.join(text_parts)
    
    def load_from_directory(self, pdf_dir: str, 
                          pattern: str = '*.pdf',
                          max_files: Optional[int] = None) -> str:
        """
        Load all PDFs from a directory.
        
        Args:
            pdf_dir: Directory containing PDFs
            pattern: File pattern to match (e.g., '*.pdf')
            max_files: Maximum number of files to process
        
        Returns:
            Combined text from all PDFs
        """
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")
        
        import glob
        pdf_files = glob.glob(os.path.join(pdf_dir, pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return ""
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_text = []
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                all_text.append(text)
        
        logger.info(f"Successfully extracted text from {len(all_text)} PDFs")
        return '\n\n'.join(all_text)
    
    def load_from_file_list(self, pdf_paths: List[str]) -> str:
        """
        Load specific PDF files.
        
        Args:
            pdf_paths: List of PDF file paths
        
        Returns:
            Combined text from all PDFs
        """
        all_text = []
        for pdf_path in pdf_paths:
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                all_text.append(text)
        
        return '\n\n'.join(all_text)


class TextFileLoader:
    """Load text from plain text files."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize text file loader.
        
        Args:
            encoding: Text file encoding (default: utf-8)
        """
        self.encoding = encoding
    
    def load_file(self, filepath: str) -> str:
        """
        Load a single text file.
        
        Args:
            filepath: Path to text file
        
        Returns:
            File contents
        """
        logger.info(f"Loading: {os.path.basename(filepath)}")
        with open(filepath, 'r', encoding=self.encoding) as f:
            return f.read()
    
    def load_directory(self, directory: str, 
                      extensions: List[str] = ['.txt'],
                      max_files: Optional[int] = None) -> str:
        """
        Load all text files from directory.
        
        Args:
            directory: Directory path
            extensions: File extensions to include
            max_files: Maximum files to load
        
        Returns:
            Combined text from all files
        """
        import glob
        
        all_text = []
        for ext in extensions:
            pattern = os.path.join(directory, f'*{ext}')
            files = glob.glob(pattern)
            
            if max_files:
                files = files[:max_files]
            
            for filepath in files:
                text = self.load_file(filepath)
                all_text.append(text)
        
        logger.info(f"Loaded {len(all_text)} text files")
        return '\n\n'.join(all_text)


def clean_text(text: str, 
               remove_urls: bool = True,
               remove_emails: bool = True,
               normalize_whitespace: bool = True) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text
        remove_urls: Remove URLs
        remove_emails: Remove email addresses
        normalize_whitespace: Normalize whitespace
    
    Returns:
        Cleaned text
    """
    if remove_urls:
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    if remove_emails:
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
    
    if normalize_whitespace:
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text


if __name__ == '__main__':
    # Quick test
    print("Testing NLP Corpus Loader...")
    nlp_loader = NLPCorpusLoader()
    print("\nAvailable corpora:")
    for name, desc in nlp_loader.get_corpus_info().items():
        print(f"  {name}: {desc}")
    
    # Test loading a small corpus
    print("\nLoading sample from webtext...")
    sample = nlp_loader.load_webtext()
    print(f"Loaded {len(sample)} characters")
    print(f"First 200 chars: {sample[:200]}...")
