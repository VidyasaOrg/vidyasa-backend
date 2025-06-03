import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import threading
from typing import List, Optional


class TextPreprocessor:
    """
    A class for text preprocessing operations including tokenization, 
    stemming, and stopword removal with lazy initialization of NLTK resources.
    Thread-safe singleton implementation.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, language: str = "english"):
        """Thread-safe singleton implementation"""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super(TextPreprocessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, language: str = "english"):
        """
        Initialize the preprocessor (only once due to singleton).
        
        Args:
            language (str): Language for stopwords. Defaults to "english".
        """
        # Prevent re-initialization of singleton
        if TextPreprocessor._initialized:
            return
            
        with TextPreprocessor._lock:
            if TextPreprocessor._initialized:
                return
                
            self.language = language
            self._stemmer: Optional[PorterStemmer] = None
            self._stop_words: Optional[set] = None
            self._nltk_initialized = False
            self._nltk_lock = threading.Lock()  # Separate lock for NLTK operations
            
            TextPreprocessor._initialized = True
    
    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded (thread-safe)"""
        if self._nltk_initialized:
            return
            
        with self._nltk_lock:
            # Double-checked locking for NLTK initialization
            if self._nltk_initialized:
                return
                
            required_data = [
                ('tokenizers/punkt', 'punkt'),
                ('tokenizers/punkt_tab', 'punkt_tab'),  # For newer NLTK versions
                ('corpora/stopwords', 'stopwords'),
            ]
            
            for data_path, data_name in required_data:
                try:
                    nltk.data.find(data_path)
                except LookupError:
                    print(f"NLTK '{data_name}' not found, downloading...")
                    try:
                        nltk.download(data_name, quiet=True)
                        print(f"NLTK '{data_name}' downloaded successfully.")
                    except Exception as e:
                        print(f"Failed to download NLTK '{data_name}': {e}")
            
            self._nltk_initialized = True
    
    @property
    def stemmer(self) -> PorterStemmer:
        """Lazy initialization of stemmer (thread-safe)"""
        if self._stemmer is None:
            with self._nltk_lock:
                if self._stemmer is None:
                    self._ensure_nltk_data()
                    self._stemmer = PorterStemmer()
        return self._stemmer
    
    @property
    def stop_words(self) -> set:
        """Lazy initialization of stopwords (thread-safe)"""
        if self._stop_words is None:
            with self._nltk_lock:
                if self._stop_words is None:
                    self._ensure_nltk_data()
                    self._stop_words = set(stopwords.words(self.language))
        return self._stop_words
    
    def tokenize_words(self, text: str, remove_punctuation: bool = True) -> List[str]:
        """
        Tokenize input text using NLTK, convert to lowercase, and optionally remove punctuation.
        
        Args:
            text (str): Input text to tokenize
            remove_punctuation (bool): Whether to remove punctuation. Defaults to True.
            
        Returns:
            List[str]: List of tokens
        """
        self._ensure_nltk_data()
        
        # Convert text to lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())

        if remove_punctuation:
            # Remove punctuation and filter out empty tokens
            tokens = [
                word for word in (
                    w.translate(str.maketrans('', '', string.punctuation)) 
                    for w in tokens
                ) if word
            ]

        return tokens

    def preprocess_text(
        self, 
        tokens: List[str], 
        is_stem: bool = True, 
        is_stop_words_removal: bool = True
    ) -> List[str]:
        """
        Optionally remove stopwords and/or stem tokens.
        
        Args:
            tokens (List[str]): List of tokens to process
            is_stem (bool): Whether to apply stemming. Defaults to True.
            is_stop_words_removal (bool): Whether to remove stopwords. Defaults to True.
            
        Returns:
            List[str]: Processed tokens
        """
        processed = tokens.copy()  # Don't modify original list
        
        if is_stop_words_removal:
            processed = [word for word in processed if word not in self.stop_words]
        
        if is_stem:
            processed = [self.stemmer.stem(word) for word in processed]
        
        return processed
    
    def tokenize_and_preprocess(
        self, 
        text: str, 
        remove_punctuation: bool = True,
        is_stem: bool = True, 
        is_stop_words_removal: bool = True
    ) -> List[str]:
        """
        Complete preprocessing pipeline: tokenize and preprocess in one step.
        
        Args:
            text (str): Input text
            remove_punctuation (bool): Whether to remove punctuation. Defaults to True.
            is_stem (bool): Whether to apply stemming. Defaults to True.
            is_stop_words_removal (bool): Whether to remove stopwords. Defaults to True.
            
        Returns:
            List[str]: Fully processed tokens
        """
        tokens = self.tokenize_words(text, remove_punctuation)
        return self.preprocess_text(tokens, is_stem, is_stop_words_removal)
    
    @classmethod
    def reset_singleton(cls):
        """Reset singleton (useful for testing)"""
        with cls._lock:
            cls._instance = None
            cls._initialized = False


# Convenience functions that use the singleton
def get_preprocessor() -> TextPreprocessor:
    """Get the singleton preprocessor instance"""
    return TextPreprocessor()

def tokenize_words(text: str, remove_punctuation: bool = True) -> List[str]:
    """
    Backward-compatible tokenization function
    
    Tokenize input text using NLTK, convert to lowercase, and optionally remove punctuation.
    Args:
        text (str): Input text to tokenize
        remove_punctuation (bool): Whether to remove punctuation. Defaults to True.
    Returns:
        List[str]: List of tokens
    """
    return get_preprocessor().tokenize_words(text, remove_punctuation)

def preprocess_text(tokens: List[str], is_stem: bool = True, is_stop_words_removal: bool = True) -> List[str]:
    """
    Backward-compatible preprocessing function
    
    Optionally remove stopwords and/or stem tokens.
    Args:
        tokens (List[str]): List of tokens to process
        is_stem (bool): Whether to apply stemming. Defaults to True.
        is_stop_words_removal (bool): Whether to remove stopwords. Defaults to True.
    Returns:
        List[str]: Processed tokens
    """
    return get_preprocessor().preprocess_text(tokens, is_stem, is_stop_words_removal)

def tokenize_and_preprocess(
    text: str, 
    remove_punctuation: bool = True,
    is_stem: bool = True, 
    is_stop_words_removal: bool = True
) -> List[str]:
    """
    Backward-compatible tokenization and preprocessing function
    
    Complete preprocessing pipeline: tokenize and preprocess in one step.
    
    Args:
        text (str): Input text
        remove_punctuation (bool): Whether to remove punctuation. Defaults to True.
        is_stem (bool): Whether to apply stemming. Defaults to True.
        is_stop_words_removal (bool): Whether to remove stopwords. Defaults to True.
        
    Returns:
        List[str]: Fully processed tokens
    """
    return get_preprocessor().tokenize_and_preprocess(text, remove_punctuation, is_stem, is_stop_words_removal)