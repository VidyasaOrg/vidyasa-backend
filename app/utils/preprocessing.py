import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def tokenize_words(text: str, remove_punctuation: bool = True) -> list[str]:
    """
    Tokenize input text using NLTK, convert to lowercase, and optionally remove punctuation.
    """
    # Convert text to lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())

    if remove_punctuation:
        # Remove punctuation and filter out empty tokens
        tokens = [word for word in (w.translate(str.maketrans('', '', string.punctuation)) for w in tokens) if word]

    return tokens

def preprocess_text(tokens: list[str], is_stem: bool = True, remove_stop_words: bool = True) -> list[str]:
    """
    Do stemming on an input array of tokens, returns an array of stemmed tokens.
    """
    if is_stem & remove_stop_words:
        processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    elif is_stem:
        processed_tokens = [stemmer.stem(word) for word in tokens]
    elif remove_stop_words:
        processed_tokens = [word for word in tokens if word not in stop_words]
    else:
        processed_tokens = tokens

    return processed_tokens
