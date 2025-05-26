import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(tokens):

    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(processed_tokens)