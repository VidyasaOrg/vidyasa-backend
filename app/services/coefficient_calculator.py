import math
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

class CoefficientCalculator:
    """
    Kelas untuk menghitung berbagai koefisien dalam sistem temu balik informasi
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Download jika belum ada
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str, do_stemming: bool = True, 
                       remove_stopwords: bool = True) -> List[str]:
        """
        Preprocessing teks dengan opsi stemming dan stopword removal
        """
        # Lowercase dan tokenisasi
        text = text.lower()
        # Hapus karakter non-alphanumeric kecuali spasi
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        
        # Hapus stopwords jika diminta
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stemming jika diminta
        if do_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def calculate_tf(self, term_freq: int, total_terms: int, method: str = 'raw') -> float:
        """
        Menghitung Term Frequency dengan berbagai metode
        
        Args:
            term_freq: Frekuensi term dalam dokumen
            total_terms: Total term dalam dokumen
            method: 'raw', 'logarithmic', 'binary', 'augmented'
        """
        if method == 'raw':
            return term_freq
        elif method == 'logarithmic':
            return math.log(1 + term_freq) if term_freq > 0 else 0
        elif method == 'binary':
            return 1 if term_freq > 0 else 0
        elif method == 'augmented':
            if total_terms == 0:
                return 0
            max_freq = max(term_freq, 1)  # Avoid division by zero
            return 0.5 + (0.5 * term_freq / max_freq)
        else:
            raise ValueError(f"Unknown TF method: {method}")
    
    def calculate_idf(self, total_docs: int, docs_with_term: int) -> float:
        """
        Menghitung Inverse Document Frequency
        """
        if docs_with_term == 0:
            return 0
        return math.log(total_docs / docs_with_term)
    
    def calculate_tf_idf(self, tf: float, idf: float) -> float:
        """
        Menghitung TF-IDF
        """
        return tf * idf
    
    def cosine_normalization(self, vector: Dict[str, float]) -> Dict[str, float]:
        """
        Normalisasi cosine untuk vektor
        """
        # Hitung magnitude
        magnitude = math.sqrt(sum(weight ** 2 for weight in vector.values()))
        
        if magnitude == 0:
            return vector
        
        # Normalisasi
        normalized = {term: weight / magnitude for term, weight in vector.items()}
        return normalized
    
    def build_term_document_matrix(self, documents: List[str], 
                                 do_stemming: bool = True,
                                 remove_stopwords: bool = True,
                                 tf_method: str = 'raw',
                                 use_idf: bool = True,
                                 use_cosine_norm: bool = False) -> Tuple[Dict, Dict, List]:
        """
        Membangun term-document matrix dengan berbagai opsi pembobotan
        
        Returns:
            - term_doc_matrix: Dict dengan struktur {term: {doc_id: weight}}
            - doc_vectors: Dict dengan struktur {doc_id: {term: weight}}
            - vocabulary: List of all terms
        """
        # Preprocessing semua dokumen
        processed_docs = []
        for i, doc in enumerate(documents):
            tokens = self.preprocess_text(doc, do_stemming, remove_stopwords)
            processed_docs.append((i, tokens))
        
        # Hitung term frequencies untuk setiap dokumen
        doc_term_counts = {}  # {doc_id: {term: count}}
        doc_total_terms = {}  # {doc_id: total_terms}
        term_doc_freq = defaultdict(int)  # {term: jumlah dokumen yang mengandung term}
        
        for doc_id, tokens in processed_docs:
            term_counts = Counter(tokens)
            doc_term_counts[doc_id] = term_counts
            doc_total_terms[doc_id] = len(tokens)
            
            # Hitung document frequency untuk setiap term
            for term in set(tokens):
                term_doc_freq[term] += 1
        
        # Bangun vocabulary
        vocabulary = list(term_doc_freq.keys())
        total_docs = len(documents)
        
        # Bangun matrix
        term_doc_matrix = defaultdict(dict)
        doc_vectors = defaultdict(dict)
        
        for doc_id in range(total_docs):
            doc_vector = {}
            
            for term in vocabulary:
                term_freq = doc_term_counts[doc_id].get(term, 0)
                total_terms_in_doc = doc_total_terms[doc_id]
                
                # Hitung TF
                tf = self.calculate_tf(term_freq, total_terms_in_doc, tf_method)
                
                # Hitung weight
                if use_idf:
                    idf = self.calculate_idf(total_docs, term_doc_freq[term])
                    weight = self.calculate_tf_idf(tf, idf)
                else:
                    weight = tf
                
                doc_vector[term] = weight
                term_doc_matrix[term][doc_id] = weight
            
            # Normalisasi cosine jika diminta
            if use_cosine_norm:
                doc_vector = self.cosine_normalization(doc_vector)
                # Update term_doc_matrix dengan nilai normalized
                for term, norm_weight in doc_vector.items():
                    term_doc_matrix[term][doc_id] = norm_weight
            
            doc_vectors[doc_id] = doc_vector
        
        return dict(term_doc_matrix), dict(doc_vectors), vocabulary
    
    def calculate_cosine_similarity(self, vector1: Dict[str, float], 
                                  vector2: Dict[str, float]) -> float:
        """
        Menghitung cosine similarity antara dua vektor
        """
        # Dapatkan semua terms yang ada di kedua vektor
        all_terms = set(vector1.keys()) | set(vector2.keys())
        
        # Hitung dot product
        dot_product = sum(vector1.get(term, 0) * vector2.get(term, 0) 
                         for term in all_terms)
        
        # Hitung magnitude masing-masing vektor
        magnitude1 = math.sqrt(sum(weight ** 2 for weight in vector1.values()))
        magnitude2 = math.sqrt(sum(weight ** 2 for weight in vector2.values()))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def process_query(self, query: str, term_doc_matrix: Dict,
                     do_stemming: bool = True,
                     remove_stopwords: bool = True,
                     tf_method: str = 'raw',
                     use_idf: bool = True,
                     use_cosine_norm: bool = False) -> Dict[str, float]:
        """
        Memproses query dan mengubahnya menjadi vektor dengan pembobotan yang sama
        """
        # Preprocessing query
        query_tokens = self.preprocess_text(query, do_stemming, remove_stopwords)
        query_term_counts = Counter(query_tokens)
        total_query_terms = len(query_tokens)
        
        # Bangun query vector
        query_vector = {}
        
        for term in term_doc_matrix.keys():
            term_freq = query_term_counts.get(term, 0)
            
            # Hitung TF
            tf = self.calculate_tf(term_freq, total_query_terms, tf_method)
            
            # Hitung weight (sama seperti dokumen)
            if use_idf:
                # Untuk IDF, kita perlu tahu berapa dokumen yang mengandung term ini
                docs_with_term = len([doc_id for doc_id, weight in term_doc_matrix[term].items() if weight > 0])
                total_docs = len(set(doc_id for term_dict in term_doc_matrix.values() 
                                   for doc_id in term_dict.keys()))
                idf = self.calculate_idf(total_docs, docs_with_term)
                weight = self.calculate_tf_idf(tf, idf)
            else:
                weight = tf
            
            if weight > 0:
                query_vector[term] = weight
        
        # Normalisasi cosine jika diminta
        if use_cosine_norm:
            query_vector = self.cosine_normalization(query_vector)
        
        return query_vector
    
    def get_term_info(self, term: str, term_doc_matrix: Dict) -> Dict:
        """
        Mendapatkan informasi detail tentang sebuah term
        """
        if term not in term_doc_matrix:
            return {"error": f"Term '{term}' not found in collection"}
        
        term_info = {
            "term": term,
            "document_frequency": len(term_doc_matrix[term]),
            "documents": []
        }
        
        for doc_id, weight in term_doc_matrix[term].items():
            if weight > 0:
                term_info["documents"].append({
                    "doc_id": doc_id,
                    "weight": weight
                })
        
        # Sort by weight descending
        term_info["documents"].sort(key=lambda x: x["weight"], reverse=True)
        
        return term_info