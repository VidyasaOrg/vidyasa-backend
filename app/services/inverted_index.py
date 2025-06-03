import json
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from .coefficient_calculator import CoefficientCalculator

class InvertedIndex:
    """
    Kelas untuk membangun dan mengelola inverted index
    """
    
    def __init__(self):
        self.index = defaultdict(dict)  # {term: {doc_id: {tf, positions, weight}}}
        self.document_info = {}  # {doc_id: {length, unique_terms, content}}
        self.vocabulary = set()
        self.coefficient_calc = CoefficientCalculator()
        self.total_documents = 0
    
    def build_index(self, documents: List[str], 
                   do_stemming: bool = True,
                   remove_stopwords: bool = True,
                   tf_method: str = 'raw',
                   use_idf: bool = True,
                   use_cosine_norm: bool = False) -> None:
        """
        Membangun inverted index dari koleksi dokumen
        """
        self.total_documents = len(documents)
        
        # Preprocessing dan build index
        for doc_id, document in enumerate(documents):
            tokens = self.coefficient_calc.preprocess_text(
                document, do_stemming, remove_stopwords)
            
            # Simpan informasi dokumen
            self.document_info[doc_id] = {
                'length': len(tokens),
                'unique_terms': len(set(tokens)),
                'content': document[:200] + "..." if len(document) > 200 else document
            }
            
            # Build index untuk dokumen ini
            term_positions = defaultdict(list)
            term_frequencies = defaultdict(int)
            
            for position, term in enumerate(tokens):
                term_frequencies[term] += 1
                term_positions[term].append(position)
                self.vocabulary.add(term)
            
            # Hitung weight untuk setiap term dalam dokumen
            for term, freq in term_frequencies.items():
                tf = self.coefficient_calc.calculate_tf(
                    freq, len(tokens), tf_method)
                
                self.index[term][doc_id] = {
                    'tf': tf,
                    'raw_tf': freq,
                    'positions': term_positions[term],
                    'weight': tf  # Will be updated with IDF if needed
                }
        
        # Hitung IDF dan update weights jika diperlukan
        if use_idf:
            self._calculate_idf_weights()
        
        # Normalisasi cosine jika diperlukan
        if use_cosine_norm:
            self._apply_cosine_normalization()
    
    def _calculate_idf_weights(self):
        """
        Menghitung IDF dan update weights dalam index
        """
        for term in self.vocabulary:
            docs_with_term = len(self.index[term])
            idf = self.coefficient_calc.calculate_idf(
                self.total_documents, docs_with_term)
            
            # Update weights
            for doc_id in self.index[term]:
                tf = self.index[term][doc_id]['tf']
                self.index[term][doc_id]['weight'] = tf * idf
                self.index[term][doc_id]['idf'] = idf
    
    def _apply_cosine_normalization(self):
        """
        Apply normalisasi cosine ke semua dokumen
        """
        # Hitung magnitude untuk setiap dokumen
        doc_magnitudes = defaultdict(float)
        
        for term in self.vocabulary:
            for doc_id, term_info in self.index[term].items():
                weight = term_info['weight']
                doc_magnitudes[doc_id] += weight ** 2
        
        # Hitung square root
        for doc_id in doc_magnitudes:
            doc_magnitudes[doc_id] = doc_magnitudes[doc_id] ** 0.5
        
        # Normalisasi weights
        for term in self.vocabulary:
            for doc_id in self.index[term]:
                if doc_magnitudes[doc_id] > 0:
                    original_weight = self.index[term][doc_id]['weight']
                    normalized_weight = original_weight / doc_magnitudes[doc_id]
                    self.index[term][doc_id]['weight'] = normalized_weight
    
    def get_term_info(self, term: str) -> Dict:
        """
        Mendapatkan informasi lengkap tentang sebuah term
        """
        if term not in self.index:
            return {"error": f"Term '{term}' not found in index"}
        
        term_data = self.index[term]
        
        info = {
            "term": term,
            "document_frequency": len(term_data),
            "total_occurrences": sum(data['raw_tf'] for data in term_data.values()),
            "documents": []
        }
        
        for doc_id, data in term_data.items():
            doc_info = {
                "doc_id": doc_id,
                "raw_tf": data['raw_tf'],
                "tf": data['tf'],
                "weight": data['weight'],
                "positions": data['positions'],
                "document_preview": self.document_info[doc_id]['content']
            }
            
            if 'idf' in data:
                doc_info['idf'] = data['idf']
            
            info["documents"].append(doc_info)
        
        # Sort by weight descending
        info["documents"].sort(key=lambda x: x["weight"], reverse=True)
        
        return info
    
    def get_document_vector(self, doc_id: int) -> Dict[str, float]:
        """
        Mendapatkan vektor dokumen (term weights)
        """
        if doc_id not in self.document_info:
            return {}
        
        doc_vector = {}
        for term in self.vocabulary:
            if doc_id in self.index[term]:
                doc_vector[term] = self.index[term][doc_id]['weight']
        
        return doc_vector
    
    def get_document_info(self, doc_id: int) -> Dict:
        """
        Mendapatkan informasi lengkap tentang sebuah dokumen
        """
        if doc_id not in self.document_info:
            return {"error": f"Document {doc_id} not found"}
        
        doc_info = self.document_info[doc_id].copy()
        
        # Tambahkan informasi terms dalam dokumen
        doc_terms = []
        for term in self.vocabulary:
            if doc_id in self.index[term]:
                term_data = self.index[term][doc_id]
                doc_terms.append({