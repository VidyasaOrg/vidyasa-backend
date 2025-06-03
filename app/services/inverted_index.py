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
                    "term": term,
                    "raw_tf": term_data['raw_tf'],
                    "tf": term_data['tf'],
                    "weight": term_data['weight'],
                    "positions": term_data['positions']
                })
        
        # Sort terms by weight descending
        doc_terms.sort(key=lambda x: x["weight"], reverse=True)
        doc_info["terms"] = doc_terms
        doc_info["total_terms"] = len(doc_terms)
        
        return doc_info
    
    def search_terms(self, terms: List[str]) -> Dict:
        """
        Mencari dokumen yang mengandung terms tertentu
        """
        if not terms:
            return {"error": "No search terms provided"}
        
        # Preprocessing terms
        processed_terms = []
        for term in terms:
            processed = self.coefficient_calc.preprocess_text(
                term, do_stemming=True, remove_stopwords=True)
            processed_terms.extend(processed)
        
        # Cari dokumen yang mengandung terms
        document_scores = defaultdict(float)
        term_matches = defaultdict(list)
        
        for term in processed_terms:
            if term in self.index:
                for doc_id, term_data in self.index[term].items():
                    document_scores[doc_id] += term_data['weight']
                    term_matches[doc_id].append({
                        "term": term,
                        "weight": term_data['weight'],
                        "positions": term_data['positions']
                    })
        
        # Format hasil
        results = []
        for doc_id, score in document_scores.items():
            results.append({
                "doc_id": doc_id,
                "score": score,
                "document_preview": self.document_info[doc_id]['content'],
                "matched_terms": term_matches[doc_id]
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "query_terms": processed_terms,
            "total_results": len(results),
            "results": results
        }
    
    def get_vocabulary_stats(self) -> Dict:
        """
        Mendapatkan statistik vocabulary
        """
        term_frequencies = {}
        document_frequencies = {}
        
        for term in self.vocabulary:
            total_freq = sum(data['raw_tf'] for data in self.index[term].values())
            doc_freq = len(self.index[term])
            
            term_frequencies[term] = total_freq
            document_frequencies[term] = doc_freq
        
        # Most frequent terms
        most_frequent = sorted(term_frequencies.items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        
        # Most distributed terms (appearing in most documents)
        most_distributed = sorted(document_frequencies.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "vocabulary_size": len(self.vocabulary),
            "total_documents": self.total_documents,
            "most_frequent_terms": most_frequent,
            "most_distributed_terms": most_distributed,
            "average_document_length": sum(info['length'] for info in self.document_info.values()) / self.total_documents if self.total_documents > 0 else 0,
            "average_unique_terms": sum(info['unique_terms'] for info in self.document_info.values()) / self.total_documents if self.total_documents > 0 else 0
        }
    
    def calculate_similarity(self, doc_id1: int, doc_id2: int) -> float:
        """
        Menghitung cosine similarity antara dua dokumen
        """
        if doc_id1 not in self.document_info or doc_id2 not in self.document_info:
            return 0.0
        
        vector1 = self.get_document_vector(doc_id1)
        vector2 = self.get_document_vector(doc_id2)
        
        # Hitung dot product
        dot_product = 0.0
        for term in self.vocabulary:
            weight1 = vector1.get(term, 0.0)
            weight2 = vector2.get(term, 0.0)
            dot_product += weight1 * weight2
        
        # Hitung magnitudes
        magnitude1 = sum(weight ** 2 for weight in vector1.values()) ** 0.5
        magnitude2 = sum(weight ** 2 for weight in vector2.values()) ** 0.5
        
        # Hitung cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def find_similar_documents(self, doc_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Mencari dokumen yang mirip dengan dokumen tertentu
        """
        if doc_id not in self.document_info:
            return []
        
        similarities = []
        for other_doc_id in self.document_info:
            if other_doc_id != doc_id:
                similarity = self.calculate_similarity(doc_id, other_doc_id)
                similarities.append((other_doc_id, similarity))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_index(self, filepath: str) -> None:
        """
        Menyimpan index ke file
        """
        index_data = {
            'index': dict(self.index),
            'document_info': self.document_info,
            'vocabulary': list(self.vocabulary),
            'total_documents': self.total_documents
        }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)
    
    def load_index(self, filepath: str) -> None:
        """
        Memuat index dari file
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
        
        self.index = defaultdict(dict, index_data['index'])
        self.document_info = index_data['document_info']
        self.vocabulary = set(index_data['vocabulary'])
        self.total_documents = index_data['total_documents']
    
    def get_index_summary(self) -> Dict:
        """
        Mendapatkan ringkasan lengkap dari index
        """
        return {
            "total_documents": self.total_documents,
            "vocabulary_size": len(self.vocabulary),
            "total_term_document_pairs": sum(len(doc_dict) for doc_dict in self.index.values()),
            "average_terms_per_document": sum(info['unique_terms'] for info in self.document_info.values()) / self.total_documents if self.total_documents > 0 else 0,
            "vocabulary_stats": self.get_vocabulary_stats()
        }