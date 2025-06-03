from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import math
import json
from collections import defaultdict, Counter
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class Document:
    """
    Represents a document in the information retrieval system.
    Attributes:
        id (int): Unique identifier for the document.
        title (str): Title of the document.
        author (str): Author of the document.
        content (str): Main content of the document.
        tokens (List[str]): Tokenized content of the document.
        raw_tf (Dict[str, int]): Raw term frequency for terms in the document.
        log_tf (Dict[str, float]): Logarithmic term frequency for terms in the document.
        aug_tf (Dict[str, float]): Augmented term frequency for terms in the document.
        binary_tf (Dict[str, int]): Binary term frequency for terms in the document.
        
    Example JSON input:
    ```
        {
            "id": 1,
            "title": "Information Retrieval",
            "author": "J. Doe",
            "content": "What is information retrieval?",
            "tokenized_content": ["what", "is", "information", "retrieval"],
            "raw_tf": {"what": 1, "is": 1, "information": 1, "retrieval": 1}
        }
    ```
    """
    id: int
    title: str
    author: str
    content: str
    tokens: List[str] = field(default_factory=list)
    raw_tf: Dict[str, int] = field(default_factory=dict)
    log_tf: Dict[str, float] = field(default_factory=dict)
    aug_tf: Dict[str, float] = field(default_factory=dict)
    binary_tf: Dict[str, int] = field(default_factory=dict)
    
    @staticmethod
    def from_json(data: dict) -> "Document":
        """
        Create a Document object from a JSON-like dictionary.

        Args:
            data (dict): Dictionary containing document data.

        Returns:
            Document: An instance of the Document class.
        """
        return Document(
            id=int(data["id"]),
            title=data.get("title", ""),
            author=data.get("author", ""),
            content=data.get("content", ""),
            tokens=data.get("tokenized_content", []),
            raw_tf=data.get("raw_tf", {}),
            log_tf=data.get("log_tf", {}),
            aug_tf=data.get("aug_tf", {}),
            binary_tf=data.get("binary_tf", {}),
        )


@dataclass
class IRData:
    """
    Data class for storing information retrieval (IR) data loaded from a JSON file.
    Enhanced with coefficient calculation and MAP computation capabilities.

    Attributes:
        documents (List[Document]): List of document objects.
        idf (Dict[str, float]): Inverse document frequency values for terms.
        inverse_doc_by_term (Dict[str, List[int]]): Inverted index mapping terms to document indices.
        inverse_doc_by_id (Dict[int, Dict[str, List[int]]]): Inverted index mapping document indices to term positions.

    Example JSON input:
    ```
        {
            "docs": [
                {
                    "id": 1,
                    "title": "Information Retrieval",
                    "author": "J. Doe",
                    "content": "What is information retrieval?",
                    "tokenized_content": ["what", "is", "information", "retrieval"],
                    "raw_tf": {"what": 1, "is": 1, "information": 1, "retrieval": 1}
                }
            ],
            "idf": {"information": 1.0, "retrieval": 1.0},
            "inverted_index_by_term": {"information": [0], "retrieval": [0]},
            "inverted_index_by_doc": {0: {"information": [0,1], "retrieval": [3]}}
        }
    ```
    """

    documents: List[Document] = field(default_factory=list)
    idf: Dict[str, float] = field(default_factory=dict)
    inverse_doc_by_term: Dict[str, List[int]] = field(default_factory=dict)
    inverse_doc_by_id: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)

    @staticmethod
    def from_json(data: dict) -> "IRData":
        """
        Create an IRData instance from a dictionary loaded from JSON.

        Args:
            data (dict): Dictionary containing IR data, typically loaded from a JSON file.

        Returns:
            IRData: An instance of IRData populated with documents and index data.
        """
        documents = []
        for doc in data.get("docs", []):
            documents.append(Document.from_json(doc))
        return IRData(
            documents=documents,
            idf=data.get("idf", {}),
            inverse_doc_by_term=data.get("inverted_index_by_term", {}),
            inverse_doc_by_id=data.get("inverted_index_by_doc", {}),
        )
    
    # === COEFFICIENT CALCULATION METHODS ===
    
    def calculate_tf(self, term: str, doc_id: int, tf_scheme: str = 'raw') -> float:
        """
        Calculate Term Frequency with various schemes
        
        Args:
            term: The term to calculate TF for
            doc_id: Document ID (integer)
            tf_scheme: 'raw', 'logarithmic', 'binary', 'augmented'
            
        Returns:
            float: TF value according to the scheme
        """
        # Find document by ID
        doc = None
        for d in self.documents:
            if d.id == doc_id:
                doc = d
                break
        
        if not doc:
            return 0.0
        
        if tf_scheme == 'raw':
            return float(doc.raw_tf.get(term, 0))
        elif tf_scheme == 'logarithmic':
            if term in doc.log_tf:
                return doc.log_tf[term]
            else:
                raw_tf = doc.raw_tf.get(term, 0)
                return 1 + math.log10(raw_tf) if raw_tf > 0 else 0.0
        elif tf_scheme == 'binary':
            if term in doc.binary_tf:
                return float(doc.binary_tf[term])
            else:
                return 1.0 if doc.raw_tf.get(term, 0) > 0 else 0.0
        elif tf_scheme == 'augmented':
            if term in doc.aug_tf:
                return doc.aug_tf[term]
            else:
                raw_tf = doc.raw_tf.get(term, 0)
                max_tf = max(doc.raw_tf.values()) if doc.raw_tf else 1
                return 0.5 + 0.5 * (raw_tf / max_tf) if max_tf > 0 else 0.0
        else:
            return float(doc.raw_tf.get(term, 0))
    
    def calculate_idf(self, term: str) -> float:
        """
        Calculate Inverse Document Frequency
        
        Args:
            term: The term to calculate IDF for
            
        Returns:
            float: IDF value
        """
        if term in self.idf:
            return self.idf[term]
        
        # Calculate IDF if not precomputed
        df = len(self.inverse_doc_by_term.get(term, []))
        total_docs = len(self.documents)
        
        if df > 0 and total_docs > 0:
            idf_value = math.log10(total_docs / df)
            self.idf[term] = idf_value  # Cache the result
            return idf_value
        return 0.0
    
    def calculate_tf_idf(self, term: str, doc_id: int, tf_scheme: str = 'raw') -> float:
        """
        Calculate TF-IDF weight
        
        Args:
            term: The term
            doc_id: Document ID
            tf_scheme: TF calculation scheme
            
        Returns:
            float: TF-IDF value
        """
        tf = self.calculate_tf(term, doc_id, tf_scheme)
        idf = self.calculate_idf(term)
        return tf * idf
    
    def calculate_document_vector(self, doc_id: int, 
                                weighting_scheme: str = 'tf_idf', 
                                tf_scheme: str = 'raw') -> Dict[str, float]:
        """
        Calculate document vector with specified weighting scheme
        
        Args:
            doc_id: Document ID
            weighting_scheme: 'tf', 'idf', 'tf_idf', 'tf_idf_cosine'
            tf_scheme: TF calculation scheme
            
        Returns:
            Dict[str, float]: Document vector
        """
        # Find document
        doc = None
        for d in self.documents:
            if d.id == doc_id:
                doc = d
                break
        
        if not doc:
            return {}
        
        vector = {}
        
        for term in doc.raw_tf.keys():
            if weighting_scheme == 'tf':
                weight = self.calculate_tf(term, doc_id, tf_scheme)
            elif weighting_scheme == 'idf':
                weight = self.calculate_idf(term)
            elif weighting_scheme in ['tf_idf', 'tf_idf_cosine']:
                weight = self.calculate_tf_idf(term, doc_id, tf_scheme)
            else:
                weight = self.calculate_tf(term, doc_id, tf_scheme)
            
            if weight > 0:
                vector[term] = weight
        
        # Apply cosine normalization if needed
        if weighting_scheme == 'tf_idf_cosine':
            vector = self._cosine_normalize(vector)
            
        return vector
    
    def calculate_query_vector(self, query_terms: List[str], 
                             weighting_scheme: str = 'tf_idf',
                             tf_scheme: str = 'raw') -> Dict[str, float]:
        """
        Calculate query vector
        
        Args:
            query_terms: List of query terms
            weighting_scheme: Weighting scheme
            tf_scheme: TF calculation scheme
            
        Returns:
            Dict[str, float]: Query vector
        """
        query_tf = Counter(query_terms)
        vector = {}
        
        for term, tf in query_tf.items():
            if weighting_scheme == 'tf':
                if tf_scheme == 'logarithmic':
                    weight = 1 + math.log10(tf)
                elif tf_scheme == 'binary':
                    weight = 1.0
                else:  # raw or augmented
                    weight = float(tf)
            elif weighting_scheme == 'idf':
                weight = self.calculate_idf(term)
            elif weighting_scheme in ['tf_idf', 'tf_idf_cosine']:
                tf_weight = 1 + math.log10(tf) if tf_scheme == 'logarithmic' else float(tf)
                if tf_scheme == 'binary':
                    tf_weight = 1.0
                idf_weight = self.calculate_idf(term)
                weight = tf_weight * idf_weight
            else:
                weight = float(tf)
            
            if weight > 0:
                vector[term] = weight
        
        # Apply cosine normalization if needed
        if weighting_scheme == 'tf_idf_cosine':
            vector = self._cosine_normalize(vector)
            
        return vector
    
    def _cosine_normalize(self, vector: Dict[str, float]) -> Dict[str, float]:
        """Apply cosine normalization to vector"""
        magnitude = math.sqrt(sum(weight ** 2 for weight in vector.values()))
        if magnitude == 0:
            return vector
        return {term: weight / magnitude for term, weight in vector.items()}
    
    def calculate_similarity(self, query_vector: Dict[str, float], 
                           doc_vector: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between query and document vectors
        
        Args:
            query_vector: Query vector
            doc_vector: Document vector
            
        Returns:
            float: Cosine similarity score
        """
        # Dot product
        dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) 
                         for term in set(query_vector.keys()) | set(doc_vector.keys()))
        
        # Magnitudes
        query_magnitude = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
        doc_magnitude = math.sqrt(sum(weight ** 2 for weight in doc_vector.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
            
        return dot_product / (query_magnitude * doc_magnitude)
    
    # === RETRIEVAL METHODS ===
    
    def retrieve_documents(self, query_terms: List[str], 
                          weighting_scheme: str = 'tf_idf',
                          tf_scheme: str = 'raw') -> List[Tuple[int, float]]:
        """
        Retrieve documents for a query
        
        Args:
            query_terms: List of query terms
            weighting_scheme: Weighting scheme
            tf_scheme: TF calculation scheme
            
        Returns:
            List[Tuple[int, float]]: List of (doc_id, similarity_score) sorted by score
        """
        query_vector = self.calculate_query_vector(query_terms, weighting_scheme, tf_scheme)
        
        if not query_vector:
            return []
        
        results = []
        for doc in self.documents:
            doc_vector = self.calculate_document_vector(doc.id, weighting_scheme, tf_scheme)
            similarity = self.calculate_similarity(query_vector, doc_vector)
            
            if similarity > 0:
                results.append((doc.id, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    # === MAP CALCULATION METHODS ===
    
    def calculate_precision_at_k(self, retrieved_docs: List[int], 
                               relevant_docs: Set[int], k: int) -> float:
        """Calculate precision@k"""
        if k == 0 or not retrieved_docs:
            return 0.0
            
        top_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_retrieved / len(top_k)
    
    def calculate_average_precision(self, retrieved_docs: List[int], 
                                  relevant_docs: Set[int]) -> float:
        """Calculate Average Precision for one query"""
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
    
    def calculate_map(self, query_results: Dict[str, List[int]], 
                     relevance_judgments: Dict[str, Set[int]]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Mean Average Precision (MAP)
        
        Args:
            query_results: {query_id: [retrieved_doc_ids]}
            relevance_judgments: {query_id: {relevant_doc_ids}}
            
        Returns:
            Tuple[float, Dict[str, float]]: (overall_map, individual_aps)
        """
        individual_aps = {}
        total_ap = 0.0
        valid_queries = 0
        
        for query_id, retrieved_docs in query_results.items():
            relevant_docs = relevance_judgments.get(query_id, set())
            ap = self.calculate_average_precision(retrieved_docs, relevant_docs)
            individual_aps[query_id] = ap
            total_ap += ap
            valid_queries += 1
        
        overall_map = total_ap / valid_queries if valid_queries > 0 else 0.0
        return overall_map, individual_aps
    
    # === INVERTED FILE METHODS ===
    
    def get_inverted_file_for_document(self, doc_id: int) -> Dict[str, Dict]:
        """
        Get inverted file information for a specific document
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict containing term frequencies and positions
        """
        result = {}
        
        # Find document
        doc = None
        for d in self.documents:
            if d.id == doc_id:
                doc = d
                break
        
        if not doc:
            return result
        
        # Get document index (0-based)
        doc_index = None
        for i, d in enumerate(self.documents):
            if d.id == doc_id:
                doc_index = i
                break
        
        if doc_index is None:
            return result
        
        # Collect term information
        for term in doc.raw_tf.keys():
            term_info = {
                'raw_tf': doc.raw_tf.get(term, 0),
                'log_tf': doc.log_tf.get(term, 0.0),
                'aug_tf': doc.aug_tf.get(term, 0.0),
                'binary_tf': doc.binary_tf.get(term, 0),
                'idf': self.calculate_idf(term),
                'positions': self.inverse_doc_by_id.get(doc_index, {}).get(term, [])
            }
            result[term] = term_info
        
        return result
    
    def get_term_document_matrix(self, weighting_scheme: str = 'tf_idf', 
                               tf_scheme: str = 'raw') -> Dict[str, Dict[int, float]]:
        """
        Get term-document matrix
        
        Args:
            weighting_scheme: Weighting scheme to use
            tf_scheme: TF calculation scheme
            
        Returns:
            Dict[str, Dict[int, float]]: {term: {doc_id: weight}}
        """
        matrix = {}
        
        # Get all unique terms
        all_terms = set()
        for doc in self.documents:
            all_terms.update(doc.raw_tf.keys())
        
        for term in all_terms:
            matrix[term] = {}
            for doc in self.documents:
                if weighting_scheme == 'tf':
                    weight = self.calculate_tf(term, doc.id, tf_scheme)
                elif weighting_scheme == 'idf':
                    weight = self.calculate_idf(term)
                elif weighting_scheme in ['tf_idf', 'tf_idf_cosine']:
                    weight = self.calculate_tf_idf(term, doc.id, tf_scheme)
                else:
                    weight = self.calculate_tf(term, doc.id, tf_scheme)
                
                if weight > 0:
                    matrix[term][doc.id] = weight
        
        return matrix
    
    # === UTILITY METHODS ===
    
    def save_results_to_file(self, results: Dict, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def display_retrieval_results(self, query_id: str, query_terms: List[str],
                                results: List[Tuple[int, float]], 
                                ap_score: float = None) -> str:
        """Format retrieval results for display"""
        output = f"\n=== Hasil Retrieval untuk Query '{query_id}' ===\n"
        output += f"Query terms: {query_terms}\n"
        output += f"Jumlah dokumen ditemukan: {len(results)}\n"
        
        if ap_score is not None:
            output += f"Average Precision: {ap_score:.4f}\n"
        
        output += "\nRanking Dokumen:\n"
        output += "-" * 60 + "\n"
        
        for rank, (doc_id, similarity) in enumerate(results, 1):
            # Find document title for display
            doc_title = "Unknown"
            for doc in self.documents:
                if doc.id == doc_id:
                    doc_title = doc.title or f"Document {doc_id}"
                    break
            
            output += f"{rank:2d}. Doc ID: {doc_id} | {doc_title} | Similarity: {similarity:.4f}\n"
        
        return output
    
    def get_document_by_id(self, doc_id: int) -> Document:
        """Get document by ID"""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def get_vocabulary(self) -> Set[str]:
        """Get all unique terms in the collection"""
        vocab = set()
        for doc in self.documents:
            vocab.update(doc.raw_tf.keys())
        return vocab
    
    def get_collection_statistics(self) -> Dict[str, any]:
        """Get collection statistics"""
        vocab = self.get_vocabulary()
        total_terms = sum(sum(doc.raw_tf.values()) for doc in self.documents)
        
        stats = {
            'total_documents': len(self.documents),
            'vocabulary_size': len(vocab),
            'total_terms': total_terms,
            'average_doc_length': total_terms / len(self.documents) if self.documents else 0,
            'documents_info': []
        }
        
        for doc in self.documents:
            doc_length = sum(doc.raw_tf.values())
            unique_terms = len(doc.raw_tf)
            
            stats['documents_info'].append({
                'doc_id': doc.id,
                'title': doc.title,
                'author': doc.author,
                'length': doc_length,
                'unique_terms': unique_terms
            })
        
        return stats