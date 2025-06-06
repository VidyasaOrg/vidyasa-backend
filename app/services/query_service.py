from typing import List, Dict, Union, Literal, Optional, Tuple
import math
from collections import defaultdict



from app.models import IRData, Qrels, Document, Query
from app.schemas import QueryRequest, QueryResponse, DocumentSimilarityScore, TermWeightingMethod, TermFrequencyMethod
from app.utils import tokenize_and_preprocess
from app.services.query_expansion import expand_query_kb, expand_query_from_exp

class QueryService:
    """Service untuk pemrosesan query lengkap"""
    
    def __init__(self, irdata: IRData, qrels: Qrels, queries: List[Query]):
        self.irdata = irdata
        self.qrels = qrels
        self.queries = queries
        self.similarity_service = SimilarityService(irdata)
        self.evaluation_service = EvaluationService()
    
    def _filter_and_normalize_rankings(
        self,
        original_ranking: List[DocumentSimilarityScore],
        expanded_ranking: List[DocumentSimilarityScore]
    ) -> Tuple[List[DocumentSimilarityScore], List[DocumentSimilarityScore], List[DocumentSimilarityScore]]:
        """
        Filter out documents with similarity score <= 0 and ensure consistent response length.
        
        Args:
            original_ranking: List of documents ranked by original query
            expanded_ranking: List of documents ranked by expanded query
            
        Returns:
            Tuple containing filtered and length-normalized original and expanded rankings
        """
        # Filter out documents with similarity score <= 0
        original_filtered = [doc for doc in original_ranking if doc.similarity_score > 0]
        expanded_filtered = [doc for doc in expanded_ranking if doc.similarity_score > 0]
        
        # Get the maximum length between the two rankings
        max_length = max(len(original_filtered), len(expanded_filtered))
        
        # Return top max_length documents for both rankings
        return (
            original_filtered[:max_length],
            expanded_filtered[:max_length]
        )

    def process_single_query(self, request: QueryRequest, only_expands_from_kb: bool) -> QueryResponse:
        """
        Process a single query and return results.
        """
        # 1. Input Query
        raw_query = request.query
        query_id = request.query_id
        is_stemming = request.is_stemming
        is_stop_words_removal = request.is_stop_words_removal
        query_term_frequency_method = request.query_term_frequency_method
        query_term_weighting_method = request.query_term_weighting_method
        document_term_frequency_method=request.document_term_frequency_method
        document_term_weighting_method=request.document_term_weighting_method
        cosine_normalization_query=request.cosine_normalization_query
        cosine_normalization_document=request.cosine_normalization_document
        expansion_terms_count = request.expansion_terms_count
        is_queries_from_cisi = request.is_queries_from_cisi

        if not raw_query.strip():
            raise ValueError("Query cannot be empty.")

        ### ORIGINAL QUERY PROCESSING ###

        # 2. Preprocessing Query
        preprocessed_tokens = tokenize_and_preprocess(
            raw_query,
            is_stem=is_stemming,
            is_stop_words_removal=is_stop_words_removal
        )

        if not preprocessed_tokens:
            raise ValueError("Query must contain at least one valid token after preprocessing.")

        # 3. Weights Per Term
        original_query_weights = self.similarity_service.calculate_term_weights(
            preprocessed_tokens,
            query_term_frequency_method,
            query_term_weighting_method,
            cosine_normalization_query=cosine_normalization_query
        )

        # 4. Document Ranking
        original_ranking = self.similarity_service.rank_documents(
            original_query_weights,
            document_term_frequency_method,
            document_term_weighting_method,
            cosine_normalization_document=cosine_normalization_document
        )

        # 5. MAP
        original_ranking_ids = [sim.doc_id for sim in original_ranking]
        relevant_docs = None

        if is_queries_from_cisi:
            if query_id is not None:
                relevant_docs = set(self.qrels.get_relevant_docs(query_id))

        original_map_score = self.evaluation_service.calculate_map_score(
            original_ranking_ids,
            relevant_docs,
        )

        ### QUERY EXPANSION PROCESSING ###

        # 6. Query Expansion
        if only_expands_from_kb:
            expanded_query_text = self._expand_query(request.query, original_ranking_ids[:5], expansion_terms_count)
        else:
            expanded_query_text = self._expand_query_from_exp(request.query)

        # 7. Preprocessing Expanded Query
        expanded_tokens = self._process_expanded_query(
            expanded_query_text,
            preprocessed_tokens,
            expansion_terms_count,
            is_stemming,
            is_stop_words_removal
        )

        # 8. Weights Per Term for Expanded Query
        expanded_query_weights = self.similarity_service.calculate_term_weights(
            expanded_tokens,
            request.query_term_frequency_method,
            request.query_term_weighting_method,
            cosine_normalization_query=request.cosine_normalization_query
        )

        # 9. Document Ranking for Expanded Query
        expanded_ranking = self.similarity_service.rank_documents(
            expanded_query_weights,
            request.document_term_frequency_method,
            request.document_term_weighting_method,
            cosine_normalization_document=request.cosine_normalization_document,
        )

        # Filter and normalize rankings
        original_ranking, expanded_ranking = self._filter_and_normalize_rankings(
            original_ranking, expanded_ranking
        )

        # Update ranking IDs for MAP calculation
        expanded_ranking_ids = [sim.doc_id for sim in expanded_ranking]
        expanded_map = float(self.evaluation_service.calculate_map_score(
            expanded_ranking_ids,
            relevant_docs
        ))

        return QueryResponse(
            original_ranking=original_ranking,
            expanded_ranking=expanded_ranking,
            
            original_query=raw_query,
            original_map_score=original_map_score,
            original_query_weights=original_query_weights,
            
            expanded_query=expanded_query_text,
            expanded_map_score=expanded_map,
            expanded_query_weights=expanded_query_weights
        )
    
    def _expand_query(self, original_query: str, top_doc_ids: List[int], expansion_terms_count) -> str:
        """Expand query menggunakan top documents"""
        try:
            # Get document texts for expansion
            relevant_doc_texts = []
            for doc in self.irdata.documents:
                if doc.id in top_doc_ids:
                    doc_text = f"{doc.title} {doc.content}".strip()
                    relevant_doc_texts.append(doc_text)
            
            # Expand query using AI
            expansion_result = expand_query_kb(original_query, relevant_doc_texts, expansion_terms_count)
            return expansion_result.get("expanded-query", original_query)
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return original_query
    
    def _expand_query_from_exp(self, original_query: str) -> str:
        """Expand query menggunakan top documents"""
        try:
            # Expand query using AI
            expansion_result = expand_query_from_exp(original_query)
            return expansion_result.get("expanded-query", original_query)
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return original_query
    
    def _process_expanded_query(
        self, 
        expanded_query_text: str, 
        original_tokens: List[str], 
        expansion_terms_count: Union[int, Literal["all"]], 
        is_stemming: bool, 
        is_stop_words_removal: bool
    ) -> List[str]:
        """Process expanded query dengan term limiting"""
        expanded_tokens = tokenize_and_preprocess(
            expanded_query_text,
            is_stem=is_stemming,
            is_stop_words_removal=is_stop_words_removal
        )
        
        # Limit expansion terms if specified
        if isinstance(expansion_terms_count, int) and expansion_terms_count > 0:
            original_set = set(original_tokens)
            new_tokens = [t for t in expanded_tokens if t not in original_set]
            expanded_tokens = original_tokens + new_tokens[:expansion_terms_count]
        
        return expanded_tokens

class SimilarityService:
    """Service untuk perhitungan similarity dan ranking dokumen"""
    
    def __init__(self, irdata: IRData):
        self.irdata = irdata
    
    def calculate_term_weights(
        self, 
        tokens: List[str], 
        tf_method: TermFrequencyMethod,
        weighting_method: TermWeightingMethod,
        cosine_normalization_query: bool = False
    ) -> Dict[str, float]:
        """Calculate query vector weights"""
        # Count term frequencies in query
        query_tf = defaultdict(int)
        for token in tokens:
            query_tf[token] += 1
        
        # Apply TF weighting
        query_weights = {}
        max_tf = max(query_tf.values()) if query_tf else 1
        
        for term, tf in query_tf.items():
            # Calculate TF component
            if tf_method == TermFrequencyMethod.RAW:
                tf_weight = tf
            elif tf_method == TermFrequencyMethod.LOG:
                tf_weight = 1 + math.log2(tf) if tf > 0 else 0
            elif tf_method == TermFrequencyMethod.BINARY:
                tf_weight = 1 if tf > 0 else 0
            elif tf_method == TermFrequencyMethod.AUGMENTED:
                tf_weight = 0.5 + 0.5 * (tf / max_tf) if max_tf > 0 else 0
            else:
                tf_weight = tf
            
            # Apply weighting method
            if weighting_method == TermWeightingMethod.TF:
                final_weight = tf_weight
            elif weighting_method == TermWeightingMethod.IDF:
                final_weight = self.irdata.idf.get(term, 0)
            elif weighting_method == TermWeightingMethod.TF_IDF:
                final_weight = tf_weight * self.irdata.idf.get(term, 0)
            else:
                final_weight = tf_weight
                
            query_weights[term] = final_weight
        
        # Apply cosine normalization if cosine_normalization_query = True
        if cosine_normalization_query and query_weights:
            norm = math.sqrt(sum(w**2 for w in query_weights.values()))
            if norm > 0:
                query_weights = {term: weight/norm for term, weight in query_weights.items()}
        return query_weights
    
    def calculate_document_similarity(
        self, 
        query_weights: Dict[str, float], 
        doc: Document, 
        tf_method: str, 
        weighting_method: TermWeightingMethod,
        cosine_normalization_document: bool = False
    ) -> float:
        """Calculate similarity between query and document"""
        # Get document TF based on method
        if tf_method == "raw":
            doc_tf = doc.raw_tf
        elif tf_method == "log":
            doc_tf = doc.log_tf
        elif tf_method == "binary":
            doc_tf = doc.binary_tf
        elif tf_method == "augmented":
            doc_tf = doc.aug_tf
        else:
            doc_tf = doc.raw_tf
        
        # Calculate document weights
        doc_weights = {}
        for term, tf in doc_tf.items():
            # Apply weighting method
            if weighting_method == TermWeightingMethod.TF:
                weight = tf
            elif weighting_method == TermWeightingMethod.IDF:
                weight = self.irdata.idf.get(term, 0)
            elif weighting_method == TermWeightingMethod.TF_IDF:
                weight = tf * self.irdata.idf.get(term, 0)
            else:
                weight = tf
                
            doc_weights[term] = weight
        
        # Apply cosine normalization if cosine_normalization_document = True
        if cosine_normalization_document and doc_weights:
            norm = math.sqrt(sum(w**2 for w in doc_weights.values()))
            if norm > 0:
                doc_weights = {term: weight/norm for term, weight in doc_weights.items()}
        elif doc_weights:
            # Apply normal (L1) normalization if cosine normalization is not used
            norm = sum(abs(w) for w in doc_weights.values())
            if norm > 0:
                doc_weights = {term: weight/norm for term, weight in doc_weights.items()}
                            
        # Calculate similarity (dot product)
        similarity = 0.0
        for term, q_weight in query_weights.items():
            if term in doc_weights:
                similarity += q_weight * doc_weights[term]
        
        return similarity
    
    def rank_documents(
        self, 
        query_weights: Dict[str, float], 
        tf_method: str, 
        weighting_method: TermWeightingMethod,
        cosine_normalization_document: bool = False
    ) -> List[DocumentSimilarityScore]:
        """Rank all documents by similarity to query"""
        similarities = []
        
        for doc in self.irdata.documents:
            similarity = self.calculate_document_similarity(
                query_weights, doc, tf_method, weighting_method, cosine_normalization_document
            )
            similarities.append(DocumentSimilarityScore(
                doc_id=doc.id,
                doc_title=doc.title,
                similarity_score=similarity
            ))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities

class EvaluationService:
    """Service untuk perhitungan evaluasi IR"""
    
    @staticmethod
    def calculate_map_score(ranked_doc_ids: List[int], relevant_docs: set, k: Optional[int] = None) -> float:
        """
        Calculate Mean Average Precision

        Args:
            ranked_doc_ids (List[int]): List of document IDs in the ranking order.
            relevant_docs (set): Set of relevant document IDs for the query.
            k (Optional[int]): Optional limit to consider only the top k documents.
            
        Returns:
            float: Mean Average Precision score.
        """
        if not relevant_docs:
            return 0.0
        
        if k is not None:
            ranked_doc_ids = ranked_doc_ids[:k]
        
        relevant_retrieved = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(ranked_doc_ids):
            if doc_id in relevant_docs:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                precision_sum += precision_at_i

        return float(precision_sum / len(relevant_docs)) if relevant_docs else 0.0