from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from app.utils.preprocessing import tokenize_words, preprocess_text
import tempfile
import os
import json
from typing import List, Dict, Tuple

from app.services import get_qrels, get_queries, get_irdata
from app.schemas import QueryRequest, QueryResponse, DocumentSimilarityScore, TermFrequencyMethod
from app.models import Qrels
from app.models import Query

from app.services.query_expansion import expand_query_kb

from app.services.coefficient_calculator import CoefficientCalculator
from app.services.map_calculator import MAPCalculator
from app.services.inverted_index import InvertedIndex

router = APIRouter(prefix="/query", tags=["query"])

# Global instances for reuse
coefficient_calc = CoefficientCalculator()
map_calc = MAPCalculator()
inverted_index = InvertedIndex()

# Global variables to store preprocessed data
_term_doc_matrix = None
_doc_vectors = None
_vocabulary = None
_documents_initialized = False

def initialize_document_collection():
    """Initialize the document collection and build necessary data structures"""
    global _term_doc_matrix, _doc_vectors, _vocabulary, _documents_initialized
    
    if _documents_initialized:
        return
    
    try:
        # Get document collection
        irdata = get_irdata()
        documents = [doc.content for doc in irdata.documents]
        
        # Build term-document matrix with TF-IDF and cosine normalization
        _term_doc_matrix, _doc_vectors, _vocabulary = coefficient_calc.build_term_document_matrix(
            documents=documents,
            do_stemming=True,
            remove_stopwords=True,
            tf_method='logarithmic',  # Use log normalization
            use_idf=True,
            use_cosine_norm=True
        )
        
        # Build inverted index for additional functionality
        inverted_index.build_index(
            documents=documents,
            do_stemming=True,
            remove_stopwords=True,
            tf_method='logarithmic',
            use_idf=True,
            use_cosine_norm=True
        )
        
        _documents_initialized = True
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize document collection: {str(e)}")

def calculate_similarity_ranking(query: Query, docs) -> List[DocumentSimilarityScore]:
    """
    Calculate document similarity scores using cosine similarity with TF-IDF weights
    """
    initialize_document_collection()
    
    try:
        # Process query to get query vector
        query_vector = coefficient_calc.process_query(
            query=query.content,
            term_doc_matrix=_term_doc_matrix,
            do_stemming=True,
            remove_stopwords=True,
            tf_method='logarithmic',
            use_idf=True,
            use_cosine_norm=True
        )
        
        # Calculate similarity scores for all documents
        similarity_scores = []
        
        for doc_id in range(len(docs.documents)):
            if doc_id in _doc_vectors:
                doc_vector = _doc_vectors[doc_id]
                similarity = coefficient_calc.calculate_cosine_similarity(query_vector, doc_vector)
                similarity_scores.append(DocumentSimilarityScore(
                    doc_id=docs.documents[doc_id].id,
                    similarity_score=similarity
                ))
        
        # Sort by similarity score (descending) and return top 10
        similarity_scores.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarity_scores[:10]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity ranking: {str(e)}")

def calculate_query_weights(query: str) -> Dict[str, float]:
    """
    Calculate normalized weights for query terms using TF-IDF
    """
    initialize_document_collection()
    
    try:
        # Process query to get query vector with weights
        query_vector = coefficient_calc.process_query(
            query=query,
            term_doc_matrix=_term_doc_matrix,
            do_stemming=True,
            remove_stopwords=True,
            tf_method='logarithmic',
            use_idf=True,
            use_cosine_norm=True
        )
        
        # Filter out zero weights and normalize to sum to 1
        non_zero_weights = {term: weight for term, weight in query_vector.items() if weight > 0}
        
        if not non_zero_weights:
            return {}
        
        # Normalize weights to sum to 1
        total_weight = sum(non_zero_weights.values())
        normalized_weights = {term: weight / total_weight for term, weight in non_zero_weights.items()}
        
        return normalized_weights
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating query weights: {str(e)}")

def calculate_map_score(ranking_ids: List[int], relevant_docs_set: set[int]) -> float:
    """
    Calculate Mean Average Precision (MAP) score for a single query
    """
    try:
        if not ranking_ids or not relevant_docs_set:
            return 0.0
        
        # Convert to list format expected by MAPCalculator
        relevant_docs_list = list(relevant_docs_set)
        
        # Calculate Average Precision
        ap_score = map_calc.calculate_average_precision(ranking_ids, relevant_docs_list)
        
        return ap_score
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating MAP score: {str(e)}")

def get_relevant_document_texts(docs, similarity_scores: List[DocumentSimilarityScore], top_k: int = 5) -> List[str]:
    """
    Get text content of top-k most similar documents for query expansion
    """
    try:
        # Get top-k documents based on similarity scores
        top_docs = similarity_scores[:top_k]
        
        # Extract document texts
        doc_texts = []
        for doc_score in top_docs:
            # Find document by ID
            for doc in docs.documents:
                if doc.id == doc_score.doc_id:
                    # Limit text length to prevent overly long contexts
                    text = doc.content[:1000] if len(doc.content) > 1000 else doc.content
                    doc_texts.append(text)
                    break
        
        return doc_texts
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting relevant document texts: {str(e)}")

@router.post("/", response_model=QueryResponse)
async def search_single_query(request: QueryRequest):
    """
    Search for a single query and return the results with both original and expanded query rankings.
    
    Example:
    ```json
    {
        "query": "information retrieval systems"
    }
    ```
    """
    try:
        raw_query = request.query
        qrels = get_qrels()
        irdata = get_irdata()
        docs = irdata

        # Preprocessing original query
        tokens = tokenize_words(raw_query)
        processed_tokens = preprocess_text(tokens, is_stem=True, remove_stop_words=True)
        processed_query = " ".join(processed_tokens)

        # Create query object
        query_obj = Query(id=0, content=processed_query)
        relevant_docs_set = set(qrels.get_relevant_docs(0)) if 0 in qrels.data else set()

        # ORIGINAL QUERY PROCESSING
        original_ranking_objs = calculate_similarity_ranking(query_obj, docs)
        original_ranking_ids = [doc.doc_id for doc in original_ranking_objs]
        original_map = calculate_map_score(original_ranking_ids, relevant_docs_set)
        original_query_weights = calculate_query_weights(processed_query)

        # QUERY EXPANSION
        # Get relevant document texts for expansion context
        relevant_doc_texts = get_relevant_document_texts(docs, original_ranking_objs, top_k=5)
        
        # Expand query using knowledge base
        expansion_json = expand_query_kb(raw_query, relevant_doc_texts)
        
        try:
            expansion_data = json.loads(expansion_json)
            expanded_query_text = expansion_data.get("expanded-query", raw_query)
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: use original query if expansion fails
            print(f"Warning: Query expansion failed, using original query. Error: {str(e)}")
            expanded_query_text = raw_query

        # Preprocessing expanded query
        expanded_tokens = tokenize_words(expanded_query_text)
        expanded_processed_tokens = preprocess_text(expanded_tokens, is_stem=True, remove_stop_words=True)
        expanded_processed_query = " ".join(expanded_processed_tokens)

        # EXPANDED QUERY PROCESSING
        expanded_query_obj = Query(id=0, content=expanded_processed_query)
        expanded_ranking_objs = calculate_similarity_ranking(expanded_query_obj, docs)
        expanded_ranking_ids = [doc.doc_id for doc in expanded_ranking_objs]
        expanded_map = calculate_map_score(expanded_ranking_ids, relevant_docs_set)
        expanded_query_weights = calculate_query_weights(expanded_processed_query)

        # Prepare response
        response = QueryResponse(
            original_query=raw_query,
            original_ranking=original_ranking_objs,
            original_map_score=original_map,
            original_query_weights=original_query_weights,
            expanded_query=expanded_query_text,
            expanded_ranking=expanded_ranking_objs,
            expanded_map_score=expanded_map,
            expanded_query_weights=expanded_query_weights
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/term-info/{term}")
async def get_term_information(term: str):
    """
    Get detailed information about a specific term in the collection
    """
    try:
        initialize_document_collection()
        
        # Preprocess the term the same way as during indexing
        processed_terms = coefficient_calc.preprocess_text(term, do_stemming=True, remove_stopwords=False)
        
        if not processed_terms:
            raise HTTPException(status_code=404, detail=f"Term '{term}' not found after preprocessing")
        
        processed_term = processed_terms[0]  # Take the first processed term
        
        # Get term info from inverted index
        term_info = inverted_index.get_term_info(processed_term)
        
        if "error" in term_info:
            raise HTTPException(status_code=404, detail=term_info["error"])
        
        return term_info
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving term information: {str(e)}")

@router.get("/document-info/{doc_id}")
async def get_document_information(doc_id: int):
    """
    Get detailed information about a specific document
    """
    try:
        initialize_document_collection()
        
        # Get document info from inverted index
        doc_info = inverted_index.get_document_info(doc_id)
        
        if "error" in doc_info:
            raise HTTPException(status_code=404, detail=doc_info["error"])
        
        return doc_info
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document information: {str(e)}")

@router.post("/evaluate-rankings")
async def evaluate_rankings(queries_data: Dict[str, List[Tuple[int, float]]]):
    """
    Evaluate multiple query rankings using MAP and other metrics
    
    Expected format:
    {
        "query_1": [(doc_id, similarity_score), ...],
        "query_2": [(doc_id, similarity_score), ...],
        ...
    }
    """
    try:
        qrels = get_qrels()
        
        # Prepare relevance judgments
        relevance_judgments = {}
        for query_id, doc_ids in queries_data.items():
            # Convert query_id to int if it's numeric
            try:
                numeric_id = int(query_id.split('_')[-1]) if 'query_' in query_id else int(query_id)
                if numeric_id in qrels.data:
                    relevance_judgments[query_id] = qrels.get_relevant_docs(numeric_id)
            except (ValueError, KeyError):
                # Skip queries without relevance judgments
                continue
        
        # Calculate detailed evaluation metrics
        evaluation_results = {}
        total_ap = 0
        valid_queries = 0
        
        for query_id, ranking_data in queries_data.items():
            if query_id in relevance_judgments:
                relevant_docs = relevance_judgments[query_id]
                
                # Evaluate this ranking
                eval_result = map_calc.evaluate_ranking(
                    retrieved_docs=ranking_data,
                    relevant_docs=relevant_docs,
                    k_values=[1, 5, 10, 20]
                )
                
                evaluation_results[query_id] = eval_result
                total_ap += eval_result['average_precision']
                valid_queries += 1
        
        # Calculate overall MAP
        overall_map = total_ap / valid_queries if valid_queries > 0 else 0.0
        
        return {
            "overall_map": overall_map,
            "total_queries_evaluated": valid_queries,
            "per_query_results": evaluation_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating rankings: {str(e)}")