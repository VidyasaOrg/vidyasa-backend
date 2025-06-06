from fastapi import APIRouter, HTTPException
from typing import List, Dict, Tuple

from app.models import Query
from app.schemas import QueryRequest, QueryResponse, DocumentSimilarityScore
from app.services import get_qrels, get_queries, get_irdata

from app.services import QueryService

router = APIRouter(prefix="/query", tags=["query"])
@router.get("/", response_model=List[Query])
async def get_queries_list():
    """
    Retrieve the list of queries available in the system.
    
    Returns:
        List[Query]: A list of Query objects containing query IDs and texts.
    
    Example Output in json:
    ```
    [
        {"query_id": 1, "query_text": "information retrieval"},
        {"query_id": 2, "query_text": "database systems"}
    ]
    ```
    """
    queries: List[Query] = get_queries()
    return queries

@router.post("/", response_model=QueryResponse)
async def search_single_query(request: QueryRequest):
    """
    Search for a single query and return the results with both original and expanded query rankings.
    
    Args:
        request (QueryRequest): The search query request containing the query text and options.
        if `is_queries_from_cisi` query will search exact `content` match in CISI dataset for MAP calculation.
        - if query in CISI not found MAP score are 0.0
    Returns:
        QueryResponse: The response containing rankings for both original and expanded queries.
    
    Example in json:
    ```
    {
        "query": "information retrieval",
        "is_stemming": true,
        "is_stop_words_removal": false,
        "query_term_frequency_method": "log",
        "query_term_weighting_method": "tf_idf",
        "document_term_frequency_method": "raw",
        "document_term_weighting_method": "tf",
        "cosine_normalization_query": false,
        "cosine_normalization_document": false,
        "expansion_terms_count": 5
        "is_queries_from_cisi": false
    }
    ```
    
    Example Output in json:
    ```
    {
        "original_ranking": [
            {"doc_id": 1, "doc_title": "Document 1", "similarity_score": 0.85},
            {"doc_id": 2, "doc_title": "Document 2", "similarity_score": 0.75}
        ],
        "expanded_ranking": [
            {"doc_id": 3, "doc_title": "Document 3", "similarity_score": 0.90},
            {"doc_id": 4, "doc_title": "Document 4", "similarity_score": 0.80}
        ],
        "original_query": "information retrieval",
        "original_map_score": 0.75,
        "original_query_weights": {"information": 0.5, "retrieval": 0.5}, 
        "expanded_query": "information retrieval knowledge base",
        "expanded_map_score": 0.80,
        "expanded_query_weights": {"information": 0.4, "retrieval": 0.4, "knowledge": 0.2},
    }
    ```
    """
    try:
        query_service = QueryService(
            irdata=get_irdata(),
            qrels=get_qrels(),
            queries=get_queries()
        )
        response = query_service.process_single_query(request, only_expands_from_kb=True)
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/exp/", response_model=QueryResponse)
async def search_single_query(request: QueryRequest):
    """
    Search for a single query and return the results with both original and expanded query rankings.
    
    Args:
        request (QueryRequest): The search query request containing the query text and options.
        if `is_queries_from_cisi` query will search exact `content` match in CISI dataset for MAP calculation.
        - if query in CISI not found MAP score are 0.0
    Returns:
        QueryResponse: The response containing rankings for both original and expanded queries.
    
    Example in json:
    ```
    {
        "query": "information retrieval",
        "is_stemming": true,
        "is_stop_words_removal": false,
        "query_term_frequency_method": "log",
        "query_term_weighting_method": "tf_idf",
        "document_term_frequency_method": "raw",
        "document_term_weighting_method": "tf",
        "cosine_normalization_query": false,
        "cosine_normalization_document": false,
        "expansion_terms_count": 5
        "is_queries_from_cisi": false
    }
    ```
    
    Example Output in json:
    ```
    {
        "original_ranking": [
            {"doc_id": 1, "doc_title": "Document 1", "similarity_score": 0.85},
            {"doc_id": 2, "doc_title": "Document 2", "similarity_score": 0.75}
        ],
        "expanded_ranking": [
            {"doc_id": 3, "doc_title": "Document 3", "similarity_score": 0.90},
            {"doc_id": 4, "doc_title": "Document 4", "similarity_score": 0.80}
        ],
        "original_query": "information retrieval",
        "original_map_score": 0.75,
        "original_query_weights": {"information": 0.5, "retrieval": 0.5},
        "expanded_query": "information retrieval knowledge base",
        "expanded_map_score": 0.80,
        "expanded_query_weights": {"information": 0.4, "retrieval": 0.4, "knowledge": 0.2},
    }
    ```
    """
    try:
        query_service = QueryService(
            irdata=get_irdata(),
            qrels=get_qrels(),
            queries=get_queries()
        )
        response = query_service.process_single_query(request, only_expands_from_kb=False)
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/id", response_model=QueryResponse)
async def search_query_by_id(query_id: int):
    """
    NOT IMPLEMENTED YET
    Search for a query by its ID and return the results with both original and expanded query rankings.
    
    Args:
        query_id (int): The
        ID of the query to search for.
    Returns:
        QueryResponse: The response containing rankings for both original and expanded queries.
    Example in json:
    ```
    {
        "query_id": 1
    }
    ```
    Example Output in json:
    ```
    {
        "original_ranking": [
            {"doc_id": 1, "doc_title": "Document 1", "similarity_score": 0.85},
            {"doc_id": 2, "doc_title": "Document 2", "similarity_score": 0.75}
        ],
        "expanded_ranking": [
            {"doc_id": 3, "doc_title": "Document 3", "similarity_score": 0.90},
            {"doc_id": 4, "doc_title": "Document 4", "similarity_score": 0.80}
        ]
        "original_query": "information retrieval",
        "original_map_score": 0.75,
        "original_query_weights": {"information": 0.5, "retrieval": 0.5},
        "expanded_query": "information retrieval knowledge base",
        "expanded_map_score": 0.80,
        "expanded_query_weights": {"information": 0.4, "retrieval": 0.4, "knowledge": 0.2},
    }
    """
    try:
        pass
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")