from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List
import json

from app.services import get_qrels, get_queries, get_irdata, QueryService
from app.schemas import (
    QueryResponse, 
    QueryBatchResponse, 
    QueryRequest,
    QueryBatchRequest,
    TermFrequencyMethod,
    TermWeightingMethod
)

router = APIRouter(prefix="/query_batch", tags=["batch_query"])

@router.post("/", response_model=QueryBatchResponse)
async def search_batch_queries(
    file: UploadFile = File(..., description="File containing queries (one per line)"),
    is_stemming: bool = Form(False, description="Apply stemming to queries"),
    is_stop_words_removal: bool = Form(False, description="Remove stop words from queries"),
    term_frequency_method: TermFrequencyMethod = Form(TermFrequencyMethod.RAW, description="Term frequency method to use"),
    term_weighting_method: TermWeightingMethod = Form(TermWeightingMethod.TF, description="Term weighting method to use"),
    expansion_terms_count: int = Form(5, description="Number of terms to expand the query with (0 for no expansion)"),
    is_queries_from_cisi: bool = Form(False, description="Whether queries are from CISI dataset")
):
    """
    Process a batch of queries from a file and return the results.
    The file should contain one query per line.
    Each query will be processed to retrieve original and expanded rankings, MAP scores, and term weights.
    
    Args:
        file: Text file containing one query per line
        is_stemming: Whether to apply stemming to queries
        is_stop_words_removal: Whether to remove stop words from queries
        term_frequency_method: Method to calculate term frequency
        term_weighting_method: Method to calculate term weights
        expansion_terms_count: Number of terms to use for query expansion
        is_queries_from_cisi: Whether queries are from CISI dataset
        
    Returns:
        QueryBatchResponse: List of query results, one for each query in the batch
        
    Example file content:
    ```
    information retrieval systems
    database management
    artificial intelligence
    ```
    """
    try:
        # Read and process the file content
        content = await file.read()
        queries = [q.strip() for q in content.decode().strip().split('\n') if q.strip()]
        
        # Create batch request
        batch_request = QueryBatchRequest(
            queries=queries,
            is_stemming=is_stemming,
            is_stop_words_removal=is_stop_words_removal,
            term_frequency_method=term_frequency_method,
            term_weighting_method=term_weighting_method,
            expansion_terms_count=expansion_terms_count,
            is_queries_from_cisi=is_queries_from_cisi
        )
        
        # Initialize services
        query_service = QueryService(
            irdata=get_irdata(),
            qrels=get_qrels(),
            queries=get_queries()
        )
        
        # Process each query
        results = []
        for query in batch_request.queries:
            # Create query request
            request = QueryRequest(
                query=query,
                is_stemming=batch_request.is_stemming,
                is_stop_words_removal=batch_request.is_stop_words_removal,
                term_frequency_method=batch_request.term_frequency_method,
                term_weighting_method=batch_request.term_weighting_method,
                expansion_terms_count=batch_request.expansion_terms_count,
                is_queries_from_cisi=batch_request.is_queries_from_cisi
            )
            
            # Process query
            try:
                response = query_service.process_single_query(request)
                results.append(response)
            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
                continue
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid queries were processed")
            
        return QueryBatchResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/json", response_model=QueryBatchResponse)
async def search_batch_queries_json(request: QueryBatchRequest):
    """
    Process a batch of queries provided in JSON format and return the results.
    Each query will be processed to retrieve original and expanded rankings, MAP scores, and term weights.
    
    Args:
        request (QueryBatchRequest): The batch query request containing queries and processing options
        
    Returns:
        QueryBatchResponse: List of query results, one for each query in the batch
        
    Example request body:
    ```json
    {
        "queries": [
            "information retrieval",
            "database systems",
            "artificial intelligence"
        ],
        "is_stemming": true,
        "is_stop_words_removal": false,
        "term_frequency_method": "log",
        "term_weighting_method": "tf_idf",
        "expansion_terms_count": 5,
        "is_queries_from_cisi": false
    }
    ```
    """
    try:
        # Initialize services
        query_service = QueryService(
            irdata=get_irdata(),
            qrels=get_qrels(),
            queries=get_queries()
        )
        
        # Process each query
        results = []
        for query in request.queries:
            # Create query request
            single_request = QueryRequest(
                query=query,
                is_stemming=request.is_stemming,
                is_stop_words_removal=request.is_stop_words_removal,
                term_frequency_method=request.term_frequency_method,
                term_weighting_method=request.term_weighting_method,
                expansion_terms_count=request.expansion_terms_count,
                is_queries_from_cisi=request.is_queries_from_cisi
            )
            
            # Process query
            try:
                response = query_service.process_single_query(single_request)
                results.append(response)
            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
                continue
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid queries were processed")
            
        return QueryBatchResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")