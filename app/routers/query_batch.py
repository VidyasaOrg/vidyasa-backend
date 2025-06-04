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
    file: UploadFile = File(..., description="File containing query IDs (one per line)"),
    is_stemming: bool = Form(False, description="Apply stemming to queries"),
    is_stop_words_removal: bool = Form(False, description="Remove stop words from queries"),
    query_term_frequency_method: TermFrequencyMethod = Form(TermFrequencyMethod.RAW, description="Term frequency method to use in queries"),
    query_term_weighting_method: TermWeightingMethod = Form(TermWeightingMethod.TF, description="Term weighting method to use in queries"),
    document_term_frequency_method: TermFrequencyMethod = Form(TermFrequencyMethod.RAW, description="Term frequency method to use in documents"),
    document_term_weighting_method: TermWeightingMethod = Form(TermWeightingMethod.TF, description="Term weighting method to use in documents"),
    cosine_normalization_query: bool = Form(False, description="Whether to apply cosine normalization to query vectors"),
    cosine_normalization_document: bool = Form(False, description="Whether to apply cosine normalization to document vectors"),
    expansion_terms_count: str = Form("5", description="Number of terms to expand the query with (0 for no expansion, or 'all')"),
    is_queries_from_cisi: bool = Form(True, description="Whether queries are from CISI dataset")
):
    """
    Process a batch of queries from a file and return the results.
    The file should contain query IDs, one per line.
    Each query will be processed to retrieve original and expanded rankings, MAP scores, and term weights.
    
    Args:
        file: Text file containing query IDs (one per line)
        is_stemming: Whether to apply stemming to queries
        is_stop_words_removal: Whether to remove stop words from queries
        query_term_frequency_method: Method to calculate term frequency in queries
        query_term_weighting_method: Method to calculate term weights in queries
        document_term_frequency_method: Method to calculate term frequency in documents
        document_term_weighting_method: Method to calculate term weights in documents
        cosine_normalization_query: Whether to apply cosine normalization to query vectors
        cosine_normalization_document: Whether to apply cosine normalization to document vectors
        expansion_terms_count: Number of terms to use for query expansion
        is_queries_from_cisi: Whether queries are from CISI dataset (defaults to True for batch processing)
        
    Returns:
        QueryBatchResponse: List of query results, one for each query in the batch
        
    Example file content:
    ```
    1
    44
    55
    ```
    """
    try:
        if expansion_terms_count == "all":
            expansion_terms_count_value = "all"
        else:
            try:
                expansion_terms_count_value = int(expansion_terms_count)
            except ValueError:
                raise HTTPException(status_code=400, detail="expansion_terms_count must be an integer or 'all'")
        # Read and process the file content
        content = await file.read()
        # Parse query IDs, one per line
        query_ids = [int(qid.strip()) for qid in content.decode().strip().split('\n') if qid.strip()]
        
        # Initialize services
        query_service = QueryService(
            irdata=get_irdata(),
            qrels=get_qrels(),
            queries=get_queries()
        )
        
        # Get all CISI queries for lookup
        cisi_queries = {q.id: q for q in query_service.queries}
        
        # Process each query
        results = []
        for qid in query_ids:
            if qid not in cisi_queries:
                print(f"Warning: Query ID {qid} not found in CISI dataset")
                continue
                
            # Create query request
            request = QueryRequest(
                query=cisi_queries[qid].content,
                query_id=qid,
                is_stemming=is_stemming,
                is_stop_words_removal=is_stop_words_removal,
                query_term_frequency_method=query_term_frequency_method,
                query_term_weighting_method=query_term_weighting_method,
                document_term_frequency_method=document_term_frequency_method,
                document_term_weighting_method=document_term_weighting_method,
                cosine_normalization_query=cosine_normalization_query,
                cosine_normalization_document=cosine_normalization_document,
                expansion_terms_count=expansion_terms_count_value,
                is_queries_from_cisi=is_queries_from_cisi
            )
            
            # Process query
            try:
                response = query_service.process_single_query(request)
                results.append(response)
            except Exception as e:
                print(f"Error processing query ID {qid}: {str(e)}")
                continue
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid queries were processed")
            
        return QueryBatchResponse(results=results)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid query ID format: {str(e)}")
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
        "query_term_frequency_method": "log",
        "query_term_weighting_method": "tf_idf",
        "document_term_frequency_method": "raw",
        "document_term_weighting_method": "tf",
        "cosine_normalization_query": false,
        "cosine_normalization_document": false,
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
                query_term_frequency_method=request.query_term_frequency_method,
                query_term_weighting_method=request.query_term_weighting_method,
                document_term_frequency_method=request.document_term_frequency_method,
                document_term_weighting_method=request.document_term_weighting_method,
                cosine_normalization_query=request.cosine_normalization_query,
                cosine_normalization_document=request.cosine_normalization_document,
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