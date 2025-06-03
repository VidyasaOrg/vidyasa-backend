from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from app.utils import tokenize_words, preprocess_text
import tempfile
import os
import json

from app.services import get_qrels, get_queries, get_irdata
from app.schemas import QueryResponse, DocumentSimilarityScore, TermFrequencyMethod
from app.models import Qrels
from app.models import Query

from app.services.query_expansion import expand_query_kb


router = APIRouter(prefix="/query_batch", tags=["batch_query"])

@router.post("/", response_class=FileResponse)
async def search_batch_queries(
    file: UploadFile = File(..., description="File containing queries (one per line or JSON format)"),
    is_stemming: bool = Form(False, description="Apply stemming to queries"),
    is_stop_words_removal: bool = Form(False, description="Remove stop words from queries"),
    term_frequency_method: TermFrequencyMethod = Form(TermFrequencyMethod.RAW, description="Term frequency method to use"),
    expansion_terms_count: int = Form("all", description="Number of terms to expand the query with (0 for no expansion, 'all' for all terms)")
):
    """
    Process a batch of queries from a file and return the results as a JSON file.
    The file can contain queries in JSON format or as newline-separated raw queries.
    Each query will be processed to retrieve original and expanded rankings, MAP scores, and term weights.
    
    Example of file content:
    ```
    [
        {"query": "information retrieval"},
        {"query": "database systems"},
        {"query": "machine learning"}
    ]
    ```
    
    Example request body:
    ```json
    {
        "file": "<file with queries>",
        "is_stemming": true,
        "is_stop_words_removal": false,
        "term_frequency_method": "log",
        "expansion_terms_count": 5
    }
    ```
    """
    try:
        pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")