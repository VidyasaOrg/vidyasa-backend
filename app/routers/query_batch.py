from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from app.utils.preprocessing import tokenize_words, preprocess_text
import tempfile
import os
import json

from app.services import get_qrels, get_queries, get_irdata
from app.schemas import QueryResponse, DocumentSimilarityScore, TermFrequencyMethod
from app.models import Qrels
from app.models import Query

from app.services.query_expansion import expand_query_kb

router = APIRouter(prefix="/query_batch", tags=["batch_query"])

#######################
# REMOVE LATER: Dummy functions to be replaced with actual implementations

from app.models import Document, IRData
def dummy_similarity_ranking(query: Query, docs: IRData) -> list[DocumentSimilarityScore]:
    # Dummy implementation: return 10 documents sorted by similarity score
    return [DocumentSimilarityScore(doc_id=doc.id, similarity_score=0.5) for doc in docs.documents][:10]

def dummy_query_weights(query: str) -> dict:
    # Dummy implementation: return a fixed weight for each term
    return {term: 1.0 for term in query.split()}

def dummy_expand_query(query: str) -> str:
    # Dummy implementation: return the original query with "dummy" appended
    return f"{query} dummy"

def dummy_map_score(ranking_ids: list[int], relevant_docs_set: set[int]) -> float:
    # Dummy implementation: return a fixed MAP score
    if not ranking_ids or not relevant_docs_set:
        return 0.0
    return len(set(ranking_ids) & relevant_docs_set) / len(relevant_docs_set)

#########################

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
    """
        Args:
            file: The uploaded file object (expected to be text-based) containing queries, one per line.
            is_stemming (bool): Whether to apply stemming during preprocessing.
            is_stop_words_removal (bool): Whether to remove stop words during preprocessing.
            term_frequency_method,
            expansion_terms_count
        Returns:
            FileResponse: A downloadable JSON file containing a list of responses for each query,
                          including original and expanded query rankings, MAP scores, and term weights.

        Raises:
            HTTPException: If any error occurs during processing, returns HTTP 500 with error detail.
        """
    # TODO: Implementasi ini, kalau (bebas mau nama fungsi, input fungsi diubah kalau belum sesuai)
    # @Breezy-DR
    # @satrianababan disini nanti ada similarity coefficient, original map, expanded map
    # TODO: Replace dummy calculated function with real functions

    try:
        contents = await file.read()

        queries_data = [{"query": line.strip()} for line in contents.decode().splitlines() if line.strip()]

        qrels: Qrels = get_qrels()
        irdata = get_irdata()
        docs = irdata

        # Retrieve responses from each query
        responses = []
        for idx, query_entry in enumerate(queries_data):
            raw_query = query_entry["query"]
            tokens = tokenize_words(raw_query)
            processed_tokens = preprocess_text(tokens, is_stem=is_stemming, remove_stop_words=is_stop_words_removal)
            processed_query = " ".join(processed_tokens)

            query_obj = Query(id=idx, content=processed_query)

            # If the index query is in the qrels data, it is relevant
            relevant_docs_set = set(qrels.get_relevant_docs(idx)) if idx in qrels.data else set()

            # ORIGINAL QUERY
            original_ranking_objs = dummy_similarity_ranking(query_obj, docs)
            original_ranking_ids = [doc.doc_id for doc in original_ranking_objs]
            original_map = dummy_map_score(original_ranking_ids, relevant_docs_set)
            original_query_weights = dummy_query_weights(raw_query)

            # EXPANDED QUERY
            relevant_docs_texts = [doc.content for doc in docs.documents if doc.id in original_ranking_ids]
            expansion_result = expand_query_kb(processed_query, relevant_docs_texts)
            expanded_query_text = " ".join(expansion_result.keys())

            # Preprocess expanded query
            expanded_tokens = tokenize_words(expanded_query_text)
            expanded_processed_tokens = preprocess_text(expanded_tokens, is_stem=is_stemming,
                                                        remove_stop_words=is_stop_words_removal)
            expanded_processed_query = " ".join(expanded_processed_tokens)

            query_obj.content = expanded_processed_query
            expanded_ranking_objs = dummy_similarity_ranking(query_obj, docs)
            expanded_ranking_ids = [doc.doc_id for doc in expanded_ranking_objs]
            expanded_map = dummy_map_score(expanded_ranking_ids, relevant_docs_set)
            expanded_query_weights = dummy_query_weights(expanded_query_text)

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
            responses.append(response.model_dump())

        # Save to temp file to send as FileResponse
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
        json.dump(responses, temp, indent=2)
        temp.close()

        return FileResponse(temp.name, filename="batch_query_results.json", media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")