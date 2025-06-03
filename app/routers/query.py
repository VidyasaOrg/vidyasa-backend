from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from app.utils.preprocessing import tokenize_words, preprocess_text
import tempfile
import os
import json

from app.services import get_qrels, get_queries, get_irdata
from app.schemas import QueryRequest, QueryResponse, DocumentSimilarityScore, TermFrequencyMethod
from app.models import Qrels
from app.models import Query

from app.services.query_expansion import expand_query_kb

router = APIRouter(prefix="/query", tags=["query"])

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

@router.post("/", response_model=QueryResponse)
async def search_single_query(request: QueryRequest):
    """
    Search for a single query and return the results.
    
    Example:
    ```json
    {
        "query": "example search term"
    }
    ```
    """
    
    
    
    # TODO: Implementasi ini, kalau (bebas mau nama fungsi, input fungsi diubah kalau belum sesuai)
    # @satrianababan
    """
    Contoh Output:
    json:
    ```
    {
        "original_ranking": [
            {"doc_id": 1, "similarity_score": 0.85},
            {"doc_id": 2, "similarity_score": 0.75}
        ],
        "expanded_ranking": [
            {"doc_id": 1, "similarity_score": 0.90},
            {"doc_id": 2, "similarity_score": 0.80}
        ],
        "original_query": "information retrieval",
        "original_map_score": 0.78,
        "original_query_weights": {"information": 0.5, "retrieval": 0.5},
        "expanded_query": "information retrieval systems",
        "expanded_map_score": 0.82,
        "expanded_query_weights": {"information": 0.4, "retrieval": 0.4, "systems": 0.2}
    }
    ```
    
    QueryResponse:
    QueryResponse(
        original_ranking=[DocumentSimilarityScore(doc_id=1, similarity_score=0.85), DocumentSimilarityScore(doc_id=2, similarity_score=0.75)],
        expanded_ranking=[DocumentSimilarityScore(doc_id=1, similarity_score=0.90), DocumentSimilarityScore(doc_id=2, similarity_score=0.80)],
        original_query="information retrieval",
        original_map_score=0.78,
        original_query_weights={"information": 0.5, "retrieval": 0.5},
        expanded_query="information retrieval systems",
        expanded_map_score=0.82,
        expanded_query_weights={"information": 0.4, "retrieval": 0.4, "systems": 0.2}
    )
    """
    try:
        raw_query = request.query
        qrels = get_qrels()
        irdata = get_irdata()
        docs = irdata

        # Preprocessing
        tokens = tokenize_words(raw_query)
        processed_tokens = preprocess_text(tokens, is_stem=False, remove_stop_words=False)  # Update if needed
        processed_query = " ".join(processed_tokens)

        query_obj = Query(id=0, content=processed_query)
        relevant_docs_set = set(qrels.get_relevant_docs(0)) if 0 in qrels.data else set()

        # ORIGINAL QUERY
        original_ranking_objs = dummy_similarity_ranking(query_obj, docs)
        original_ranking_ids = [doc.doc_id for doc in original_ranking_objs]
        original_map = dummy_map_score(original_ranking_ids, relevant_docs_set)
        original_query_weights = dummy_query_weights(raw_query)

        # EXPANDED QUERY
        relevant_doc_texts = [doc.content for doc in docs.documents[:5]]  # Or filter by similarity
        expansion_json = expand_query_kb(raw_query, relevant_doc_texts)
        
        try:
            import json
            expanded_query_text = json.loads(expansion_json)["expanded-query"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Invalid expansion format: {str(e)}")

        expanded_tokens = tokenize_words(expanded_query_text)
        expanded_processed_tokens = preprocess_text(expanded_tokens, is_stem=False, remove_stop_words=False)
        expanded_processed_query = " ".join(expanded_processed_tokens)

        query_obj.content = expanded_processed_query
        expanded_ranking_objs = dummy_similarity_ranking(query_obj, docs)
        expanded_ranking_ids = [doc.doc_id for doc in expanded_ranking_objs]
        expanded_map = dummy_map_score(expanded_ranking_ids, relevant_docs_set)
        expanded_query_weights = dummy_query_weights(expanded_query_text)

        return QueryResponse(
            original_query=raw_query,
            original_ranking=original_ranking_objs,
            original_map_score=original_map,
            original_query_weights=original_query_weights,
            expanded_query=expanded_query_text,
            expanded_ranking=expanded_ranking_objs,
            expanded_map_score=expanded_map,
            expanded_query_weights=expanded_query_weights
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")