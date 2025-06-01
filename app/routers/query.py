from fastapi import APIRouter, HTTPException

from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])

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
    
    raise HTTPException(status_code=400, detail="not implemented yet")