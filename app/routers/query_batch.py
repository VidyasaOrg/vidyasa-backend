from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
import json

from app.services.data_loader import get_qrels, get_queries, get_irdata
from app.schemas.query import QueryResponse, DocumentSimiliarityScore
from app.models.qrels import Qrels
from app.models.query import Query

router = APIRouter(prefix="/query_batch", tags=["batch_query"])

@router.post("/", response_class=FileResponse)
async def search_batch_queries(
    file: UploadFile = File(..., description="File containing queries (one per line or JSON format)"),
):
    # TODO: Implementasi ini, kalau (bebas mau nama fungsi, input fungsi diubah kalau belum sesuai)
    # @Breezy-DR
    # @satrianababan disini nanti ada similarity coefficient, original map, expanded map
    # TODO: Replace dummy calculated function with real functions

    try:
        contents = await file.read()
        try:
            queries_data = json.loads(contents)
            if isinstance(queries_data, dict):  # single query
                queries_data = [queries_data]
            elif not isinstance(queries_data, list):
                raise ValueError("Invalid JSON format")
        except json.JSONDecodeError:
            # fallback: treat as newline-separated raw queries
            queries_data = [{"query": line.strip()} for line in contents.decode().splitlines() if line.strip()]

        qrels: Qrels = get_qrels()
        irdata = get_irdata()
        docs = irdata

        responses = []
        for idx, query_entry in enumerate(queries_data):
            raw_query = query_entry["query"]
            query_obj = Query(id=idx, content=raw_query)

            relevant_docs_set = set(qrels.get_relevant_docs(idx)) if idx in qrels.data else set()

            # ORIGINAL QUERY
            original_ranking_objs = dummy_similarity_ranking(query_obj, docs)
            original_ranking_ids = [doc.doc_id for doc in original_ranking_objs]
            original_map = dummy_map_score(original_ranking_ids, relevant_docs_set)
            original_query_weights = dummy_query_weights(raw_query)

            # EXPANDED QUERY
            expanded_query_text = dummy_expand_query(raw_query)
            query_obj.content = expanded_query_text
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
            responses.append(response.dict())

        # Save to temp file to send as FileResponse
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
        json.dump(responses, temp, indent=2)
        temp.close()

        return FileResponse(temp.name, filename="batch_query_results.json", media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
raise HTTPException(status_code=400, detail="not implemented yet")