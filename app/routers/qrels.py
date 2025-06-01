from fastapi import APIRouter, HTTPException, Path

from app.services.data_loader import get_qrels
from app.models.qrels import Qrels
from app.schemas.qrels import QrelsResponse

router = APIRouter(prefix="/qrels", tags=["relevant_docs"])


@router.get("/{query_id}", response_model=QrelsResponse)
async def get_relevant_docs(
    query_id: int = Path(..., description="Query ID to retrieve relevance judgments for")
):
    """
    Retrieve relevance judgments (qrels) for a specific query in CISI data.
    
    Args:
        query_id (int): The query ID to get relevance judgments for
        
    Returns:
        QrelsResponse: Query ID and list of relevant document IDs
    """
    qrels: Qrels = get_qrels()
    if query_id not in qrels.data:
        raise HTTPException(status_code=404, detail="Query ID not found in relevance judgments")

    relevant_docs = qrels.get_relevant_docs(query_id)
    return QrelsResponse(
        query_id=query_id,
        relevant_docs=relevant_docs
    )