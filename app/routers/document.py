from fastapi import APIRouter, HTTPException, Path, Query

from app.schemas.document import DocumentBatchResponse, DocumentResponse
from app.services.data_loader import get_irdata
from app.models.ir_data import IRData

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("/", response_model=DocumentBatchResponse)
async def get_documents_by_ids(
    ids: str = Query(..., description="Comma-separated list of document IDs (e.g., '1,2,3')")
):
    """
    Retrieve multiple documents by their IDs.
    
    Args:
        ids (str): Comma-separated document IDs to retrieve
        
    Returns:
        DocumentBatchResponse: List of documents
    """
    try:
        doc_ids = [int(id_str.strip()) for id_str in ids.split(',') if id_str.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document IDs format. Use comma-separated integers (e.g., '1,2,3')")
    
    if not doc_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    irdata: IRData = get_irdata()
    documents = []
    not_found = []
    
    doc_map = {doc.id: doc for doc in irdata.documents}
    
    for doc_id in doc_ids:
        if doc_id not in doc_map:
            not_found.append(doc_id)
            continue
            
        document = doc_map[doc_id]
        documents.append(DocumentResponse(
            doc_id=document.id,
            title=document.title,
            author=document.author,
            content=document.content
        ))
    
    return DocumentBatchResponse(
        documents=documents,
        not_found=not_found
    )