from fastapi import APIRouter, HTTPException, Path

from app.schemas.inverted_file import InvertedFileByDocIdResponse, InvertedFileByTermResponse
from app.services.data_loader import get_irdata
from app.models.ir_data import IRData

router = APIRouter(prefix="/inverted_file", tags=["inverted_index"])

@router.get("/term/{term}", response_model=InvertedFileByTermResponse)
async def get_posting_list_by_term(
    term: str = Path(..., description="Term to lookup in inverted index", min_length=1)
):
    """
    Retrieve posting list for a specific term from the inverted index.
    
    Args:
        term (str): The term to lookup in the inverted index
        
    Returns:
        InvertedFileByTermResponse: Term and list of document IDs containing the term (empty list if term not found)
        
    Raises:
        HTTPException: 400 if term is invalid
    """
    try:
        irdata: IRData = get_irdata()
        
        # Normalize term to lowercase for consistent lookup
        normalized_term = term.lower().strip()
        
        if not normalized_term:
            raise HTTPException(status_code=400, detail="Term cannot be empty or whitespace only")
        
        doc_ids = irdata.inverse_doc_by_term.get(normalized_term, [])
        
        return InvertedFileByTermResponse(term=normalized_term, docs=doc_ids)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error while retrieving posting list: {str(e)}")

@router.get("/document/{doc_id}", response_model=InvertedFileByDocIdResponse)
async def get_document_terms_and_positions(
    doc_id: int = Path(..., description="Document ID to retrieve term positions for", ge=1)
):
    """
    Retrieve all terms and their positions within a specific document.
    
    Args:
        doc_id (int): Document ID to retrieve term positions for
        
    Returns:
        InvertedFileByDocIdResponse: Document ID and mapping of terms to their positions (empty dict if document not found)
        
    Raises:
        HTTPException: 500 if internal server error occurs
    """
    try:
        irdata: IRData = get_irdata()
        
        terms = irdata.inverse_doc_by_id.get(doc_id, {})
        
        return InvertedFileByDocIdResponse(
            doc_id=doc_id,
            term_postings=terms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error while retrieving document terms: {str(e)}")