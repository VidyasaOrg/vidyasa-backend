from fastapi import APIRouter, HTTPException, Path

from app.schemas.inverted_file import InvertedFileByDocIdResponse, InvertedFileByTermResponse

router = APIRouter(prefix="/inverted_file", tags=["inverted_index"])

@router.get("/term/{term}", response_model=InvertedFileByTermResponse)
async def get_posting_list_by_term(
    term: str = Path(..., description="Term to lookup in inverted index")
):
    # TODO: Implementasi ini, kalau (bebas mau nama fungsi, input fungsi diubah kalau belum sesuai)
    # @satrianababan
    # NOTE: udah ada ada di IRData
    raise HTTPException(status_code=400, detail="not implemented yet")

@router.get("/document/{doc_id}", response_model=InvertedFileByDocIdResponse)
async def get_document_terms_and_positions(
    doc_id: int = Path(..., description="Document ID to retrieve term positions for")
):
    
    # TODO: Implementasi ini, kalau (bebas mau nama fungsi, input fungsi diubah kalau belum sesuai)
    # @satrianababan
    # NOTE: udah ada ada di IRData
    raise HTTPException(status_code=400, detail="not implemented yet")

