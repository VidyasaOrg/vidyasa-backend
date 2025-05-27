from fastapi import APIRouter, HTTPException

from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/", response_model=QueryResponse)
async def search_single_query(request: QueryRequest):
    # TODO: Implementasi ini, kalau (bebas mau nama fungsi, input fungsi diubah kalau belum sesuai)
    # @satrianababan
    
    raise HTTPException(status_code=400, detail="not implemented yet")