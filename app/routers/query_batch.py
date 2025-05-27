from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
\
router = APIRouter(prefix="/query_batch", tags=["batch_query"])

@router.post("/", response_class=FileResponse)
async def search_batch_queries(
    file: UploadFile = File(..., description="File containing queries (one per line or JSON format)"),
):
    # TODO: Implementasi ini, kalau (bebas mau nama fungsi, input fungsi diubah kalau belum sesuai)
    # @Breezy-DR
    
    raise HTTPException(status_code=400, detail="not implemented yet")

