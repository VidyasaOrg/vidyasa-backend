from fastapi import APIRouter, HTTPException, Path

from app.schemas.inverted_file import InvertedFileByDocIdResponse, InvertedFileByTermResponse
from app.services.data_loader import get_irdata
from app.models.ir_data import IRData

router = APIRouter(prefix="/inverted_file", tags=["inverted_index"])

@router.get("/term/{term}", response_model=InvertedFileByTermResponse)
async def get_posting_list_by_term(
    term: str = Path(..., description="Term to lookup in inverted index", min_length=1)
):
    try:
        irdata: IRData = get_irdata()
        normalized_term = term.lower().strip()
        if not normalized_term:
            raise HTTPException(status_code=400, detail="Term cannot be empty or whitespace only")
        
        doc_ids = irdata.inverse_doc_by_term.get(normalized_term, [])
        docs = []
        for doc_id in doc_ids:
            doc = next((d for d in irdata.documents if d.id == doc_id), None)
            if doc:
                document_preview = (doc.content[:100] + "...") if len(doc.content) > 100 else doc.content
                weight = doc.raw_tf.get(normalized_term, 0)  
                docs.append({
                    "doc_id": doc_id,
                    "document_preview": document_preview,
                    "weight": weight
                })

        # Sort docs by weight in descending order
        docs.sort(key=lambda x: x['weight'], reverse=True)

        return InvertedFileByTermResponse(term=normalized_term, docs=docs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error while retrieving posting list: {str(e)}")

@router.get("/document/{doc_id}", response_model=InvertedFileByDocIdResponse)
async def get_document_terms_and_positions(
    doc_id: int = Path(..., description="Document ID to retrieve term positions for", ge=1)
):
    try:
        irdata: IRData = get_irdata()
        terms = irdata.inverse_doc_by_id.get(doc_id, {})
        doc = next((d for d in irdata.documents if d.id == doc_id), None)
        document_preview = (doc.content[:150] + "...") if doc and len(doc.content) > 100 else (doc.content if doc else "")
        total_terms = len(terms)

        term_postings = {}
        if doc:
            for term, positions in terms.items():
                raw_tf = doc.raw_tf.get(term, 0)
                idf = irdata.idf.get(term, 0)
                weight = raw_tf * idf
                term_postings[term] = {
                    "positions": positions,
                    "weight": weight
                }

        # Sort term_postings by weight in descending order
        term_postings = dict(sorted(term_postings.items(), key=lambda item: item[1]['weight'], reverse=True))

        return InvertedFileByDocIdResponse(
            doc_id=doc_id,
            term_postings=term_postings,
            document_preview=document_preview,
            total_terms=total_terms
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error while retrieving document terms: {str(e)}")