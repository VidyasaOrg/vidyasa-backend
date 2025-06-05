from pydantic import BaseModel
from typing import Dict, List, Any

class InvertedFileByDocIdResponse(BaseModel):
    """
    Response model for inverted file lookup by term.

    Attributes:
        term (str): The term.
        docs (List[Dict]): List of dicts with doc_id, document_preview, and weight.

    Example:
    {
        "term": "information",
        "docs": [
            {
                "doc_id": 1,
                "document_preview": "What is information retrieval?...",
                "weight": 2
            }
        ]
    }
    """
    doc_id: int
    term_postings: Dict[str, Dict[str, Any]] = {}
    document_preview: str = ""
    total_terms: int = 0

class InvertedFileByTermResponse(BaseModel):
    """
    Response model for inverted file lookup by term.

    Attributes:
        term (str): The term.
        docs (List[Dict]): List of dicts with doc_id, document_preview, and weight.

    Example:
    {
        "term": "information",
        "docs": [
            {
                "doc_id": 1,
                "document_preview": "What is information retrieval?...",
                "weight": 2
            }
        ]
    }
    """
    term: str
    docs: List[Dict[str, Any]] = []