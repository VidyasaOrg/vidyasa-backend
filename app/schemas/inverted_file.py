from pydantic import BaseModel
from typing import Dict, List, Any

class InvertedFileByDocIdResponse(BaseModel):
    """
    Response model for inverted file lookup by document ID.

    Attributes:
        doc_id (int): The document ID.
        term_postings (Dict[str, Dict[str, Any]]): Dictionary mapping terms to their positions and weights.
        document_preview (str): Preview of the document content.
        document_length (int): Total number of terms in the document (including stopwords).
        unique_terms (int): Number of unique terms in the document (excluding stopwords).

    Example:
    {
        "doc_id": 1,
        "term_postings": {
            "information": {
                "positions": [0, 4],
                "weight": 2.5
            }
        },
        "document_preview": "What is information retrieval?...",
        "document_length": 50,
        "unique_terms": 25
    }
    """
    doc_id: int
    term_postings: Dict[str, Dict[str, Any]] = {}
    document_preview: str = ""
    document_length: int = 0
    unique_terms: int = 0

class InvertedFileByTermResponse(BaseModel):
    """
    Response model for inverted file lookup by term.

    Attributes:
        term (str): The term.
        total_occurrences (int): Total number of times this term appears across all documents.
        total_documents (int): Total number of documents containing this term.
        docs (List[Dict]): List of dicts with doc_id, document_preview, and weight.

    Example:
    {
        "term": "information",
        "total_occurrences": 42,
        "total_documents": 15,
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
    total_occurrences: int = 0
    total_documents: int = 0
    docs: List[Dict[str, Any]] = []