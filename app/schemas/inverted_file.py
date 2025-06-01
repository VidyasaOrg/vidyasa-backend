from pydantic import BaseModel
from typing import Dict, List

class InvertedFileByDocIdResponse(BaseModel):
    """
    Response model for inverted file lookup by document ID.

    Attributes:
        doc_id (int): Document ID.
        term_postings (Dict[str, List[int]]): Mapping from term to list of positions in the document.

    Example:
    {
        "doc_id": 1,
        "term_postings": {
            "information": [2, 10],
            "retrieval": [5]
        }
    }
    """
    doc_id: int
    term_postings: Dict[str, List[int]] = {}

class InvertedFileByTermResponse(BaseModel):
    """
    Response model for inverted file lookup by term.

    Attributes:
        term (str): The term.
        docs List[int]: List of document IDs where the term appears.

    Example:
    {
        "term": "information",
        "docs": [1, 2, 3]
    }
    """
    term: str
    docs: List[int] = []