from pydantic import BaseModel
from typing import List

class DocumentResponse(BaseModel):
    """
    Response model for document retrieval.
    
    Attributes:
        doc_id (int): Document ID
        title (str): Document title
        author (str): Document author
        content (str): Document content/abstract
    """
    doc_id: int
    title: str
    author: str
    content: str
    
class DocumentBatchResponse(BaseModel):
    """
    Response model for batch document retrieval.
    
    Attributes:
        documents (List[DocumentResponse]): List of retrieved documents
        not_found (List[int]): List of document IDs that were not found
    """
    documents: List[DocumentResponse] = []
    not_found: List[int] = []