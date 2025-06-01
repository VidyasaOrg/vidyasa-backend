from pydantic import BaseModel
from typing import List

class QrelsResponse(BaseModel):
    """
    Response model for relevance judgments.
    
    Attributes:
        query_id (int): Query ID
        relevant_docs (List[int]): List of relevant document IDs
    """
    query_id: int
    relevant_docs: List[int] = []