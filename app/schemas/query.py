from pydantic import BaseModel
from typing import List, Optional, Union, Literal, Dict
from enum import Enum

class TermFrequencyMethod(str, Enum):
    """
    Enumeration of term frequency calculation methods used in information retrieval.
    
    This enum defines different ways to calculate term frequency for document scoring:
    - RAW: Raw frequency count (number of occurrences)
    - LOG: Logarithmically scaled frequency (1 + log(tf))
    - BINARY: Binary representation (0 or 1)
    - AUGMENTED: Augmented frequency to prevent bias towards longer documents
    """ 

    RAW = "raw"
    LOG = "log"
    BINARY = "binary"
    AUGMENTED = "augmented"
    
    def __str__(self) :
        return self.value

class QueryRequest(BaseModel):
    """
    Search query request.

    Attributes:
        query (str): Query text.
        is_stemming (Optional[bool]): Apply stemming. Default: False.
        is_stop_words_removal (Optional[bool]): Remove stop words. Default: False.
        term_frequency_method (Optional[TermFrequencyMethod]): Term frequency method. Default: RAW.
        expansion_terms_count (Optional[Union[int, Literal["all"]]]): Number of expansion terms. Default: "all".

    Example in json:
    ```
    {
        "query": "information retrieval",
        "is_stemming": true,
        "is_stop_words_removal": false,
        "term_frequency_method": "log",
        "expansion_terms_count": 5
    }
    ```
    """
    query: str
    is_stemming: Optional[bool] = False 
    is_stop_words_removal: Optional[bool] = False 
    term_frequency_method: Optional[TermFrequencyMethod] = TermFrequencyMethod.RAW
    expansion_terms_count: Optional[Union[int, Literal["all"]]] = "all"
    
class DocumentSimiliarityScore(BaseModel):
    """Document with similarity score."""
    doc_id: int
    similiarity_score: float

class QueryResponse(BaseModel):
    """
    Response model for a query operation.
    
    Attributes:
        original_ranking (List[DocumentSimiliarityScore]): Ranking of documents for the original query.
        expanded_ranking (List[DocumentSimiliarityScore]): Ranking of documents for the expanded query.
        
        original_query (str): The original query text.
        original_map_score (float): Mean Average Precision score for the original query.
        original_query_weights (Dict[str, float]): Term weights for the original query.
        
        expanded_query (str): The expanded query text.
        expanded_map_score (float): Mean Average Precision score for the expanded query.
        expanded_query_weights (Dict[str, float]): Term weights for the expanded query.
        
    Example in json:
    ```
    {
        "original_ranking": [
            {"doc_id": 1, "similiarity_score": 0.85},
            {"doc_id": 2, "similiarity_score": 0.75}
        ],
        "expanded_ranking": [
            {"doc_id": 1, "similiarity_score": 0.90},
            {"doc_id": 2, "similiarity_score": 0.80}
        ],
        "original_query": "information retrieval",
        "original_map_score": 0.78,
        "original_query_weights": {"information": 0.5, "retrieval": 0.5},
        "expanded_query": "information retrieval systems",
        "expanded_map_score": 0.82,
        "expanded_query_weights": {"information": 0.4, "retrieval": 0.4, "systems": 0.2}
    }
    ```
    """
    original_ranking: List[DocumentSimiliarityScore]
    expanded_ranking: List[DocumentSimiliarityScore]
    
    original_query: str
    original_map_score: float
    original_query_weights: Dict[str, float] 
    
    expanded_query: str
    expanded_map_score: float
    expanded_query_weights: Dict[str, float]
