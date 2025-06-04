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
    
class TermWeightingMethod(str, Enum):
    """
    Enumeration of term weighting methods used in information retrieval.
    
    This enum defines different methods for calculating term weights in document scoring:
    - TF: Term Frequency
    - IDF: Inverse Document Frequency
    - TF_IDF: TF . IDF (product of term frequency and inverse document frequency)
    """
    TF = "tf" 
    IDF = "idf"
    TF_IDF = "tf_idf"
    
    def __str__(self) :
        return self.value

class QueryRequest(BaseModel):
    """
    Search query request.

    Attributes:
        query (str): Query text.
        query_id (Optional[int]): Query ID for CISI dataset queries. Default: None.
        is_stemming (bool): Apply stemming. Default: False.
        is_stop_words_removal (bool): Remove stop words. Default: False.
        query_term_frequency_method (TermFrequencyMethod): Query term frequency method. Default: RAW.
        query_term_weighting_method (TermWeightingMethod): Query term weighting method. Default: TF.
        document_term_frequency_method (TermFrequencyMethod): Document term frequency method. Default: RAW.
        document_term_weighting_method (TermWeightingMethod): Document term weighting method. Default: TF.
        cosine_normalization_query (bool): Whether to apply cosine normalization to query vectors. Default: False.
        cosine_normalization_document (bool): Whether to apply cosine normalization to document vectors. Default: False.
        expansion_terms_count (Union[int, Literal["all"]]): Number of expansion terms. Default: "all".
        is_queries_from_cisi (bool): Indicates if the query is from CISI queries. Default: False.

    Example in json:
    ```
    {
        "query": "information retrieval",
        "query_id": 1,
        "is_stemming": true,
        "is_stop_words_removal": false,
        "query_term_frequency_method": "log",
        "query_term_weighting_method": "tf_idf",
        "document_term_frequency_method": "raw",
        "document_term_weighting_method": "tf",
        "cosine_normalization_query": false,
        "cosine_normalization_document": false,
        "expansion_terms_count": 5,
        "is_queries_from_cisi": true
    }
    ```
    """
    query: str
    query_id: Optional[int] = None
    is_stemming: bool = False 
    is_stop_words_removal: bool = False 
    query_term_frequency_method: TermFrequencyMethod = TermFrequencyMethod.RAW
    query_term_weighting_method: TermWeightingMethod = TermWeightingMethod.TF
    document_term_frequency_method: TermFrequencyMethod = TermFrequencyMethod.RAW
    document_term_weighting_method: TermWeightingMethod = TermWeightingMethod.TF
    cosine_normalization_query: bool = False 
    cosine_normalization_document: bool = False
    expansion_terms_count: Union[int, Literal["all"]] = "all"
    is_queries_from_cisi: bool = False

class DocumentSimilarityScore(BaseModel):
    """Document with similarity score."""
    doc_id: int
    doc_title: str = ""
    similarity_score: float

class QueryResponse(BaseModel):
    """
    Response model for a query operation.
    
    Attributes:
        original_ranking (List[DocumentSimilarityScore]): Ranking of documents for the original query.
        expanded_ranking (List[DocumentSimilarityScore]): Ranking of documents for the expanded query.

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
            {"doc_id": 1, "similarity_score": 0.85},
            {"doc_id": 2, "similarity_score": 0.75}
        ],
        "expanded_ranking": [
            {"doc_id": 1, "similarity_score": 0.90},
            {"doc_id": 2, "similarity_score": 0.80}
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
    original_ranking: List[DocumentSimilarityScore]
    expanded_ranking: List[DocumentSimilarityScore]
    
    original_query: str
    original_map_score: float = 0.0
    original_query_weights: Dict[str, float] 
    
    expanded_query: str
    expanded_map_score: float = 0.0
    expanded_query_weights: Dict[str, float]

class QueryBatchResponse(BaseModel):
    """
    Response model for batch query processing.
    
    Attributes:
        results (List[QueryResponse]): List of query processing results, one for each query in the batch
    """
    results: List[QueryResponse]

class QueryBatchRequest(BaseModel):
    """
    Batch search query request.

    Attributes:
        queries (List[str]): List of query texts, one per line.
        is_stemming (bool): Apply stemming. Default: False.
        is_stop_words_removal (bool): Remove stop words. Default: False.
        query_term_frequency_method (TermFrequencyMethod): Term frequency method. Default: RAW.
        query_term_weighting_method (TermWeightingMethod): Term weighting method. Default: TF.
        document_term_frequency_method (TermFrequencyMethod): Term frequency method for documents. Default: RAW.
        document_term_weighting_method (TermWeightingMethod): Term weighting method for documents. Default: TF.
        cosine_normalization_query (bool): Whether to apply cosine normalization to query vectors. Default: False.
        cosine_normalization_document (bool): Whether to apply cosine normalization to document vectors. Default: False.
        expansion_terms_count (Union[int, Literal["all"]]): Number of expansion terms. Default: "all".
        is_queries_from_cisi (bool): Indicates if the queries are from CISI queries. Default: False.

    Example in json:
    ```
    {
        "queries": [
            "information retrieval",
            "database systems",
            "artificial intelligence"
        ],
        "is_stemming": true,
        "is_stop_words_removal": false,
        "query_term_frequency_method": "log",
        "query_term_weighting_method": "tf_idf",
        "document_term_frequency_method": "raw",
        "document_term_weighting_method": "tf",
        "cosine_normalization_query": false,
        "cosine_normalization_document": false,
        "expansion_terms_count": 5,
        "is_queries_from_cisi": true
    }
    ```
    """
    queries: List[str]
    is_stemming: bool = False 
    is_stop_words_removal: bool = False 
    query_term_frequency_method: TermFrequencyMethod = TermFrequencyMethod.RAW
    query_term_weighting_method: TermWeightingMethod = TermWeightingMethod.TF
    document_term_frequency_method: TermFrequencyMethod = TermFrequencyMethod.RAW
    document_term_weighting_method: TermWeightingMethod = TermWeightingMethod.TF
    cosine_normalization_query: bool = False 
    cosine_normalization_document: bool = False
    expansion_terms_count: Union[int, Literal["all"]] = "all"
    is_queries_from_cisi: bool = False
