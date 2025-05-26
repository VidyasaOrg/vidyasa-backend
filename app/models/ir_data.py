from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Document:
    """
    Represents a document in the information retrieval system.
    Attributes:
        id (int): Unique identifier for the document.
        title (str): Title of the document.
        author (str): Author of the document.
        content (str): Main content of the document.
        tokens (List[str]): Tokenized content of the document.
        raw_tf (Dict[str, int]): Raw term frequency for terms in the document.
        log_tf (Dict[str, float]): Logarithmic term frequency for terms in the document.
        aug_tf (Dict[str, float]): Augmented term frequency for terms in the document.
        binary_tf (Dict[str, int]): Binary term frequency for terms in the document.
        
    Example JSON input:
    ```
        {
            "id": 1,
            "title": "Information Retrieval",
            "author": "J. Doe",
            "content": "What is information retrieval?",
            "tokenized_content": ["what", "is", "information", "retrieval"],
            "raw_tf": {"what": 1, "is": 1, "information": 1, "retrieval": 1}
        }
    ```
    """
    id: int
    title: str
    author: str
    content: str
    tokens: List[str] = field(default_factory=list)
    raw_tf: Dict[str, int] = field(default_factory=dict)
    log_tf: Dict[str, float] = field(default_factory=dict)
    aug_tf: Dict[str, float] = field(default_factory=dict)
    binary_tf: Dict[str, int] = field(default_factory=dict)
    
    @staticmethod
    def from_json(data: dict) -> "Document":
        """
        Create a Document object from a JSON-like dictionary.

        Args:
            data (dict): Dictionary containing document data.

        Returns:
            Document: An instance of the Document class.
        """
        # minimum there are raw_tf and id fields
        
        return Document(
            id=int(data["id"]),
            title=data.get("title", ""),
            author=data.get("author", ""),
            content=data.get("content", ""),
            tokens=data.get("tokenized_content", []),
            raw_tf=data.get("raw_tf", {}),
            log_tf=data.get("log_tf", {}),
            aug_tf=data.get("aug_tf", {}),
            binary_tf=data.get("binary_tf", {}),
        )
    


@dataclass
class IRData:
    """
    Data class for storing information retrieval (IR) data loaded from a JSON file.

    Attributes:
        documents (List[Document]): List of document objects.
        idf (Dict[str, float]): Inverse document frequency values for terms.
        inverse_doc_by_term (Dict[str, List[int]]): Inverted index mapping terms to document indices.
        inverse_doc_by_id (Dict[int, Dict[str, List[int]]]): Inverted index mapping document indices to term positions.

    Example JSON input:
    ```
        {
            "docs": [
                {
                    "id": 1,
                    "title": "Information Retrieval",
                    "author": "J. Doe",
                    "content": "What is information retrieval?",
                    "tokenized_content": ["what", "is", "information", "retrieval"],
                    "raw_tf": {"what": 1, "is": 1, "information": 1, "retrieval": 1}
                }
            ],
            "idf": {"information": 1.0, "retrieval": 1.0},
            "inverted_index_by_term": {"information": [0], "retrieval": [0]},
            "inverted_index_by_doc": {0: {"information": [0], "retrieval": [3]}}
        }
    ```
    """

    documents: List[Document] = field(default_factory=list)
    idf: Dict[str, float] = field(default_factory=dict)
    inverse_doc_by_term: Dict[str, List[int]] = field(default_factory=dict)
    inverse_doc_by_id: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)

    @staticmethod
    def from_json(data: dict) -> "IRData":
        """
        Create an IRData instance from a dictionary loaded from JSON.

        Args:
            data (dict): Dictionary containing IR data, typically loaded from a JSON file.

        Returns:
            IRData: An instance of IRData populated with documents and index data.
        """
        documents = []
        for doc in data.get("docs", []):
            # Only required fields are passed; others use default values
            documents.append(
                Document(
                    id=doc.get("id"),
                    title=doc.get("title"),
                    author=doc.get("author"),
                    content=doc.get("content"),
                    tokens=doc.get("tokenized_content", []),
                    raw_tf=doc.get("raw_tf", {}),
                    log_tf=doc.get("log_tf", {}),
                    aug_tf=doc.get("aug_tf", {}),
                    binary_tf=doc.get("binary_tf", {}),
                )
            )
        return IRData(
            documents=documents,
            idf=data.get("idf", {}),
            inverse_doc_by_term=data.get("inverted_index_by_term", {}),
            inverse_doc_by_id=data.get("inverted_index_by_doc", {}),
        )