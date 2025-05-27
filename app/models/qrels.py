from dataclasses import dataclass, field
from typing import List


@dataclass
class Qrels:
    """
    Maps query_id -> List[doc_id]
    
    Example of json structure:
        ```
        {
            "1": [1, 2, 3],
            "2": [4, 5],
            "3": [6]   
        }
        ```
    """
    data: dict[int, List[int]] = field(default_factory=dict)

    def add_qrel(self, query_id: int, doc_id: int):
        """Add a document to the relevance list for a given query_id."""
        if query_id not in self.data:
            self.data[query_id] = []
        if doc_id not in self.data[query_id]:
            self.data[query_id].append(doc_id)    

    def get_relevant_docs(self, query_id: int) -> List[int]:
        """Return docs with relevance >= min_relevance for a given query_id."""
        return self.data.get(query_id, [])

    def is_relevant(self, query_id: int, doc_id: int) -> bool:
        """Check if a document is relevant for a given query_id."""
        return doc_id in self.data.get(query_id, [])
    
    @staticmethod
    def from_json(data: dict) -> 'Qrels':
        """Create a Qrels from a JSON-like dictionary."""
        qrels = Qrels()
        for query_id, doc_ids in data.items():
            for doc_id in doc_ids:
                qrels.add_qrel(int(query_id), int(doc_id))
        return qrels
