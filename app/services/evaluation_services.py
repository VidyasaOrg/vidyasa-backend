from typing import List, Optional, Set


class EvaluationService:
    """Service for evaluating IR system performance using metrics like MAP."""
    
    def calculate_precision_at_k(self, ranking: List[int], relevant_docs: Set[int], k: Optional[int] = None) -> float:
        """
        Calculate precision@k for a ranking.
        
        Args:
            ranking (List[int]): List of document IDs in ranked order
            relevant_docs (Set[int]): Set of relevant document IDs
            k (Optional[int]): Calculate precision at this position. If None, use full ranking length.
            
        Returns:
            float: Precision@k score
        """
        if not ranking or not relevant_docs:
            return 0.0
            
        if k is None:
            k = len(ranking)
        
        # Get top k results
        top_k = ranking[:k]
        
        # Count relevant docs in top k
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)
        
        # Calculate precision
        return relevant_in_top_k / k if k > 0 else 0.0

    def calculate_average_precision(self, ranking: List[int], relevant_docs: Set[int]) -> float:
        """
        Calculate Average Precision (AP) for a ranking.
        
        AP is the average of precision values calculated at each relevant document position.
        
        Args:
            ranking (List[int]): List of document IDs in ranked order
            relevant_docs (Set[int]): Set of relevant document IDs
            
        Returns:
            float: Average Precision score
        """
        if not ranking or not relevant_docs:
            return 0.0
            
        precisions = []
        num_relevant = 0
        
        # Calculate precision at each position where a relevant doc is found
        for k, doc_id in enumerate(ranking, start=1):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_at_k = self.calculate_precision_at_k(ranking, relevant_docs, k)
                precisions.append(precision_at_k)
        
        # Calculate average precision
        if not precisions:
            return 0.0
            
        return sum(precisions) / len(relevant_docs)

    def calculate_map_score(self, ranking: List[int], relevant_docs: Optional[Set[int]], is_queries_from_cisi: bool = False) -> float:
        """
        Calculate Mean Average Precision (MAP) for a single query.
        
        Since this is for a single query, it's equivalent to Average Precision.
        For multiple queries, you would average the AP scores across all queries.
        
        Args:
            ranking (List[int]): List of document IDs in ranked order
            relevant_docs (Optional[Set[int]]): Set of relevant document IDs, or None if query not found
            is_queries_from_cisi (bool): Whether the query is from CISI dataset
            
        Returns:
            float: MAP score (equivalent to AP for single query) if query found and is_queries_from_cisi is True,
                  -1.0 otherwise
        """
        # Return -1.0 if query not found or not from CISI
        if not is_queries_from_cisi or relevant_docs is None:
            return -1.0

        temp = self.calculate_average_precision(ranking, relevant_docs)
        return temp

