    import json
    import math
    from typing import Dict, List, Tuple, Set
    from collections import defaultdict

    class EvaluationMetrics:
        """
        Class untuk menghitung berbagai metrik evaluasi dalam IR
        """
        
        @staticmethod
        def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
            """Precision@K"""
            if k <= 0 or not retrieved:
                return 0.0
            
            top_k = retrieved[:k]
            relevant_retrieved = sum(1 for doc in top_k if doc in relevant)
            return relevant_retrieved / len(top_k)
        
        @staticmethod
        def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
            """Recall@K"""
            if not relevant or k <= 0:
                return 0.0
            
            top_k = retrieved[:k]
            relevant_retrieved = sum(1 for doc in top_k if doc in relevant)
            return relevant_retrieved / len(relevant)
        
        @staticmethod
        def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
            """F1@K"""
            precision = EvaluationMetrics.precision_at_k(retrieved, relevant, k)
            recall = EvaluationMetrics.recall_at_k(retrieved, relevant, k)
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * (precision * recall) / (precision + recall)
        
        @staticmethod
        def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
            """Average Precision untuk satu query"""
            if not relevant or not retrieved:
                return 0.0
            
            precision_sum = 0.0
            relevant_count = 0
            
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    relevant_count += 1
                    precision_at_i = relevant_count / i
                    precision_sum += precision_at_i
            
            return precision_sum / len(relevant)
        
        @staticmethod
        def mean_average_precision(results: Dict[str, List[str]], 
                                qrels: Dict[str, Set[str]]) -> Tuple[float, Dict[str, float]]:
            """Mean Average Precision"""
            individual_aps = {}
            total_ap = 0.0
            valid_queries = 0
            
            for query_id, retrieved_docs in results.items():
                relevant_docs = qrels.get(query_id, set())
                ap = EvaluationMetrics.average_precision(retrieved_docs, relevant_docs)
                individual_aps[query_id] = ap
                total_ap += ap
                valid_queries += 1
            
            overall_map = total_ap / valid_queries if valid_queries > 0 else 0.0
            return overall_map, individual_aps
        
        @staticmethod
        def ndcg_at_k(retrieved: List[str], relevant_grades: Dict[str, int], k: int) -> float:
            """Normalized Discounted Cumulative Gain@K"""
            if k <= 0 or not retrieved:
                return 0.0
            
            # DCG@K
            dcg = 0.0
            for i, doc in enumerate(retrieved[:k], 1):
                relevance = relevant_grades.get(doc, 0)
                if relevance > 0:
                    dcg += (2**relevance - 1) / math.log2(i + 1)
            
            # IDCG@K (Ideal DCG)
            ideal_relevances = sorted(relevant_grades.values(), reverse=True)[:k]
            idcg = 0.0
            for i, relevance in enumerate(ideal_relevances, 1):
                if relevance > 0:
                    idcg += (2**relevance - 1) / math.log2(i + 1)
            
            return dcg / idcg if idcg > 0 else 0.0
        
        @staticmethod
        def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
            """Reciprocal Rank - rank of first relevant document"""
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    return 1.0 / i
            return 0.0
        
        @staticmethod
        def mean_reciprocal_rank(results: Dict[str, List[str]], 
                            qrels: Dict[str, Set[str]]) -> Tuple[float, Dict