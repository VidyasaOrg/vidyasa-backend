import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json

class MAPCalculator:
    """
    Kelas untuk menghitung Mean Average Precision (MAP) dan metrik evaluasi lainnya
    """
    
    def __init__(self):
        self.query_results = {}
        self.relevance_judgments = {}
    
    def load_relevance_judgments(self, relevance_file: str = None, 
                               relevance_data: Dict = None):
        """
        Load relevance judgments dari file atau data
        Format: {query_id: [list_of_relevant_doc_ids]}
        """
        if relevance_file:
            try:
                with open(relevance_file, 'r') as f:
                    self.relevance_judgments = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Relevance file {relevance_file} not found")
                self.relevance_judgments = {}
        elif relevance_data:
            self.relevance_judgments = relevance_data
        else:
            # Default: assume all documents are relevant for demo purposes
            self.relevance_judgments = {}
    
    def calculate_precision_at_k(self, retrieved_docs: List[int], 
                               relevant_docs: List[int], k: int) -> float:
        """
        Menghitung Precision@K
        """
        if k == 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_retrieved = len([doc for doc in top_k_docs if doc in relevant_docs])
        
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, retrieved_docs: List[int], 
                            relevant_docs: List[int], k: int) -> float:
        """
        Menghitung Recall@K
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_retrieved = len([doc for doc in top_k_docs if doc in relevant_docs])
        
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_average_precision(self, retrieved_docs: List[int], 
                                  relevant_docs: List[int]) -> float:
        """
        Menghitung Average Precision untuk satu query
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        precision_values = []
        relevant_retrieved = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                precision_values.append(precision_at_i)
        
        if len(precision_values) == 0:
            return 0.0
        
        return sum(precision_values) / len(relevant_docs)
    
    def calculate_mean_average_precision(self, query_results: Dict[str, List[int]], 
                                       relevance_judgments: Dict[str, List[int]] = None) -> Tuple[float, Dict]:
        """
        Menghitung MAP untuk semua query
        
        Args:
            query_results: {query_id: [ranked_list_of_doc_ids]}
            relevance_judgments: {query_id: [list_of_relevant_doc_ids]}
        
        Returns:
            MAP score dan detail per query
        """
        if relevance_judgments is None:
            relevance_judgments = self.relevance_judgments
        
        ap_scores = {}
        total_queries = 0
        
        for query_id, retrieved_docs in query_results.items():
            if query_id in relevance_judgments:
                relevant_docs = relevance_judgments[query_id]
                ap = self.calculate_average_precision(retrieved_docs, relevant_docs)
                ap_scores[query_id] = {
                    'average_precision': ap,
                    'relevant_docs': len(relevant_docs),
                    'retrieved_docs': len(retrieved_docs)
                }
                total_queries += 1
        
        if total_queries == 0:
            return 0.0, {}
        
        map_score = sum(score['average_precision'] for score in ap_scores.values()) / total_queries
        
        return map_score, ap_scores
    
    def calculate_precision_recall_curve(self, retrieved_docs: List[int], 
                                       relevant_docs: List[int]) -> Tuple[List[float], List[float]]:
        """
        Menghitung precision-recall curve
        """
        if len(relevant_docs) == 0:
            return [0], [0]
        
        precisions = []
        recalls = []
        relevant_retrieved = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_retrieved += 1
            
            precision = relevant_retrieved / (i + 1)
            recall = relevant_retrieved / len(relevant_docs)
            
            precisions.append(precision)
            recalls.append(recall)
        
        return precisions, recalls
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Menghitung F1 Score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_ranking(self, retrieved_docs: List[Tuple[int, float]], 
                        relevant_docs: List[int], 
                        k_values: List[int] = [1, 5, 10, 20]) -> Dict:
        """
        Evaluasi lengkap untuk ranking dokumen
        
        Args:
            retrieved_docs: List of (doc_id, similarity_score) tuples, sorted by score
            relevant_docs: List of relevant document IDs
            k_values: List of k values for Precision@K and Recall@K
        
        Returns:
            Dictionary dengan berbagai metrik evaluasi
        """
        doc_ids = [doc_id for doc_id, _ in retrieved_docs]
        
        # Hitung Average Precision
        ap = self.calculate_average_precision(doc_ids, relevant_docs)
        
        # Hitung Precision@K dan Recall@K
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        
        for k in k_values:
            p_at_k = self.calculate_precision_at_k(doc_ids, relevant_docs, k)
            r_at_k = self.calculate_recall_at_k(doc_ids, relevant_docs, k)
            f1_at_k[k] = self.calculate_f1_score(p_at_k, r_at_k)
            precision_at_k[k] = p_at_k
            recall_at_k[k] = r_at_k
        
        # Hitung Precision-Recall Curve
        precisions, recalls = self.calculate_precision_recall_curve(doc_ids, relevant_docs)
        
        return {
            'average_precision': ap,
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'f1_at_k': f1_at_k,
            'precision_recall_curve': {
                'precisions': precisions,
                'recalls': recalls
            },
            'total_relevant': len(relevant_docs),
            'total_retrieved': len(retrieved_docs),
            'relevant_retrieved': len([doc for doc in doc_ids if doc in relevant_docs])
        }
    
    def compare_rankings(self, original_results: Dict[str, List[Tuple[int, float]]], 
                        expanded_results: Dict[str, List[Tuple[int, float]]], 
                        relevance_judgments: Dict[str, List[int]] = None) -> Dict:
        """
        Membandingkan hasil ranking antara query asli dan query yang sudah di-expand
        """
        if relevance_judgments is None:
            relevance_judgments = self.relevance_judgments
        
        comparison = {
            'original': {},
            'expanded': {},
            'improvement': {}
        }
        
        # Evaluasi query asli
        original_query_results = {qid: [doc_id for doc_id, _ in results] 
                                for qid, results in original_results.items()}
        original_map, original_details = self.calculate_mean_average_precision(
            original_query_results, relevance_judgments)
        
        # Evaluasi query expanded
        expanded_query_results = {qid: [doc_id for doc_id, _ in results] 
                                for qid, results in expanded_results.items()}
        expanded_map, expanded_details = self.calculate_mean_average_precision(
            expanded_query_results, relevance_judgments)
        
        comparison['original']['map'] = original_map
        comparison['original']['details'] = original_details
        comparison['expanded']['map'] = expanded_map
        comparison['expanded']['details'] = expanded_details
        
        # Hitung improvement
        comparison['improvement']['map_improvement'] = expanded_map - original_map
        comparison['improvement']['percentage_improvement'] = (
            ((expanded_map - original_map) / original_map * 100) if original_map > 0 else 0
        )
        
        # Detail improvement per query
        query_improvements = {}
        for qid in original_details.keys():
            if qid in expanded_details:
                orig_ap = original_details[qid]['average_precision']
                exp_ap = expanded_details[qid]['average_precision']
                query_improvements[qid] = {
                    'original_ap': orig_ap,
                    'expanded_ap': exp_ap,
                    'improvement': exp_ap - orig_ap,
                    'percentage_improvement': ((exp_ap - orig_ap) / orig_ap * 100) if orig_ap > 0 else 0
                }
        
        comparison['improvement']['per_query'] = query_improvements
        
        return comparison
    
    def save_results(self, results: Dict, filename: str):
        """
        Menyimpan hasil evaluasi ke file JSON
        """
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def load_results(self, filename: str) -> Dict:
        """
        Load hasil evaluasi dari file JSON
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found")
            return {}
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
    
    def print_evaluation_summary(self, evaluation_results: Dict):
        """
        Print ringkasan hasil evaluasi
        """
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if 'original' in evaluation_results and 'expanded' in evaluation_results:
            # Comparison mode
            print(f"Original Query MAP: {evaluation_results['original']['map']:.4f}")
            print(f"Expanded Query MAP: {evaluation_results['expanded']['map']:.4f}")
            print(f"MAP Improvement: {evaluation_results['improvement']['map_improvement']:.4f}")
            print(f"Percentage Improvement: {evaluation_results['improvement']['percentage_improvement']:.2f}%")
        else:
            # Single evaluation mode
            if 'map' in evaluation_results:
                print(f"MAP Score: {evaluation_results['map']:.4f}")
            
            if 'precision_at_k' in evaluation_results:
                print("\nPrecision@K:")
                for k, precision in evaluation_results['precision_at_k'].items():
                    print(f"  P@{k}: {precision:.4f}")
            
            if 'recall_at_k' in evaluation_results:
                print("\nRecall@K:")
                for k, recall in evaluation_results['recall_at_k'].items():
                    print(f"  R@{k}: {recall:.4f}")