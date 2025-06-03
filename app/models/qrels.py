from typing import Dict, Set, List, Tuple, Union, Optional
from dataclasses import dataclass, field
import json

@dataclass
class QRels:
    """
    Class untuk menangani relevance judgments (qrels) 
    dalam format standar TREC dengan dukungan multiple format dan tipe data fleksibel
    
    Mendukung:
    - String dan integer untuk query_id dan doc_id
    - Multiple relevance levels (0, 1, 2, 3, dst.)
    - Binary relevance (relevant/non-relevant)
    - Multiple format file (TREC, JSON, simple)
    """
    
    # Main qrels data: {query_id: {doc_id: relevance_level}}
    qrels: Dict[Union[str, int], Dict[Union[str, int], int]] = field(default_factory=dict)
    
    # Binary qrels untuk akses cepat: {query_id: set(relevant_doc_ids)}
    binary_qrels: Dict[Union[str, int], Set[Union[str, int]]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize binary_qrels berdasarkan qrels yang ada"""
        self._update_binary_qrels()
    
    def _update_binary_qrels(self):
        """Update binary_qrels berdasarkan qrels"""
        self.binary_qrels = {}
        for query_id, doc_dict in self.qrels.items():
            self.binary_qrels[query_id] = set()
            for doc_id, relevance in doc_dict.items():
                if relevance > 0:
                    self.binary_qrels[query_id].add(doc_id)
    
    def _normalize_id(self, id_value: Union[str, int]) -> Union[str, int]:
        """Normalize ID untuk konsistensi (bisa disesuaikan kebutuhan)"""
        return id_value
    
    def load_qrels_from_file(self, filepath: str, format_type: str = 'trec'):
        """
        Load qrels dari file
        format_type: 'trec', 'json', 'simple'
        """
        if format_type == 'trec':
            self._load_trec_format(filepath)
        elif format_type == 'json':
            self._load_json_format(filepath)
        elif format_type == 'simple':
            self._load_simple_format(filepath)
        else:
            raise ValueError("Format tidak didukung. Gunakan 'trec', 'json', atau 'simple'")
    
    def _load_trec_format(self, filepath: str):
        """Load qrels dalam format TREC: query_id 0 doc_id relevance"""
        self.qrels = {}
        self.binary_qrels = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id = self._normalize_id(parts[0])
                    doc_id = self._normalize_id(parts[2])
                    relevance = int(parts[3])
                    
                    if query_id not in self.qrels:
                        self.qrels[query_id] = {}
                        self.binary_qrels[query_id] = set()
                    
                    self.qrels[query_id][doc_id] = relevance
                    
                    # Binary relevance (>0 = relevant)
                    if relevance > 0:
                        self.binary_qrels[query_id].add(doc_id)
    
    def _load_json_format(self, filepath: str):
        """Load qrels dari file JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support both formats: direct qrels atau nested structure
        if 'qrels' in data:
            self.qrels = data['qrels']
        else:
            # Assume direct format like {"1": [1, 2, 3], "2": [4, 5]}
            self.qrels = {}
            for query_id, doc_list in data.items():
                query_id = self._normalize_id(query_id)
                self.qrels[query_id] = {}
                if isinstance(doc_list, list):
                    # List of relevant docs (binary relevance)
                    for doc_id in doc_list:
                        doc_id = self._normalize_id(doc_id)
                        self.qrels[query_id][doc_id] = 1
                elif isinstance(doc_list, dict):
                    # Dict of doc_id: relevance_level
                    for doc_id, relevance in doc_list.items():
                        doc_id = self._normalize_id(doc_id)
                        self.qrels[query_id][doc_id] = relevance
        
        self._update_binary_qrels()
    
    def _load_simple_format(self, filepath: str):
        """
        Load qrels dalam format sederhana:
        query_id:doc_id1,doc_id2,doc_id3
        """
        self.qrels = {}
        self.binary_qrels = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    parts = line.strip().split(':', 1)
                    query_id = self._normalize_id(parts[0].strip())
                    doc_ids = [self._normalize_id(doc.strip()) for doc in parts[1].split(',')]
                    
                    self.qrels[query_id] = {}
                    self.binary_qrels[query_id] = set()
                    
                    for doc_id in doc_ids:
                        if doc_id:  # Skip empty strings
                            self.qrels[query_id][doc_id] = 1
                            self.binary_qrels[query_id].add(doc_id)
    
    def add_qrel(self, query_id: Union[str, int], doc_id: Union[str, int], relevance: int = 1):
        """
        Tambah relevance judgment secara manual
        Kompatibel dengan method dari implementasi sebelumnya
        """
        query_id = self._normalize_id(query_id)
        doc_id = self._normalize_id(doc_id)
        
        if query_id not in self.qrels:
            self.qrels[query_id] = {}
            self.binary_qrels[query_id] = set()
        
        self.qrels[query_id][doc_id] = relevance
        
        if relevance > 0:
            self.binary_qrels[query_id].add(doc_id)
        elif doc_id in self.binary_qrels[query_id]:
            self.binary_qrels[query_id].remove(doc_id)
    
    def add_relevance_judgment(self, query_id: Union[str, int], doc_id: Union[str, int], relevance: int):
        """Alias untuk add_qrel untuk backward compatibility"""
        self.add_qrel(query_id, doc_id, relevance)
    
    def get_relevant_docs(self, query_id: Union[str, int], binary: bool = True) -> Union[Set[Union[str, int]], List[Union[str, int]]]:
        """
        Mendapatkan dokumen relevan untuk query tertentu
        Return Set jika binary=True, List jika binary=False (untuk backward compatibility)
        """
        query_id = self._normalize_id(query_id)
        
        if binary:
            result = self.binary_qrels.get(query_id, set())
            return list(result)  # Return list untuk kompatibilitas
        else:
            query_rels = self.qrels.get(query_id, {})
            return [doc_id for doc_id, rel in query_rels.items() if rel > 0]
    
    def get_relevance_level(self, query_id: Union[str, int], doc_id: Union[str, int]) -> int:
        """Mendapatkan tingkat relevansi dokumen untuk query tertentu"""
        query_id = self._normalize_id(query_id)
        doc_id = self._normalize_id(doc_id)
        return self.qrels.get(query_id, {}).get(doc_id, 0)
    
    def is_relevant(self, query_id: Union[str, int], doc_id: Union[str, int]) -> bool:
        """Check apakah dokumen relevan untuk query"""
        query_id = self._normalize_id(query_id)
        doc_id = self._normalize_id(doc_id)
        return doc_id in self.binary_qrels.get(query_id, set())
    
    def get_all_query_ids(self) -> List[Union[str, int]]:
        """Mendapatkan semua query ID"""
        return list(self.qrels.keys())
    
    def get_statistics(self) -> Dict[str, any]:
        """Mendapatkan statistik qrels"""
        stats = {
            'total_queries': len(self.qrels),
            'total_judgments': sum(len(docs) for docs in self.qrels.values()),
            'total_relevant': sum(len(docs) for docs in self.binary_qrels.values()),
            'query_stats': {}
        }
        
        for query_id in self.qrels:
            query_rels = self.qrels[query_id]
            relevant_count = len(self.binary_qrels[query_id])
            
            stats['query_stats'][query_id] = {
                'total_judgments': len(query_rels),
                'relevant_docs': relevant_count,
                'non_relevant_docs': len(query_rels) - relevant_count,
                'relevance_levels': {}
            }
            
            # Count by relevance level
            for rel_level in query_rels.values():
                if rel_level not in stats['query_stats'][query_id]['relevance_levels']:
                    stats['query_stats'][query_id]['relevance_levels'][rel_level] = 0
                stats['query_stats'][query_id]['relevance_levels'][rel_level] += 1
        
        return stats
    
    def save_qrels(self, filepath: str, format_type: str = 'trec'):
        """Simpan qrels ke file"""
        if format_type == 'trec':
            with open(filepath, 'w', encoding='utf-8') as f:
                for query_id, doc_dict in self.qrels.items():
                    for doc_id, relevance in doc_dict.items():
                        f.write(f"{query_id} 0 {doc_id} {relevance}\n")
        
        elif format_type == 'json':
            data = {
                'qrels': self.qrels,
                'binary_qrels': {qid: list(docs) for qid, docs in self.binary_qrels.items()}
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'simple':
            with open(filepath, 'w', encoding='utf-8') as f:
                for query_id, doc_set in self.binary_qrels.items():
                    if doc_set:
                        f.write(f"{query_id}:{','.join(map(str, sorted(doc_set)))}\n")
    
    def validate_qrels(self) -> Dict[str, List[str]]:
        """Validasi qrels dan return daftar masalah"""
        issues = {
            'empty_queries': [],
            'invalid_relevance': [],
            'inconsistent_data': []
        }
        
        for query_id, doc_dict in self.qrels.items():
            if not doc_dict:
                issues['empty_queries'].append(str(query_id))
            
            for doc_id, relevance in doc_dict.items():
                if not isinstance(relevance, int) or relevance < 0:
                    issues['invalid_relevance'].append(f"{query_id}:{doc_id}={relevance}")
                
                # Check consistency between qrels and binary_qrels
                is_relevant_qrels = relevance > 0
                is_relevant_binary = doc_id in self.binary_qrels.get(query_id, set())
                
                if is_relevant_qrels != is_relevant_binary:
                    issues['inconsistent_data'].append(f"{query_id}:{doc_id}")
        
        return issues
    
    def create_sample_qrels(self, queries: List[Union[str, int]], documents: List[Union[str, int]], 
                          relevance_ratio: float = 0.3) -> None:
        """
        Membuat sample qrels untuk testing
        relevance_ratio: proporsi dokumen yang dianggap relevan
        """
        import random
        
        self.qrels = {}
        self.binary_qrels = {}
        
        for query_id in queries:
            query_id = self._normalize_id(query_id)
            self.qrels[query_id] = {}
            self.binary_qrels[query_id] = set()
            
            # Randomly select relevant documents
            num_relevant = max(1, int(len(documents) * relevance_ratio))
            relevant_docs = random.sample(documents, num_relevant)
            
            for doc_id in documents:
                doc_id = self._normalize_id(doc_id)
                if doc_id in relevant_docs:
                    # Random relevance level 1-3 for relevant docs
                    relevance = random.randint(1, 3)
                    self.qrels[query_id][doc_id] = relevance
                    self.binary_qrels[query_id].add(doc_id)
                else:
                    # Non-relevant
                    self.qrels[query_id][doc_id] = 0
    
    def display_qrels_summary(self) -> str:
        """Display summary of qrels"""
        stats = self.get_statistics()
        
        output = "\n=== QRELS Summary ===\n"
        output += f"Total Queries: {stats['total_queries']}\n"
        output += f"Total Judgments: {stats['total_judgments']}\n"
        output += f"Total Relevant Docs: {stats['total_relevant']}\n"
        
        if stats['total_queries'] > 0:
            output += f"Average Relevant per Query: {stats['total_relevant']/stats['total_queries']:.2f}\n\n"
        
        output += "Per Query Statistics:\n"
        output += "-" * 50 + "\n"
        
        for query_id, query_stats in stats['query_stats'].items():
            output += f"Query {query_id}:\n"
            output += f"  Total judgments: {query_stats['total_judgments']}\n"
            output += f"  Relevant docs: {query_stats['relevant_docs']}\n"
            output += f"  Non-relevant docs: {query_stats['non_relevant_docs']}\n"
            output += f"  Relevance levels: {query_stats['relevance_levels']}\n\n"
        
        return output
    
    @classmethod
    def from_json(cls, data: dict) -> 'QRels':
        """
        Create a QRels from a JSON-like dictionary
        Kompatibel dengan implementasi sebelumnya
        """
        qrels = cls()
        
        # Handle different JSON formats
        if isinstance(list(data.values())[0], list):
            # Format: {"1": [1, 2, 3], "2": [4, 5]}
            for query_id, doc_ids in data.items():
                for doc_id in doc_ids:
                    qrels.add_qrel(query_id, doc_id, 1)
        elif isinstance(list(data.values())[0], dict):
            # Format: {"1": {"doc1": 1, "doc2": 2}, "2": {"doc3": 1}}
            for query_id, doc_dict in data.items():
                for doc_id, relevance in doc_dict.items():
                    qrels.add_qrel(query_id, doc_id, relevance)
        
        return qrels
    
    @property
    def data(self) -> Dict[Union[str, int], List[Union[str, int]]]:
        """
        Property untuk backward compatibility dengan implementasi sebelumnya
        Return format: {query_id: [list_of_relevant_doc_ids]}
        """
        result = {}
        for query_id, doc_set in self.binary_qrels.items():
            result[query_id] = list(doc_set)
        return result