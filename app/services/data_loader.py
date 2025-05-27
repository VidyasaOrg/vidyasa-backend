import os
import json
import pickle
import math

from models.ir_data import IRData
from models.query import Query
from models.qrels import Qrels

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_PATH = os.path.join(BASE_PATH, "data", "preprocessed")

CISI_FILENAME = os.path.join(DATABASE_PATH, "cisi_all.pkl")
QRELS_FILENAME = os.path.join(DATABASE_PATH, "qrels.json")
QUERIES_FILENAME = os.path.join(DATABASE_PATH, "queries.json")

class DataLoader:
    """
    Usage:
    ```
    from services.data_loader import get_qrels, get_queries, get_irdata
    
    qrels: QrelsDict = get_qrels()
    queries: list[Query] = get_queries()
    irdata: IRData = get_irdata()
    
    qrels['1']  # Access relevance judgments for query ID '1'
    irdata.docs[0].log_tf  # Access log_tf for the first document
    queries[0].content # Access the text of the first query
    ```
    """
    _qrels_data = None
    _queries_data = None
    _cisi_data = None

    @classmethod
    def json_load(cls, file_path: str) -> dict:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)

    @classmethod
    def pickle_load(cls, file_path: str) -> dict:
        with open(file_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    @classmethod
    def get_qrels(cls) -> Qrels:
        if cls._qrels_data is None:
            qrels_data_temp = cls.json_load(QRELS_FILENAME)
            cls._qrels_data = Qrels.from_json(qrels_data_temp)
        return cls._qrels_data

    @classmethod
    def get_queries(cls) -> list[Query]:
        if cls._queries_data is None:
            queries_data_temp = cls.json_load(QUERIES_FILENAME)
            cls._queries_data = [Query.from_json(query) for query in queries_data_temp]
        return cls._queries_data

    @staticmethod
    def _calculate_log_tf(raw_tf: dict) -> dict:
        return {term: 1 + math.log2(tf) if tf > 0 else 0 for term, tf in raw_tf.items()}

    @staticmethod
    def _calculate_augmented_tf(raw_tf: dict) -> dict:
        max_tf = max(raw_tf.values(), default=0)
        if max_tf == 0:
            return {term: 0 for term in raw_tf.keys()}
        return {term: 0.5 + 0.5 * (tf / max_tf) for term, tf in raw_tf.items() if tf > 0}

    @staticmethod
    def _calculate_binary_tf(raw_tf: dict) -> dict:
        return {term: 1 if tf > 0 else 0 for term, tf in raw_tf.items()}

    @classmethod
    def get_irdata(cls) -> IRData:
        if cls._cisi_data is None:
            cisi_data_temp = cls.pickle_load(CISI_FILENAME)
            for doc in cisi_data_temp["docs"]:
                doc["log_tf"] = cls._calculate_log_tf(doc["raw_tf"])
                doc["aug_tf"] = cls._calculate_augmented_tf(doc["raw_tf"])
                doc["binary_tf"] = cls._calculate_binary_tf(doc["raw_tf"])
            cls._cisi_data = IRData.from_json(cisi_data_temp)
        return cls._cisi_data

get_qrels = DataLoader.get_qrels
get_queries = DataLoader.get_queries
get_irdata = DataLoader.get_irdata
