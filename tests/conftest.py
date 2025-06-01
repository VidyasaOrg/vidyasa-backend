"""
Test configuration and fixtures for Vidyasa Backend tests.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
import tempfile
import os

from app.main import app
from app.models.ir_data import IRData, Document
from app.models.qrels import Qrels
from app.models.query import Query

@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)

@pytest.fixture
def mock_irdata():
    """Mock IRData for testing"""
    mock_doc1 = Document(
        id=1,
        title="Information Retrieval Systems",
        author="John Doe",
        content="This document discusses information retrieval systems and their applications.",
        tokens=["information", "retrieval", "systems", "applications"],
        raw_tf={"information": 2, "retrieval": 1, "systems": 1, "applications": 1},
        log_tf={"information": 2.0, "retrieval": 1.0, "systems": 1.0, "applications": 1.0},
        aug_tf={"information": 1.0, "retrieval": 0.75, "systems": 0.75, "applications": 0.75},
        binary_tf={"information": 1, "retrieval": 1, "systems": 1, "applications": 1}
    )
    
    mock_doc2 = Document(
        id=2,
        title="Database Management",
        author="Jane Smith",
        content="Database systems and management techniques for large datasets.",
        tokens=["database", "systems", "management", "techniques"],
        raw_tf={"database": 1, "systems": 1, "management": 2, "techniques": 1},
        log_tf={"database": 1.0, "systems": 1.0, "management": 2.0, "techniques": 1.0},
        aug_tf={"database": 0.75, "systems": 0.75, "management": 1.0, "techniques": 0.75},
        binary_tf={"database": 1, "systems": 1, "management": 1, "techniques": 1}
    )
    
    return IRData(
        documents=[mock_doc1, mock_doc2],
        idf={"information": 1.5, "retrieval": 2.0, "systems": 0.5, "database": 2.0, "management": 1.8},
        inverse_doc_by_term={
            "information": [1],
            "retrieval": [1],
            "systems": [1, 2],
            "database": [2],
            "management": [2]
        },
        inverse_doc_by_id={
            1: {"information": [0, 2], "retrieval": [1], "systems": [2]},
            2: {"database": [0], "systems": [1], "management": [2, 4]}
        }
    )

@pytest.fixture
def mock_qrels():
    """Mock Qrels for testing"""
    qrels = Qrels()
    qrels.add_qrel(1, 1)
    qrels.add_qrel(1, 2)
    qrels.add_qrel(2, 2)
    return qrels

@pytest.fixture
def mock_queries():
    """Mock queries for testing"""
    return [
        Query(id=1, title="IR Systems", content="information retrieval systems"),
        Query(id=2, title="Database", content="database management systems")
    ]