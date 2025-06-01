"""Test cases for qrels router endpoints"""
import pytest
from unittest.mock import patch


class TestQrelsRouter:
    """Test cases for qrels retrieval endpoints"""
    
    @patch('app.routers.qrels.get_qrels')
    def test_get_relevant_docs_success(self, mock_get_qrels, client, mock_qrels):
        """Test successful retrieval of relevance judgments"""
        mock_get_qrels.return_value = mock_qrels
        
        response = client.get("/qrels/1")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["query_id"] == 1
        assert len(data["relevant_docs"]) == 2
    
    @patch('app.routers.qrels.get_qrels')
    def test_get_relevant_docs_not_found(self, mock_get_qrels, client, mock_qrels):
        """Test retrieval of relevance judgments for non-existing query"""
        mock_get_qrels.return_value = mock_qrels
        
        response = client.get("/qrels/999")
        
        assert response.status_code == 404
    
    @patch('app.routers.qrels.get_qrels')
    def test_get_relevant_docs_service_error(self, mock_get_qrels, client):
        """Test handling of service errors"""
        mock_get_qrels.side_effect = Exception("Service error")
        
        with pytest.raises(Exception):
            client.get("/qrels/1")
