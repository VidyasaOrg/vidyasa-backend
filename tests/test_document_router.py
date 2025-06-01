"""Test cases for document router endpoints"""
import pytest
from unittest.mock import patch


class TestDocumentRouter:
    """Test cases for document retrieval endpoints"""
    
    @patch('app.routers.document.get_irdata')
    def test_get_documents_success(self, mock_get_irdata, client, mock_irdata):
        """Test successful retrieval of documents by IDs"""
        mock_get_irdata.return_value = mock_irdata
        
        response = client.get("/documents/?ids=1,2")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["documents"]) == 2
        assert len(data["not_found"]) == 0
        
        # Check first document
        doc1 = data["documents"][0]
        assert doc1["doc_id"] == 1
        assert doc1["title"] == "Information Retrieval Systems"
    
    @patch('app.routers.document.get_irdata')
    def test_get_documents_not_found(self, mock_get_irdata, client, mock_irdata):
        """Test when documents are not found"""
        mock_get_irdata.return_value = mock_irdata
        
        response = client.get("/documents/?ids=999,888")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["documents"]) == 0
        assert len(data["not_found"]) == 2
    
    @patch('app.routers.document.get_irdata')
    def test_get_documents_service_error(self, mock_get_irdata, client):
        """Test handling of service errors"""
        mock_get_irdata.side_effect = Exception("Service error")
        
        with pytest.raises(Exception):
            client.get("/documents/?ids=1,2")
