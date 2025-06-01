"""Test cases for inverted file router endpoints"""
import pytest
from unittest.mock import patch


class TestInvertedFileRouter:
    """Test cases for inverted index endpoints"""
    
    @patch('app.routers.inverted_file.get_irdata')
    def test_get_posting_list_by_term_success(self, mock_get_irdata, client, mock_irdata):
        """Test successful retrieval of posting list by term"""
        mock_get_irdata.return_value = mock_irdata
        
        response = client.get("/inverted_file/term/information")
        assert response.status_code == 200
        data = response.json()
        
        assert data["term"] == "information"
        assert len(data["docs"]) == 1
    
    @patch('app.routers.inverted_file.get_irdata')
    def test_get_posting_list_term_not_found(self, mock_get_irdata, client, mock_irdata):
        """Test posting list for non-existing term"""
        mock_get_irdata.return_value = mock_irdata
        
        response = client.get("/inverted_file/term/nonexistent")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["term"] == "nonexistent"
        assert len(data["docs"]) == 0
    
    @patch('app.routers.inverted_file.get_irdata')
    def test_get_document_terms_success(self, mock_get_irdata, client, mock_irdata):
        """Test successful retrieval of document terms"""
        mock_get_irdata.return_value = mock_irdata
        
        response = client.get("/inverted_file/document/1")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["doc_id"] == 1
        assert "term_postings" in data
    
    @patch('app.routers.inverted_file.get_irdata')
    def test_get_document_terms_not_found(self, mock_get_irdata, client, mock_irdata):
        """Test document terms for non-existing document"""
        mock_get_irdata.return_value = mock_irdata
        
        response = client.get("/inverted_file/document/999")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["doc_id"] == 999
        assert data["term_postings"] == {}
    
    @patch('app.routers.inverted_file.get_irdata')
    def test_inverted_file_service_error(self, mock_get_irdata, client):
        """Test handling of service errors"""
        mock_get_irdata.side_effect = Exception("Service error")
        
        response = client.get("/inverted_file/term/test")
        
        assert response.status_code == 500
