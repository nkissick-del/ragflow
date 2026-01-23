import pytest
from unittest.mock import MagicMock, patch
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.db_models import DB
from common.constants import TaskStatus

@pytest.mark.p2
class TestKnowledgebasePerformance:
    """Test performance optimization for is_parsed_done."""

    @patch.object(KnowledgebaseService, 'query')
    @patch('api.db.services.document_service.DocumentService')
    @patch.object(DB, 'connect')
    @patch.object(DB, 'close')
    def test_is_parsed_done_optimized(self, mock_close, mock_connect, mock_doc_service, mock_query, sample_kb):
        """Test that is_parsed_done uses the optimized get_first_unparsed_document method."""
        # Arrange
        mock_query.return_value = [sample_kb]
        mock_doc_service.get_first_unparsed_document.return_value = None  # All done

        # Act
        is_done, error_msg = KnowledgebaseService.is_parsed_done(sample_kb['id'])

        # Assert
        assert is_done is True
        assert error_msg is None
        # Verify optimization: should call get_first_unparsed_document, NOT get_by_kb_id
        mock_doc_service.get_first_unparsed_document.assert_called_once_with(sample_kb['id'])
        mock_doc_service.get_by_kb_id.assert_not_called()

    @patch.object(KnowledgebaseService, 'query')
    @patch('api.db.services.document_service.DocumentService')
    @patch.object(DB, 'connect')
    @patch.object(DB, 'close')
    def test_is_parsed_done_returns_error_running(self, mock_close, mock_connect, mock_doc_service, mock_query, sample_kb):
        """Test is_parsed_done returns error when document is running."""
        # Arrange
        kb_mock = MagicMock()
        kb_mock.name = sample_kb['name']
        mock_query.return_value = [kb_mock]

        mock_doc_service.get_first_unparsed_document.return_value = {
            'name': 'running_doc.pdf',
            'run': TaskStatus.RUNNING.value,
            'chunk_num': 0
        }

        # Act
        is_done, error_msg = KnowledgebaseService.is_parsed_done(sample_kb['id'])

        # Assert
        assert is_done is False
        assert "is still being parsed" in error_msg
        assert "running_doc.pdf" in error_msg

    @patch.object(KnowledgebaseService, 'query')
    @patch('api.db.services.document_service.DocumentService')
    @patch.object(DB, 'connect')
    @patch.object(DB, 'close')
    def test_is_parsed_done_returns_error_unstart(self, mock_close, mock_connect, mock_doc_service, mock_query, sample_kb):
        """Test is_parsed_done returns error when document is unstarted."""
        # Arrange
        kb_mock = MagicMock()
        kb_mock.name = sample_kb['name']
        mock_query.return_value = [kb_mock]

        mock_doc_service.get_first_unparsed_document.return_value = {
            'name': 'unstarted_doc.pdf',
            'run': TaskStatus.UNSTART.value,
            'chunk_num': 0
        }

        # Act
        is_done, error_msg = KnowledgebaseService.is_parsed_done(sample_kb['id'])

        # Assert
        assert is_done is False
        assert "has not been parsed yet" in error_msg
        assert "unstarted_doc.pdf" in error_msg
