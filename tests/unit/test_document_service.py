"""
Unit tests for document processing service
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from app.services.document_service import DocumentService


@pytest.fixture
def document_service():
    """Create document service instance"""
    return DocumentService()


@pytest.mark.asyncio
async def test_process_pdf_document(document_service):
    """Test PDF document processing"""
    # Create a mock PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with patch('fitz.open') as mock_fitz:
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample PDF text content"
        mock_page.get_images.return_value = []
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_fitz.return_value = mock_doc

        result = await document_service.process_document(tmp_path, 'application/pdf')

        assert result['status'] == 'success'
        assert 'chunks' in result
        assert len(result['chunks']) > 0
        assert 'Sample PDF text' in result['chunks'][0]['content']

    # Cleanup
    tmp_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_process_docx_document(document_service):
    """Test DOCX document processing"""
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with patch('docx.Document') as mock_docx:
        mock_doc = Mock()
        mock_para = Mock()
        mock_para.text = "Sample DOCX paragraph"
        mock_doc.paragraphs = [mock_para]
        mock_docx.return_value = mock_doc

        result = await document_service.process_document(tmp_path, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')

        assert result['status'] == 'success'
        assert len(result['chunks']) > 0
        assert 'Sample DOCX' in result['chunks'][0]['content']

    tmp_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_process_text_document(document_service):
    """Test plain text document processing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp.write("Sample text content for testing.\n" * 100)
        tmp_path = Path(tmp.name)

    result = await document_service.process_document(tmp_path, 'text/plain')

    assert result['status'] == 'success'
    assert len(result['chunks']) > 0
    assert 'Sample text content' in result['chunks'][0]['content']

    tmp_path.unlink()


@pytest.mark.asyncio
async def test_chunk_text(document_service):
    """Test text chunking with overlap"""
    text = "This is a test. " * 200  # Long text

    chunks = document_service._chunk_text(text, chunk_size=500, overlap=50)

    assert len(chunks) > 1
    # Check overlap
    assert chunks[1]['content'][:20] in chunks[0]['content']


def test_extract_metadata(document_service):
    """Test metadata extraction"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    metadata = document_service._extract_metadata(tmp_path, 'application/pdf')

    assert 'filename' in metadata
    assert 'file_type' in metadata
    assert 'mime_type' in metadata
    assert metadata['mime_type'] == 'application/pdf'

    tmp_path.unlink()


@pytest.mark.asyncio
async def test_unsupported_format(document_service):
    """Test handling of unsupported file format"""
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    result = await document_service.process_document(tmp_path, 'application/xyz')

    assert result['status'] == 'error'
    assert 'unsupported' in result['message'].lower()

    tmp_path.unlink()


@pytest.mark.asyncio
async def test_process_markdown(document_service):
    """Test Markdown document processing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp.write("# Test Markdown\n\nSome **bold** text and *italic* text.\n\n## Section 2\n\nMore content here.")
        tmp_path = Path(tmp.name)

    result = await document_service.process_document(tmp_path, 'text/markdown')

    assert result['status'] == 'success'
    assert len(result['chunks']) > 0
    assert 'Markdown' in result['chunks'][0]['content']

    tmp_path.unlink()


@pytest.mark.asyncio
async def test_ocr_fallback(document_service):
    """Test OCR fallback for scanned PDFs"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with patch('fitz.open') as mock_fitz, \
         patch('pytesseract.image_to_string') as mock_ocr:

        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # Empty text triggers OCR
        mock_page.get_pixmap.return_value = Mock()
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_fitz.return_value = mock_doc

        mock_ocr.return_value = "OCR extracted text"

        result = await document_service.process_document(tmp_path, 'application/pdf')

        # OCR should have been called
        assert mock_ocr.called

    tmp_path.unlink(missing_ok=True)
