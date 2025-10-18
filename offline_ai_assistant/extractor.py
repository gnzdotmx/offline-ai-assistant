"""
Document extraction module for PDF and DOCX files.

This module provides functionality to extract text content from PDF and DOCX files
with robust error handling and metadata extraction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
from datetime import datetime

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document
except ImportError:
    Document = None

from .config import Config

logger = logging.getLogger("OfflineAIAssistant.extractor")


class DocumentExtractor:
    """Extract text content from PDF and DOCX files."""
    
    def __init__(self):
        """Initialize the document extractor."""
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if fitz is None:
            logger.error("PyMuPDF not installed. PDF extraction will not work.")
        if Document is None:
            logger.error("python-docx not installed. DOCX extraction will not work.")
    
    def extract_from_file(self, file_path: Path) -> Dict[str, any]:
        """
        Extract text and metadata from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted text and metadata
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
            Exception: For extraction errors
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in Config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        logger.info(f"Extracting text from: {file_path}")
        
        try:
            if suffix == ".pdf":
                return self._extract_pdf(file_path)
            elif suffix == ".docx":
                return self._extract_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
                
        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {str(e)}")
            raise
    
    def _extract_pdf(self, file_path: Path) -> Dict[str, any]:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF not installed. Cannot extract PDF files.")
        
        doc = None
        try:
            # Open document with proper error handling
            doc = fitz.open(str(file_path))
            
            if doc.is_closed:
                raise RuntimeError("Failed to open PDF document")
            
            pages = []
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                pages.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
                full_text += page_text + "\n"
            
            # Extract metadata before closing
            metadata = doc.metadata if doc and not doc.is_closed else {}
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": "pdf",
                "file_size": file_path.stat().st_size,
                "file_hash": file_hash,
                "extraction_date": datetime.now().isoformat(),
                "full_text": full_text.strip(),
                "page_count": len(doc),
                "pages": pages,
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
                "metadata": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", "")
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to extract PDF: {str(e)}")
        finally:
            # Ensure document is properly closed
            if doc and not doc.is_closed:
                try:
                    doc.close()
                except:
                    pass
    
    def _extract_docx(self, file_path: Path) -> Dict[str, any]:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        if Document is None:
            raise RuntimeError("python-docx not installed. Cannot extract DOCX files.")
        
        try:
            doc = Document(file_path)
            paragraphs = []
            full_text = ""
            
            for i, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text
                if para_text.strip():  # Skip empty paragraphs
                    paragraphs.append({
                        "paragraph_number": i + 1,
                        "text": para_text,
                        "char_count": len(para_text)
                    })
                    full_text += para_text + "\n"
            
            # Extract core properties
            core_props = doc.core_properties
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": "docx",
                "file_size": file_path.stat().st_size,
                "file_hash": file_hash,
                "extraction_date": datetime.now().isoformat(),
                "full_text": full_text.strip(),
                "paragraph_count": len(paragraphs),
                "paragraphs": paragraphs,
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
                "metadata": {
                    "title": core_props.title or "",
                    "author": core_props.author or "",
                    "subject": core_props.subject or "",
                    "keywords": core_props.keywords or "",
                    "comments": core_props.comments or "",
                    "category": core_props.category or "",
                    "created": core_props.created.isoformat() if core_props.created else "",
                    "modified": core_props.modified.isoformat() if core_props.modified else "",
                    "last_modified_by": core_props.last_modified_by or ""
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to extract DOCX: {str(e)}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of a file for deduplication.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return ""
    
    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate if a file can be processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        suffix = file_path.suffix.lower()
        if suffix not in Config.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {suffix}"
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_path.stat().st_size > max_size:
            return False, f"File too large (max 100MB): {file_path}"
        
        # Check if dependencies are available
        if suffix == ".pdf" and fitz is None:
            return False, "PyMuPDF not installed. Cannot process PDF files."
        
        if suffix == ".docx" and Document is None:
            return False, "python-docx not installed. Cannot process DOCX files."
        
        return True, ""
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        supported = []
        
        if fitz is not None:
            supported.append(".pdf")
        
        if Document is not None:
            supported.append(".docx")
        
        return supported


def extract_document(file_path: Path) -> Dict[str, any]:
    """
    Convenience function to extract text from a document.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    extractor = DocumentExtractor()
    return extractor.extract_from_file(file_path)
