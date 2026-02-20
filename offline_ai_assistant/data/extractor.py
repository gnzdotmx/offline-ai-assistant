"""
Document extraction module for PDF and DOCX files.

Uses config for supported extensions. File validation includes size limit and type checks.
"""

import hashlib
import logging
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document
except ImportError:
    Document = None

from ..config import Config

logger = logging.getLogger("OfflineAIAssistant.extractor")


class DocumentExtractor:
    """Extract text content from PDF and DOCX files."""

    def __init__(self) -> None:
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        if fitz is None:
            logger.error("PyMuPDF not installed. PDF extraction will not work.")
        if Document is None:
            logger.error("python-docx not installed. DOCX extraction will not work.")

    @staticmethod
    def _clean_extracted_text(text: str, window_size: int = 30) -> str:
        """Clean extracted text: merge hyphenated line breaks and remove repeated lines.

        (1) Merges line-end hyphenation (e.g. "word-\\nnext" -> "word next").
        (2) Removes repeated lines using a sliding window (e.g. repeated headers/footers).
        Lines are compared after stripping; first occurrence in the window is kept.
        """
        if not text or not text.strip():
            return text
        # Merge hyphenated line breaks: "word-\nnext" -> "word next"
        merged = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1 \2", text)
        lines = merged.splitlines()
        if not lines:
            return text
        # Sliding window dedup: skip a line if its stripped form was seen in the last window_size lines
        result: List[str] = []
        window: deque = deque(maxlen=window_size)
        for line in lines:
            key = line.strip()
            if key and key in window:
                continue
            result.append(line)
            if key:
                window.append(key)
        return "\n".join(result).strip()

    def extract_from_file(self, file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in Config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.info("Extracting text from: %s", file_path)

        try:
            if suffix == ".pdf":
                return self._extract_pdf(file_path)
            if suffix == ".docx":
                return self._extract_docx(file_path)
            raise ValueError(f"Unsupported file type: {suffix}")
        except Exception as e:
            logger.error("Error extracting from %s: %s", file_path, e)
            raise

    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        if fitz is None:
            raise RuntimeError("PyMuPDF not installed. Cannot extract PDF files.")

        doc = None
        try:
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
                    "char_count": len(page_text),
                })
                full_text += page_text + "\n"

            metadata = doc.metadata if doc and not doc.is_closed else {}
            file_hash = self._calculate_file_hash(file_path)

            full_text = full_text.strip()
            if Config.EXTRACTOR_CLEAN_TEXT:
                full_text = self._clean_extracted_text(full_text)

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": "pdf",
                "file_size": file_path.stat().st_size,
                "file_hash": file_hash,
                "extraction_date": datetime.now().isoformat(),
                "full_text": full_text,
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
                    "modification_date": metadata.get("modDate", ""),
                },
            }
        except Exception as e:
            logger.error("Error extracting PDF %s: %s", file_path, e)
            raise RuntimeError(f"Failed to extract PDF: {e}") from e
        finally:
            if doc and not doc.is_closed:
                try:
                    doc.close()
                except Exception:
                    pass

    def _extract_docx(self, file_path: Path) -> Dict[str, Any]:
        if Document is None:
            raise RuntimeError("python-docx not installed. Cannot extract DOCX files.")

        try:
            doc = Document(file_path)
            paragraphs = []
            full_text = ""
            for i, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text
                if para_text.strip():
                    paragraphs.append({
                        "paragraph_number": i + 1,
                        "text": para_text,
                        "char_count": len(para_text),
                    })
                    full_text += para_text + "\n"

            core_props = doc.core_properties
            file_hash = self._calculate_file_hash(file_path)

            full_text = full_text.strip()
            if Config.EXTRACTOR_CLEAN_TEXT:
                full_text = self._clean_extracted_text(full_text)

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": "docx",
                "file_size": file_path.stat().st_size,
                "file_hash": file_hash,
                "extraction_date": datetime.now().isoformat(),
                "full_text": full_text,
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
                    "last_modified_by": core_props.last_modified_by or "",
                },
            }
        except Exception as e:
            logger.error("Error extracting DOCX %s: %s", file_path, e)
            raise RuntimeError(f"Failed to extract DOCX: {e}") from e

    def _calculate_file_hash(self, file_path: Path) -> str:
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error("Error calculating hash for %s: %s", file_path, e)
            return ""

    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"

        suffix = file_path.suffix.lower()
        if suffix not in Config.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {suffix}"

        max_size = 100 * 1024 * 1024  # 100MB
        if file_path.stat().st_size > max_size:
            return False, f"File too large (max 100MB): {file_path}"

        if suffix == ".pdf" and fitz is None:
            return False, "PyMuPDF not installed. Cannot process PDF files."
        if suffix == ".docx" and Document is None:
            return False, "python-docx not installed. Cannot process DOCX files."

        return True, ""

    def get_supported_extensions(self) -> List[str]:
        supported = []
        if fitz is not None:
            supported.append(".pdf")
        if Document is not None:
            supported.append(".docx")
        return supported


def extract_document(file_path: Path) -> Dict[str, Any]:
    """Convenience function to extract text from a document."""
    extractor = DocumentExtractor()
    return extractor.extract_from_file(file_path)
