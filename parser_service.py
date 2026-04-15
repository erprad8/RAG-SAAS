"""
Document parser: supports PDF and TXT formats.
Author: Pradeep Kumar Verma

Design notes:
  - PDF parsing uses PyPDF2 with page-level metadata so chunks can be
    traced back to their source page — critical for debugging retrieval failures.
  - TXT parsing reads the full file and records character offsets.
  - Both return a flat list of (text, metadata_dict) tuples consumed by the chunker.
"""

import io
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class UnsupportedFormatError(Exception):
    pass


class DocumentParser:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def parse(self, file_bytes: bytes, filename: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse a document into a list of (text_block, metadata) tuples.
        Each text_block corresponds to one logical unit (e.g., one PDF page or
        one continuous TXT block). The chunker further splits these.

        Returns:
            List of (text, metadata) where metadata contains at minimum:
            - filename
            - page_number (PDF only, else None)
        """
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file format '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if ext == ".pdf":
            return self._parse_pdf(file_bytes, filename)
        elif ext == ".txt":
            return self._parse_txt(file_bytes, filename)

    def _parse_pdf(self, file_bytes: bytes, filename: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract text page-by-page from a PDF."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF parsing: pip install PyPDF2")

        result = []
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            logger.info(f"PDF '{filename}' has {len(reader.pages)} pages.")
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    logger.debug(f"Page {page_num} of '{filename}' yielded no text (possibly image-based).")
                    continue
                result.append((text, {"filename": filename, "page_number": page_num}))
        except Exception as e:
            logger.error(f"Failed to parse PDF '{filename}': {e}")
            raise

        if not result:
            logger.warning(f"No extractable text found in '{filename}'. It may be a scanned PDF.")
        return result

    def _parse_txt(self, file_bytes: bytes, filename: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Read a plain-text file, splitting on double newlines into paragraphs."""
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to decode TXT file '{filename}': {e}")
            raise

        # Split on paragraph boundaries (double newlines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []

        logger.info(f"TXT '{filename}' split into {len(paragraphs)} paragraph block(s).")
        return [(p, {"filename": filename, "page_number": None}) for p in paragraphs]
