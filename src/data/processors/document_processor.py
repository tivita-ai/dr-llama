import re
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.phi_patterns = [
            r"\d{3}-\d{2}-\d{4}",   # SSN
            r"\d{3}/\d{3}/\d{4}",   # Date of birth
            r"[A-Z]\d{8}",          # Medical record number
            r"\d{10}",              # Phone numbers
        ]

    @metrics.track_request("document_processing")
    async def process_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            processed_text = self._remove_phi(text)
            chunks = self.chunk_text(processed_text)

            return {
                "chunks": chunks,
                "metadata": self.process_metadata(metadata),
                "stats": {
                    "original_length": len(text),
                    "processed_length": len(processed_text),
                    "chunk_count": len(chunks),
                },
            }
        except Exception as e:
            logger.error("Document processing failed: %s", e)
            raise

    def _remove_phi(self, text: str) -> str:
        for pattern in self.phi_patterns:
            text = re.sub(pattern, "[REDACTED]", text)
        return text

    def chunk_text(
        self,
        text: str,
        max_chunk_size: int = 1000,
        overlap: int = 100,
    ) -> List[str]:
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + max_chunk_size, len(text))

            if end_pos < len(text):
                # Try to find a good breaking point
                last_period = text.rfind(". ", current_pos, end_pos)
                if last_period != -1:
                    end_pos = last_period + 1

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            current_pos = end_pos - overlap

        return chunks

    def process_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not metadata:
            return {}

        processed = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                processed[key] = self._remove_phi(value)
            else:
                processed[key] = value

        return processed

    @metrics.track_request("document_validation")
    async def validate_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            phi_count = sum(
                len(re.findall(pattern, text))
                for pattern in self.phi_patterns
            )

            return {
                "has_phi": phi_count > 0,
                "phi_count": phi_count,
                "length": len(text),
                "metadata_complete": self._validate_metadata(metadata),
            }
        except Exception as e:
            logger.error("Document validation failed: %s", e)
            raise

    def _validate_metadata(self, metadata: Optional[Dict[str, Any]]) -> bool:
        if not metadata:
            return False

        required_fields = ["document_type", "creation_date", "author"]

        return all(field in metadata for field in required_fields)
