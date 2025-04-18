import uuid
from typing import Any, Dict, List, Optional

from src.data.processors.document_processor import DocumentProcessor
from src.data.vectors.service import VectorDBService
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger(__name__)


class DocumentService:
    def __init__(
        self,
        vector_db: VectorDBService,
        processor: DocumentProcessor,
    ):
        self.vector_db = vector_db
        self.processor = processor

    @metrics.track_request("document_ingestion")
    async def ingest_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            # Validate document first
            validation = await self.processor.validate_document(text, metadata)
            if not validation["metadata_complete"]:
                raise ValueError("Incomplete metadata")

            # Process document
            processed = await self.processor.process_document(text, metadata)

            # Generate unique IDs for chunks
            chunk_ids = [str(uuid.uuid4()) for _ in processed["chunks"]]

            # Add to vector database
            await self.vector_db.add_documents(
                documents=processed["chunks"],
                metadatas=[processed["metadata"]] * len(processed["chunks"]),
                ids=chunk_ids,
            )

            return {
                "document_id": str(uuid.uuid4()),
                "chunk_ids": chunk_ids,
                "stats": processed["stats"],
                "validation": validation,
            }
        except Exception as e:
            logger.error("Document ingestion failed: %s", e)
            raise

    @metrics.track_request("document_retrieval")
    async def retrieve_similar_documents(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            results = await self.vector_db.query(
                query_text=query,
                n_results=n_results,
                where=filters,
            )

            return [
                {
                    "content": result["document"],
                    "metadata": result["metadata"],
                    # Convert distance to similarity
                    "relevance": 1 - result["distance"],
                }
                for result in results
            ]
        except Exception as e:
            logger.error("Document retrieval failed: %s", e)
            raise

    @metrics.track_request("document_update")
    async def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            if text:
                processed = await self.processor.process_document(
                    text=text,
                    metadata=metadata,
                )
                await self.vector_db.update_document(
                    document_id=document_id,
                    document=(
                        processed["chunks"][0] if processed["chunks"] else None
                    ),
                    metadata=processed["metadata"],
                )
            elif metadata:
                processed_metadata = self.processor.process_metadata(metadata)
                await self.vector_db.update_document(
                    document_id=document_id,
                    metadata=processed_metadata,
                )
        except Exception as e:
            logger.error("Document update failed: %s", e)
            raise

    @metrics.track_request("document_deletion")
    async def delete_document(self, document_ids: List[str]):
        try:
            await self.vector_db.delete_documents(document_ids)
        except Exception as e:
            logger.error("Document deletion failed: %s", e)
            raise

    async def get_database_stats(self) -> Dict[str, Any]:
        return self.vector_db.get_collection_stats()
