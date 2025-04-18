from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dr_llama.data.processors.document_processor import DocumentProcessor
from dr_llama.data.services.document_service import DocumentService
from dr_llama.data.vectors.config import VectorDBConfig
from dr_llama.data.vectors.service import VectorDBService
from dr_llama.utils.metrics import metrics

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class DocumentMetadata(BaseModel):
    document_type: str = Field(
        ...,
        description="Type of the medical document",
    )
    creation_date: datetime = Field(
        ...,
        description="Document creation date",
    )
    author: str = Field(
        ...,
        description="Author of the document",
    )
    department: Optional[str] = Field(
        None,
        description="Department associated with the document",
    )
    patient_id: Optional[str] = Field(
        None,
        description="Patient identifier (if applicable)",
    )
    visit_id: Optional[str] = Field(
        None,
        description="Visit identifier (if applicable)",
    )


class DocumentCreate(BaseModel):
    text: str = Field(
        ...,
        description="Content of the medical document",
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Document metadata",
    )


class DocumentUpdate(BaseModel):
    text: Optional[str] = Field(
        None,
        description="Updated document content",
    )
    metadata: Optional[DocumentMetadata] = Field(
        None,
        description="Updated metadata",
    )


class DocumentResponse(BaseModel):
    document_id: str
    chunk_ids: List[str]
    stats: Dict[str, Any]
    validation: Dict[str, Any]


class SimilarDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    relevance: float


class DocumentQuery(BaseModel):
    query: str = Field(..., description="Search query")
    n_results: int = Field(5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Filter criteria",
    )


async def get_document_service() -> DocumentService:
    config = VectorDBConfig()
    vector_db = VectorDBService(config)
    processor = DocumentProcessor()
    return DocumentService(vector_db, processor)


@router.post("/", response_model=DocumentResponse)
@metrics.track_request("create_document")
async def create_document(
    document: DocumentCreate,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    try:
        result = await document_service.ingest_document(
            text=document.text, metadata=document.metadata.dict()
        )
        return DocumentResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/search", response_model=List[SimilarDocument])
@metrics.track_request("search_documents")
async def search_documents(
    query: DocumentQuery,
    document_service: DocumentService = Depends(get_document_service),
) -> List[SimilarDocument]:
    try:
        results = await document_service.retrieve_similar_documents(
            query=query.query, n_results=query.n_results, filters=query.filters
        )
        return [SimilarDocument(**result) for result in results]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/{document_id}")
@metrics.track_request("update_document")
async def update_document(
    document_id: str,
    update: DocumentUpdate,
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        await document_service.update_document(
            document_id=document_id,
            text=update.text,
            metadata=update.metadata.dict() if update.metadata else None,
        )
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to update document"
        ) from exc


@router.delete("/{document_id}")
@metrics.track_request("delete_document")
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        await document_service.delete_document([document_id])
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to delete document"
        ) from exc


@router.get("/stats")
@metrics.track_request("get_stats")
async def get_stats(
    document_service: DocumentService = Depends(get_document_service),
) -> Dict[str, Any]:
    try:
        return await document_service.get_database_stats()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to get database stats"
        ) from exc
