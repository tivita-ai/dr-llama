# pylint: disable=W0603, W0613, W0621
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.api.routes import router as document_router
from app.data.processors.document_processor import DocumentProcessor
from app.data.services.document_service import DocumentService
from app.data.vector_store.config import VectorDBConfig
from app.data.vector_store.service import VectorDBService
from app.utils.logger import get_logger
from app.utils.metrics import metrics

logger = get_logger(__name__)


vector_db = None
document_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, document_service

    logger.info("Starting up application...")

    config = VectorDBConfig()
    vector_db = VectorDBService(config)
    processor = DocumentProcessor()
    document_service = DocumentService(vector_db, processor)

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title="Tivita LLM model API",
    description="API for Tivita LLM model Doctor Lamma",
    version="1.0.0",
    lifespan=lifespan,
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


app.include_router(document_router)


@app.get("/health", tags=["health"])
@metrics.track_request("health_check")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "vector_db": vector_db is not None,
            "document_service": document_service is not None,
        },
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
