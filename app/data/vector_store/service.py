from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

from app.data.vector_store.config import VectorDBConfig


class VectorDBService:
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._connect()
        self._setup_collection()

    def _connect(self):
        connections.connect(
            alias="default",
            host=self.config.host,
            port=self.config.port,
            user=self.config.username,
            password=self.config.password,
        )

    def _setup_collection(self):
        if not utility.has_collection(self.config.collection_name):
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.config.embedding_dimension,
                ),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(
                fields=fields,
                description="Medical documents collection",
            )
            self.collection = Collection(
                name=self.config.collection_name,
                schema=schema,
            )

            index_params = {
                "metric_type": self.config.metric_type,
                "index_type": self.config.index_type,
                "params": {"nlist": self.config.nlist},
            }
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params,
            )
        else:
            self.collection = Collection(self.config.collection_name)
            self.collection.load()

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.encode(contents).tolist()
        metadata = [doc.get("metadata", {}) for doc in documents]

        entities = [
            contents,
            embeddings,
            metadata,
        ]

        insert_result = self.collection.insert(entities)
        self.collection.flush()
        return insert_result.primary_keys

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": self.config.nprobe},
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=search_filter,
            output_fields=["content", "metadata"],
        )

        return [
            {
                "content": hit.entity.get("content"),
                "metadata": hit.entity.get("metadata"),
                "score": hit.score,
            }
            for hit in results[0]
        ]

    def delete_documents(self, ids: List[int]) -> None:
        expr = f"id in {ids}"
        self.collection.delete(expr)

    def get_document_count(self) -> int:
        return self.collection.num_entities

    def close(self):
        connections.disconnect("default")
