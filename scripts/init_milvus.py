from app.data.vector_store.config import VectorDBConfig
from app.data.vector_store.service import VectorDBService


def init_milvus():
    config = VectorDBConfig()
    vector_db = VectorDBService(config)

    print("Milvus initialized successfully!")
    print(f"Collection name: {config.collection_name}")
    print(f"Embedding dimension: {config.embedding_dimension}")
    print(f"Document count: {vector_db.get_document_count()}")

    vector_db.close()


if __name__ == "__main__":
    init_milvus()
