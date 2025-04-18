from src.data.vectors.config import VectorDBConfig
from src.data.vectors.service import VectorDBService


def init_milvus():
    print("Initializing Milvus...")

    config = VectorDBConfig()
    vector_db = VectorDBService(config)

    print("Milvus initialized successfully!")
    print(f"Collection name: {config.collection_name}")
    print(f"Embedding dimension: {config.embedding_dimension}")
    print(f"Document count: {vector_db.get_document_count()}")

    vector_db.close()


if __name__ == "__main__":
    init_milvus()
