from typing import Optional

from pydantic_settings import BaseSettings


class VectorDBConfig(BaseSettings):
    host: str = "localhost"
    port: str = "19530"
    collection_name: str = "medical_documents"
    embedding_dimension: int = 768
    metric_type: str = "L2"
    index_type: str = "IVF_FLAT"
    nlist: int = 1024
    nprobe: int = 10
    consistency_level: str = "Strong"
    username: Optional[str] = None
    password: Optional[str] = None

    class Config:
        env_prefix = "MILVUS_"
