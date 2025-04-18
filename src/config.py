import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class ModelSettings(BaseSettings):
    base_model_name: str = "mistralai/Mistral-7B-v0.1"
    model_revision: str = "main"
    quantization_bits: int = 4  # 4-bit quantization for efficiency
    max_sequence_length: int = 2048
    device: str = "cuda"  # or "cpu"
    batch_size: int = 32
    num_gpus_per_replica: float = 0.5  # Share GPUs between replicas
    max_concurrent_requests: int = 10


class VectorDBSettings(BaseSettings):
    db_path: Path = Path("data/vector_store")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    distance_metric: str = "cosine"
    max_elements_per_shard: int = 100_000


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 60
    cors_origins: List[str] = ["*"]
    api_version: str = "v1"


class SecuritySettings(BaseSettings):
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    ssl_keyfile: Optional[Path] = None
    ssl_certfile: Optional[Path] = None


class MonitoringSettings(BaseSettings):
    metrics_port: int = 9090
    enable_tracing: bool = True
    log_level: str = "INFO"
    prometheus_path: str = "/metrics"


class Settings(BaseSettings):
    environment: str = os.getenv("ENVIRONMENT", "development")
    project_root: Path = Path(__file__).parent.parent

    model: ModelSettings = ModelSettings()
    vector_db: VectorDBSettings = VectorDBSettings()
    api: APISettings = APISettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
