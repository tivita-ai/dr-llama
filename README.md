# Dr. LLama

A LLM-based project using state-of-the-art machine learning models.

## Prerequisites

- Python 3.13+
- Poetry for dependency management
- Docker and Docker Compose (for Milvus)

## Installation

### 1. Python Dependencies

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 2. Environment Setup

```bash
# Activate the virtual environment
poetry shell

# Start Milvus services
docker-compose up -d

# Initialize Milvus
python scripts/init_milvus.py
```

## Core Dependencies

- **Vector Database**:
  - Milvus (^2.5.6)
  - Sentence Transformers (^4.1.0)

- **API & Services**:
  - FastAPI (^0.115.12)
  - Uvicorn (^0.34.1)
  - Pydantic (^2.11.3)
  - Pydantic Settings (^2.9.0)

- **Monitoring**:
  - Prometheus Client (^0.21.1)

- **Development**:
  - Ruff (^0.11.6)
  - Pytest (^8.0.0)
  - Pytest Asyncio (^0.23.0)

## Project Structure

```
dr-llama/
├── app/
│   ├── api/                  # API routes and endpoints
│   ├── data/                 # Data processing and services
│   │   ├── processors/       # Document processors
│   │   ├── services/         # Business logic services
│   │   └── vector_store/     # Vector database integration
│   └── utils/                # Utility functions
├── data/                     # Data storage and resources
├── scripts/                  # Utility scripts
│   ├── data/                 # Data processing scripts
│   └── output/               # Script output directory
├── tests/                    # Test files
├── volumes/                  # Docker volumes
│   ├── etcd/                 # Milvus etcd data
│   ├── minio/                # Milvus minio data
│   └── milvus/               # Milvus data
├── docker-compose.yml        # Milvus services configuration
├── pyproject.toml            # Project dependencies and metadata
└── poetry.lock               # Lock file for dependencies
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Running the API Server

```bash
poetry run uvicorn app.main:app --reload
```

## Development Guidelines

1. Always use Poetry for dependency management
2. Follow Ruff linting rules (line length: 120)
3. Run tests before committing changes
4. Use type hints and docstrings
5. Keep the vector store configuration in sync with Milvus setup

## Troubleshooting

### Common Issues

1. **Milvus Connection Issues**:
   - Ensure Docker services are running: `docker-compose ps`
   - Check Milvus logs: `docker-compose logs standalone`

2. **Dependency Conflicts**:
   - Clear Poetry cache: `poetry cache clear . --all`
   - Reinstall dependencies: `poetry install`

3. **Linting Issues**:
   - Run Ruff: `poetry run ruff check .`
   - Fix formatting: `poetry run ruff format .`

