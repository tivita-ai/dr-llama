# Dr. LLama

A LLM-based project using state-of-the-art machine learning models.

## Prerequisites

- Python 3.12+
- Poetry for dependency management

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
```

## Core Dependencies

- **ML & LLM**:
  - PyTorch (^2.0.0)
  - Transformers (^4.30.0)

- **API & Services**:
  - FastAPI (^0.100.0)
  - Uvicorn (^0.22.0)
  - Pydantic (^2.0.0)

- **Utilities**:
  - Python-dotenv (^1.0.0)

## Adding Additional Dependencies

You can add new dependencies as needed using Poetry:

```bash
# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Add a dependency with extras
poetry add "package-name[extra1,extra2]"
```

Common packages you might need:

```bash
# For vector database
poetry add chromadb sentence-transformers

# For model optimization
poetry add accelerate bitsandbytes peft

# For monitoring
poetry add prometheus-client opentelemetry-api opentelemetry-sdk

# For security
poetry add "python-jose[cryptography]" "passlib[bcrypt]"
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

## Project Structure

```
dr-llama/
├── app/              # Main application code
├── config/           # Configuration files
├── data/            # Data storage and resources
├── models/          # ML model definitions and weights
├── scripts/         # Utility scripts
├── tests/           # Test files
├── utils/           # Helper utilities
├── main.py          # Application entry point
├── pyproject.toml   # Project dependencies and metadata
└── poetry.lock      # Lock file for dependencies
```

## Development Guidelines

1. Always use Poetry for dependency management
2. Add dependencies only when needed
3. Run tests before committing changes
4. Follow the project's code style and documentation standards

## Troubleshooting

### Common Issues on macOS

If you encounter build issues with `chroma-hnswlib`, ensure you have:
1. Installed the required build tools (`cmake` and `libomp`)
2. Set the correct environment variables (`LDFLAGS` and `CPPFLAGS`)
3. Using the compatible version of ChromaDB (0.4.22)

