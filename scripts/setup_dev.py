import subprocess
from pathlib import Path


def setup_development_environment():
    print("Setting up development environment...")

    # Create necessary directories
    directories = [
        "data/vector_store",
        "volumes/etcd",
        "volumes/minio",
        "volumes/milvus",
        "scripts/data/documents",
        "scripts/output",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Start Milvus services
    print("\nStarting Milvus services...")
    subprocess.run(["docker-compose", "up", "-d"], check=True)

    # Install dependencies
    print("\nInstalling dependencies...")
    subprocess.run(["poetry", "install"], check=True)

    # Initialize Milvus
    print("\nInitializing Milvus...")
    subprocess.run(["python", "scripts/init_milvus.py"], check=True)

    print("\nDevelopment environment setup complete!")


if __name__ == "__main__":
    setup_development_environment()
