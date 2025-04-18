import subprocess
from pathlib import Path


def setup_development_environment():
    print("Setting up development environment...")

    # Create necessary directories
    directories = [
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

    # Initialize Milvus
    print("\nRunning init_milvus.py...")
    subprocess.run(["python", "scripts/init_milvus.py"], check=True)

    print("\nDevelopment environment setup complete!")


if __name__ == "__main__":
    setup_development_environment()
