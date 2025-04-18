from pathlib import Path

from src.models.training.trainer import ModelTrainer


def train_model():
    print("Starting model training...")

    # Initialize trainer
    trainer = ModelTrainer()

    # Load training data
    data_path = Path("scripts/data/documents")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {data_path}",
        )

    # Train model
    trainer.train(data_path)

    print("Model training completed successfully!")


if __name__ == "__main__":
    train_model()
