import json
from pathlib import Path

from models.document import Document
from models.training.trainer import DocumentTrainer


def load_documents(directory: str) -> list[Document]:
    documents = []
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            document = Document(**data)
            documents.append(document)
    return documents


def main():
    documents = load_documents("scripts/data/documents")

    trainer = DocumentTrainer(
        model_name="meta-llama/Llama-2-7b-hf",
        max_length=2048,
        batch_size=4,
        learning_rate=2e-5,
        num_epochs=3,
    )

    trainer.train(
        documents=documents,
        output_dir="scripts/output",
        save_steps=500,
        logging_steps=100,
    )


if __name__ == "__main__":
    main()
