from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from models.document import Document


class DocumentTrainer:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        max_length: int = 2048,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def prepare_document(self, document: Document) -> str:
        title = document.title
        content = document.get_text_content()
        return f"Title: {title}\n\nContent: {content}"

    def create_dataset(self, documents: List[Document]) -> Dataset:
        texts = [self.prepare_document(doc) for doc in documents]

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        return tokenized_dataset

    def train(
        self,
        documents: List[Document],
        output_dir: str = "output",
        save_steps: int = 500,
        logging_steps: int = 100,
    ):
        dataset = self.create_dataset(documents)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            save_steps=save_steps,
            logging_steps=logging_steps,
            save_total_limit=2,
            fp16=True,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
