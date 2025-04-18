from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformers import AutoConfig


@dataclass
class ModelConfig:
    model_name: str
    revision: str
    quantization_bits: int
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    low_cpu_mem_usage: bool = True

    def to_transformers_config(self) -> Dict[str, Any]:
        config = AutoConfig.from_pretrained(
            self.model_name,
            revision=self.revision,
        )

        # Set optimizations for inference
        config.use_cache = True
        config.output_attentions = False
        config.output_hidden_states = False

        return {
            "config": config,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "load_in_4bit": self.quantization_bits == 4,
            "load_in_8bit": self.quantization_bits == 8,
        }


@dataclass
class InferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1

    def to_generate_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "num_return_sequences": self.num_return_sequences,
            "pad_token_id": 50256,  # Default GPT-2 pad token
            "eos_token_id": 50256,  # Default GPT-2 eos token
        }


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    max_steps: int = 10000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # LoRA specific settings
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    def to_training_arguments(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "fp16": True,
            "logging_steps": 10,
            "save_steps": 1000,
            "eval_steps": 500,
        }
