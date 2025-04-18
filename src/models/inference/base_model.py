from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logger import get_logger
from src.models.model_config import InferenceConfig, ModelConfig

logger = get_logger(__name__)


class BaseModel:
    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ):
        self.model_config = model_config
        self.inference_config = inference_config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        self._load_model()
        self._load_tokenizer()

    def _load_model(self):
        logger.info("Loading model %s", self.model_config.model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name,
                **self.model_config.to_transformers_config(),
            )
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def _load_tokenizer(self):
        logger.info("Loading tokenizer for %s", self.model_config.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name,
                revision=self.model_config.revision,
                use_fast=True,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", e)
            raise

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config.max_sequence_length,
            ).to(self.device)

            generate_config = self.inference_config.to_generate_config()
            if max_new_tokens is not None:
                generate_config["max_new_tokens"] = max_new_tokens
            generate_config.update(kwargs)

            outputs = self.model.generate(**inputs, **generate_config)

            return self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )

        except Exception as e:
            logger.error("Generation failed: %s", e)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_config.model_name,
            "revision": self.model_config.revision,
            "device": str(self.device),
            "quantization": f"{self.model_config.quantization_bits}-bit",
            "max_sequence_length": self.model_config.max_sequence_length,
        }

    @property
    def max_sequence_length(self) -> int:
        return self.model_config.max_sequence_length
