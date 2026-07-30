from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def _extract_images(messages: List[Dict[str, Any]]) -> List[Any]:
    images: List[Any] = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        for item in content:
            if item.get("type") == "image" and item.get("image") is not None:
                images.append(item["image"])
    return images


@dataclass
class MultimodalSFTCollator:
    """Tokenize chat examples and mask padding plus non-assistant tokens."""

    processor: Any
    max_length: int = 4096

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [
            self.processor.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            for item in examples
        ]
        images = [_extract_images(item["messages"]) for item in examples]
        kwargs: Dict[str, Any] = {
            "text": texts,
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if any(images):
            kwargs["images"] = images
        batch = self.processor(**kwargs)
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        # Prefer the model's assistant-token mask when the processor exposes it.
        try:
            templated = self.processor.apply_chat_template(
                [item["messages"] for item in examples],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                return_assistant_tokens_mask=True,
            )
            mask = templated.get("assistant_masks")
            if mask is not None and mask.shape == labels.shape:
                labels[mask == 0] = -100
        except (TypeError, ValueError, KeyError):
            pass
        batch["labels"] = labels
        return batch
