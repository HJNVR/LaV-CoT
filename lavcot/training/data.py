from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset


SYSTEM_PROMPT = (
    "You are a vision-language assistant. Explain the visual evidence in the "
    "requested language. Put reasoning in <think></think> and the final answer "
    "in <answer></answer>."
)


def load_training_dataset(
    source: str,
    split: str = "train",
    config: Optional[str] = None,
) -> Dataset:
    """Load JSON/JSONL files, a directory of files, or a Hub dataset."""
    path = Path(source)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported dataset file: {path}")
        return load_dataset("json", data_files=str(path), split="train")
    if path.is_dir():
        files = sorted([*path.glob("*.json"), *path.glob("*.jsonl")])
        if not files:
            raise ValueError(f"No JSON or JSONL files found under {path}")
        return load_dataset("json", data_files=[str(p) for p in files], split="train")
    loaded = load_dataset(source, config)
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise KeyError(f"Split {split!r} not found; available: {list(loaded)}")
        return loaded[split]
    return loaded


def _first(example: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in example and example[name] is not None:
            return example[name]
    return default


def resolve_image(image: Any, image_root: Optional[str]) -> Any:
    if not isinstance(image, str) or image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    if image_root and not path.is_absolute():
        path = Path(image_root) / path
    return str(path.resolve())


def build_messages(example: Dict[str, Any], image_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Normalize common VQA schemas to the multimodal chat-template schema."""
    if "messages" in example:
        return example["messages"]
    question = _first(example, "question", "query", "prompt")
    if not question:
        raise ValueError("Each example needs `messages` or a question/query/prompt field.")
    language = _first(example, "language", "lang", "target_language", default="English")
    image = resolve_image(_first(example, "image", "image_path"), image_root)
    user_content: List[Dict[str, Any]] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append(
        {"type": "text", "text": f"Target language: {language}\nQuestion: {question}"}
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    response = _first(example, "response", "completion", "cot")
    answer = _first(example, "answer", "final_answer")
    if response is None and answer is not None:
        response = f"<answer>{answer}</answer>"
    if response is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(response)}]}
        )
    return messages


def prepare_sft_example(
    example: Dict[str, Any], processor: Any, image_root: Optional[str] = None
) -> Dict[str, Any]:
    messages = build_messages(example, image_root)
    return {
        "messages": messages,
        "text": processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ),
    }


def prepare_grpo_example(
    example: Dict[str, Any], processor: Any, image_root: Optional[str] = None
) -> Dict[str, Any]:
    messages = build_messages(example, image_root)
    if messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    result = dict(example)
    result["prompt"] = messages
    result["answer"] = str(_first(example, "answer", "final_answer", default=""))
    result["lang"] = str(
        _first(example, "language", "lang", "target_language", default="English")
    )
    return result
