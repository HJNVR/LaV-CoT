from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Local path or Hugging Face model id."})
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class DataArguments:
    dataset_name_or_path: str = field(metadata={"help": "JSON/JSONL file, directory, or HF dataset id."})
    dataset_config: Optional[str] = None
    train_split: str = "train"
    eval_split: Optional[str] = None
    image_root: Optional[str] = None
    num_proc: int = 4
