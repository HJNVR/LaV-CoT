#!/usr/bin/env python
from __future__ import annotations

from functools import partial
import sys

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from lavcot.training.arguments import DataArguments, ModelArguments
from lavcot.training.collator import MultimodalSFTCollator
from lavcot.training.data import load_training_dataset, prepare_sft_example


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, data_args, training_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    model_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": model_args.torch_dtype,
    }
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=model_args.lora_target_modules.split(","),
                task_type="CAUSAL_LM",
            ),
        )

    train_dataset = load_training_dataset(
        data_args.dataset_name_or_path, data_args.train_split, data_args.dataset_config
    )
    mapper = partial(prepare_sft_example, processor=processor, image_root=data_args.image_root)
    train_dataset = train_dataset.map(mapper, num_proc=data_args.num_proc)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=MultimodalSFTCollator(processor, training_args.max_length),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
