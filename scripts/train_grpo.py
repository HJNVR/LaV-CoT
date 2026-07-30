#!/usr/bin/env python
from __future__ import annotations

from functools import partial
import sys

from transformers import AutoProcessor, HfArgumentParser, set_seed
from trl import GRPOConfig, GRPOTrainer

from lavcot.training.arguments import DataArguments, ModelArguments
from lavcot.training.data import load_training_dataset, prepare_grpo_example
from lavcot.training.rewards import answer_reward, count_reward, format_reward, language_reward


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, data_args, training_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    dataset = load_training_dataset(
        data_args.dataset_name_or_path, data_args.train_split, data_args.dataset_config
    )
    dataset = dataset.map(
        partial(prepare_grpo_example, processor=processor, image_root=data_args.image_root),
        num_proc=data_args.num_proc,
    )
    reward_funcs = [format_reward, language_reward, count_reward, answer_reward]
    if training_args.reward_weights is None:
        training_args.reward_weights = [0.25] * len(reward_funcs)
    if len(training_args.reward_weights) != len(reward_funcs):
        raise ValueError("reward_weights must contain four comma-separated numbers")
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
