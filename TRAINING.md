# LaV-CoT training

This directory adds a readable training stack without changing the original
reward or prompt files. `lavcot/training/grpo_core.py` expands the central GRPO
math (group-relative advantages, clipped policy ratio, reference KL, and masked
token reduction); the command-line entry points use TRL for the distributed and
model-specific infrastructure around that objective.

## Data format

Use JSON or JSONL. A minimal row is:

```json
{"image":"0001.jpg","question":"What color is the sign?","answer":"blue","language":"English","response":"<think>...</think><answer>blue</answer>","gt_counts":2}
```

SFT consumes `response` (or `cot`). GRPO removes that response from the prompt
and samples new completions. A preformatted multimodal `messages` field is also
accepted.

## Install and train

```bash
pip install -r requirements-training.txt
accelerate launch scripts/train_sft.py configs/sft_example.yaml
accelerate launch scripts/train_grpo.py configs/grpo_example.yaml
```

Run SFT first, then set `model_name_or_path` in the GRPO config to the SFT
checkpoint. The example configs use LoRA for SFT and conservative memory
settings; tune batch sizes and generation counts for the available GPUs.

## Code map

- `data.py`: schema normalization and multimodal chat construction.
- `collator.py`: processor batching, padding masks, and assistant-only labels.
- `rewards.py`: TRL-compatible format, language, count, and exact-answer rewards.
- `grpo_core.py`: framework-independent, tested GRPO optimization objective.
- `scripts/train_sft.py`: supervised fine-tuning with optional LoRA.
- `scripts/train_grpo.py`: online grouped generation and reward optimization.
