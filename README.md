<h1 align="center">
	LaV-CoT: Language-Aware Visual CoT with Multi-Aspect Reward
Optimization for Real-World Multilingual VQA
</h1>

<div align="center">

![](./assets/framework.jpg)
</div>

# Table of Contents

- [Prompt Design](#prompt)
- [Data](#data)
- [Training Methods](#training-methods)
- [Citation](#citation)

# Prompt
Please check `./prompts` for both generator and evaluator prompts.

# Data
We design an automatic data curation method that produces
scalable, high-quality multilingual CoT annotations through
iterative generation, correction, and refinement. All images are resized to 896 * 896.

# Training Methods
We adopt TRL offical training scripts (https://github.com/huggingface/trl) to do both SFT and GRPO training.

# Citation  

```bibtex
WIP
```


For more work, please refer to [Academic Work](docs/ACADEMIC_WORK.md).