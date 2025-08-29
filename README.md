<h1 align="center">
	LaV-CoT: Language-Aware Visual CoT with Multi-Aspect Reward
Optimization for Real-World Multilingual VQA
</h1>

<div align="center">

![](./assets/framework.jpg)
</div>

# Table of Contents

- [Main](#main)
- [Data](#data)
- [Training Methods](#training-methods)
- [Citation](#citation)
# Main
It was trained on `Model_Arts`\
Please check `main/run_train_text_image_lora_csag.py`\
attention_guide_type == "sag" or "cag"

# Data
we crop out the half-body human images with sizes 512 x 512, since we are more interested in human details such as faces and eyes.

# Training Methods

![](./source/training_method.png)

# Citation  

```bibtex
WIP
```


For more work, please refer to [Academic Work](docs/ACADEMIC_WORK.md).