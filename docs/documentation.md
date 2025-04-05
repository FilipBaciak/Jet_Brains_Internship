# Documentation

## Overview
This repository contains a recruiting project for the internship titled **"Project Adaptation of LLMs for Code Generation"** at JetBrains in the summer of 2025.

## References
- [CHRF Evaluation Metric](https://huggingface.co/spaces/evaluate-metric/chrf)  
- [JetBrains Research Paper (arXiv)](https://arxiv.org/pdf/2406.11612)  
- [APPS Benchmark](https://arxiv.org/pdf/2105.09938)

## Installation
The required dependencies are in the ```requirements.txt``` file.

# Fine-tuning algorithm 
We use QLoRA fine-tuning on the Llama-3.2-1B model on python code examples, to enhace the model's programming and try to make it write a code more similar to the model solutions.



## Code Implementation
We use QLoRA fine-tuning on the Llama-3.2-1B model on python code examples, to enhance the model's programming capabilities and encourage code generation that more closely mirrors model solutions.
### Dataset Preparation -- ```preprocess.py```
**APPS Dataset Processing:** 

We use the [APPS](https://arxiv.org/pdf/2105.09938) dataset which consists of python coding problems
- Filters problems with:
  - At least 1 solution
  - Valid input/output examples
  - Questions <500 characters
  - Solutions <400 lines
- Formats prompts using structured template:
  ```python
  "Write Python code to solve this problem:\nProblem: {question}\nInput: {input}\nOutput: {output}\nConstraints..." 
  ```
  Due to the limits in the resources, we restrict ourselves to only 500 examples.

  The file also contains the function to properly tokenize the prompts.


### Training

The implementation provides an efficient fine-tuning pipeline for code generation using Meta's Llama-3.2-1B model with Parameter-Efficient Fine-Tuning (PEFT). The system combines 4-bit quantization and Low-Rank Adaptation (LoRA) to enable cost-effective training while maintaining performance.


### 1. Configuration Settings
**Model Architecture:**
```python
MODEL_NAME = "meta-llama/Llama-3.2-1B"
```
**Quantization setup**
```python
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```
Features:
- 4-bit NormalFloat quantization

- Mixed precision computation (FP16)

- Reduces memory footprint by ~75%

**QLoRA setup**

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

- Rank-16 adaptation matrices

- Attention projection layer targeting

- Dropout for regularization

- Quantization (which might not be as helpful in our case of small dataset)

**Training Parameters**

```python
TRAINING_ARGS = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 7e-5,
    "fp16": True,
    "bf16": False,  # Ensure you're not using both bf16 and fp16
    "optim": "adamw_bnb_8bit",  # Use memory-efficient optimizer
    "fp16_full_eval": True,  # FP16 for evaluation
    "eval_accumulation_steps": 1,
    "logging_steps": 20,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "remove_unused_columns": False,
    "report_to": "none"
}
```
- This arguments are optimized to save vram memory.
- Effective batch size: 16 (8 × 2 accumulation)
- Conservative learning rate for stable adaptation
- 10 fine tuning epochs

### Evalution
After each epoch in addition to the validation loss a [CHRF](https://huggingface.co/spaces/evaluate-metric/chrf) metric was compute in order to measure the text similarity between input and output.

Additionally, the ```eval.py``` file contains a function to compute the CHRF metric and perform an actual Python evaluation. This includes output comparisons of generated code on a couple of random examples.

### Remarks

Generally speaking, I was limited by the computing power (T4 GPU with 15 GB vram), as I have used the free version of Google Collab.
The attached notebook contains the code which was actually used to obtain results.

One think worth noting is that the ```transformers``` library contains a memory leakage when using compute metrics parameter. However, I have managed to solve this issue.

Retrieval techniques were not used, as finding appropriate context for algorithmic programming would have required significant additional effort.