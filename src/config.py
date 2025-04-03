import os
import sys

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig



sys.set_int_max_str_digits(100000)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model and quantization settings
MODEL_NAME = "meta-llama/Llama-3.2-1B"
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# LoRA settings for Llama
PEFT_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training hyperparameters
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


DATASET_NAME = "codeparrot/apps"
