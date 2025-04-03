import json
import torch
from datasets import load_dataset, Dataset
from src.config import DATASET_NAME


def safe_parse(json_str):
    """Robust JSON parsing with error handling"""
    try:
        return json.loads(json_str) if json_str.strip() else None
    except json.JSONDecodeError:
        return None

def format_prompt(example):
    """Prompt structure for code generation"""

    return f"""Write Python code to solve this problem:
Problem: {example['question']}
Input: {example['input_output'].get('inputs', [''])[0]}
Output: {example['input_output'].get('outputs', [''])[0]}
Your solution must:
- Be under 400 lines
- Use efficient algorithms
- Include proper error handling

Solution Code:
"""

def load_filtered_apps_dataset(split="train", max_samples=500):
    """Dataset loading with filtering"""
    """Only 500 samples of short questions with short answers """
    ds = load_dataset(DATASET_NAME, split=split, trust_remote_code=True)

    filtered = []
    for ex in ds:
        try:
            ex['solutions'] = safe_parse(ex['solutions']) or []
            ex['input_output'] = safe_parse(ex['input_output']) or {'inputs': [], 'outputs': []}

            if (len(ex['solutions']) > 0 and
                len(ex['input_output']['inputs']) > 0 and
                len(ex['question']) < 500 and
                any(len(sol) < 400 for sol in ex['solutions'])):

                filtered.append({
                    'prompt': format_prompt(ex),
                    'solution': ex['solutions'][0]
                })

        except Exception as e:
            print(f"Skipping {ex['problem_id']}: {str(e)}")

    return Dataset.from_list(filtered[:max_samples])


# Preprocessing with proper labels
def preprocess(examples, tokenizer):
    texts = [p + s for p, s in zip(examples["prompt"], examples["solution"])]

    # Tokenize text
    tokenized = tokenizer(
        texts,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Compute prompt lengths
    prompt_lens = [
        len(tokenizer(p, add_special_tokens=False)["input_ids"])
        for p in examples["prompt"]
    ]

    # Mask prompt tokens in labels
    labels = []
    for input_ids, plen in zip(tokenized["input_ids"], prompt_lens):
        label = torch.full(input_ids.shape, -100, dtype=torch.long)  # Initialize with -100
        label[plen:] = input_ids[plen:]  # Only keep the solution part
        labels.append(label)

    # Convert labels list to tensor
    tokenized["labels"] = torch.stack(labels)

    return tokenized