import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, \
    DataCollatorForLanguageModeling, TrainerCallback
from peft import get_peft_model, prepare_model_for_kbit_training
from src.config import MODEL_NAME, BNB_CONFIG, PEFT_CONFIG, TRAINING_ARGS
from src.preprocess import preprocess, load_filtered_apps_dataset
import evaluate
from src.eval import do_evaluation

# Get an API Key
HF_API_KEY = os.getenv("HF_API_KEY")
if HF_API_KEY is None:
    try:
        from google.colab import userdata
        HF_API_KEY = userdata.get("HF_API_KEY")
        print("Using API key from Colab userdata.")
    except ImportError:
        print("No API key found.")

# Load Model & Tokenizer
login(token=HF_API_KEY)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BNB_CONFIG,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model) #Using QLoRA
model = get_peft_model(model, PEFT_CONFIG)
model.gradient_checkpointing_enable()






# Load dataset and apply preprocessing
dataset = load_filtered_apps_dataset()

tokenized_data = dataset.map((lambda x : preprocess(x, tokenizer)) , batched=True, remove_columns=["prompt", "solution"])
# Split into train and validation dataset (named "train" and "test").
# The validation dataset will be used as a benchmarking task, as this is only a demonstration example.
tokenized_data = tokenized_data.train_test_split(test_size=0.1)




# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    **TRAINING_ARGS
)


# Loading CHRF
chrf = evaluate.load("chrf")


# Validation metric
def compute_metrics(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)


    # Compute CHRF++
    results = chrf.compute(
        predictions=pred_str,
        references=label_str,
        word_order=2,
        beta=2
    )
    return {"chrf": results["score"]}


# This function is needed to prevented a bugged memory leakage in the transformers library
# Taken from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/16
def preprocess_logits_for_metrics(logits, labels):

    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics = compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics
)


# Clear cache after each epoch to save memory
class ClearCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print("Cleared GPU cache at the end of epoch", state.epoch)
        return control


trainer.add_callback(ClearCacheCallback())


# Train the model
trainer.train()

# Save model
model.save_pretrained("./results/fine_tuned_model")
tokenizer.save_pretrained("./results/fine_tuned_model")

do_evaluation(model, tokenizer)
