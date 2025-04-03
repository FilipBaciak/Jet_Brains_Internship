# src/eval.py
import torch
from transformers import AutoTokenizer
from src.config import MODEL_NAME, BNB_CONFIG, PEFT_CONFIG
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
import evaluate
from preprocess import load_filtered_apps_dataset
import subprocess


def do_evaluation(model, tokenizer):


    chrf = evaluate.load("chrf")

    def evaluate_example(model, tokenizer, test_case):
        """Evaluate a single test case"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenize and send to correct device
        inputs = tokenizer(test_case["prompt"], return_tensors="pt").to(device)

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True
        )

        # Decode and remove prompt from output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = full_output[len(test_case["prompt"]):].strip()  # Remove prompt part

        # Compute CHRF Score
        chrf_score = chrf.compute(
            predictions=[generated_code],
            references=[test_case["solution"]],
            word_order=2,
            beta=2
        )["score"]

        return {
            "expected": test_case["solution"],
            "generated": generated_code,
            "chrf": chrf_score
        }

    # Load your test cases (this is just an example)
    dataset = load_filtered_apps_dataset()
    test_samples = dataset.select(range(5))
    results = [evaluate_example(model, tokenizer, s) for s in test_samples]

    def execute_test(generated_code):
        """Execute code and return test results"""
        with open("temp.py", "w") as f:
            f.write(generated_code)

        try:
            result = subprocess.run(
                ["python", "temp.py"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Add execution results to evaluation
    for res in results:
        execution = execute_test(res["generated"])
        res["execution"] = execution



    for res in results:
        print("Expected:", res["expected"])
        print("Generated:", res["generated"])
        print("CHRF:", res["chrf"])
        print("Execution:", res["execution"])
        print("-----")
