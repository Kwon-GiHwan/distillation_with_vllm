# eval_bench.py
"""
Teacher/Student 모델을 표준 벤치마크(PIQA, HellaSwag 등)에서 평가.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from evaluate import load as load_metric


TASKS = {
    "piqa": ("piqa", None, "goal", "sol1", "sol2"),
    "hellaswag": ("hellaswag", None, "ctx", "endings", None),
}


def get_choice(logits, option_ids, mask_value=-1e9):
    # logits: [V], option_ids: [[tok1,tok2,...], ...]
    scores = []
    for ids in option_ids:
        if not ids:
            scores.append(mask_value)
            continue
        token_logits = logits[ids]
        scores.append(token_logits.mean().item())
    return int(torch.tensor(scores).argmax())


def evaluate_task(model, tok, device, task_name, batch_size):
    dataset_name, subset, context_col, opt1, opt2 = TASKS[task_name]
    ds = load_dataset(dataset_name, subset, split="validation")
    acc_metric = load_metric("accuracy")

    model.eval()
    for ex in tqdm(ds, desc=f"Evaluating {task_name}"):
        if task_name == "piqa":
            ctx = ex[context_col]
            opts = [ex[opt1], ex[opt2]]
        elif task_name == "hellaswag":
            ctx = ex[context_col]
            opts = ex["endings"]

        ctx_ids = tok(ctx, return_tensors="pt").input_ids.to(device)
        opt_ids = [tok(o, add_special_tokens=False).input_ids[0] for o in opts]

        with torch.no_grad():
            out = model(ctx_ids)
            logits = out.logits[0, -1, :]  # 마지막 토큰 기준

        pred = get_choice(logits, opt_ids)
        label = int(ex["label"])
        acc_metric.add(prediction=pred, reference=label)

    result = acc_metric.compute()
    print(f"Task: {task_name} | Accuracy: {result['accuracy']:.4f}")
    return result["accuracy"]


def run_eval_bench(model_path: str, tasks: str = "piqa,hellaswag", batch_size: int = 4):
    """
    Run benchmark evaluation on a given model.

    Args:
        model_path (str): Path to the model to evaluate.
        tasks (str, optional): Comma-separated list of tasks. Defaults to "piqa,hellaswag".
        batch_size (int, optional): Batch size for evaluation. Defaults to 4.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )

    total = []
    task_list = [t.strip() for t in tasks.split(",") if t.strip() in TASKS]
    
    print(f"--- Running Benchmark Evaluation for {model_path} ---")
    for t in task_list:
        acc = evaluate_task(model, tok, device, t, batch_size)
        total.append(acc)
    
    if total:
        avg_acc = sum(total) / len(total)
        print(f"Average Accuracy: {avg_acc:.4f}")
        
    print("--- Benchmark Evaluation Finished ---")
    return {task: acc for task, acc in zip(task_list, total)}
