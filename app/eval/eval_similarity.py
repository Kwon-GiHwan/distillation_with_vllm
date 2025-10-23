# eval_similarity.py
"""
Teacher/Student 모델의 생성 결과 의미 유사도 평가 (BERTScore 기반)
"""

import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from bert_score import score as bert_score


def generate_batch(model, tok, prompts, max_new_tokens=128, temperature=0.7):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    texts = tok.batch_decode(out, skip_special_tokens=True)
    return texts


def run_eval_similarity(teacher_path: str, student_path: str, sample_size: int = 1000, dataset: str = "tatsu-lab/alpaca_farm"):
    """
    Run similarity evaluation between a teacher and student model.

    Args:
        teacher_path (str): Path to the teacher model.
        student_path (str): Path to the student model.
        sample_size (int, optional): Number of samples to generate. Defaults to 1000.
        dataset (str, optional): Dataset to use for generating prompts. Defaults to "tatsu-lab/alpaca_farm".
    """
    ds = load_dataset(dataset, "alpaca_instructions", split="validation")
    prompts = []
    for ex in ds.select(range(min(sample_size, len(ds)))):
        inst = ex["instruction"]
        inp = ex.get("input", "")
        if inp:
            p = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:"
        else:
            p = f"### Instruction:\n{inst}\n\n### Response:"
        prompts.append(p)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Running Similarity Evaluation ---")
    print(f"Generating with teacher: {teacher_path}")
    tok_t = AutoTokenizer.from_pretrained(teacher_path, use_fast=True)
    model_t = AutoModelForCausalLM.from_pretrained(
        teacher_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    outs_t = generate_batch(model_t, tok_t, prompts)

    print(f"Generating with student: {student_path}")
    tok_s = AutoTokenizer.from_pretrained(student_path, use_fast=True)
    model_s = AutoModelForCausalLM.from_pretrained(
        student_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    outs_s = generate_batch(model_s, tok_s, prompts)

    print("Computing BERTScore ...")
    P, R, F1 = bert_score(outs_s, outs_t, lang="en", verbose=True)
    
    results = {
        "f1": F1.mean().item(),
        "precision": P.mean().item(),
        "recall": R.mean().item(),
    }
    
    print(f"Mean F1: {results['f1']:.4f}")
    print(f"Mean Precision: {results['precision']:.4f}")
    print(f"Mean Recall: {results['recall']:.4f}")
    print("--- Similarity Evaluation Finished ---")
    return results
