import json, math, torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)

# --------- 유틸: sparse top-k 분포 → (ids, probs)로 변환 ----------
def _normalize_probs(logprobs: List[float], eps: float = 1e-8) -> List[float]:
    # logprobs → probs, 합=1 (안정화)
    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    s = sum(exps) + eps
    return [x / s for x in exps]

def _teacher_dist_to_student_ids(
    entries: List[Dict[str, Any]],  # [{"token":"...", "logprob":-0.3}, ...]
    tokenizer
) -> Tuple[List[int], List[float]]:
    """
    teacher top-k 엔트리(토큰 문자열 기반)를 student vocab id로 매핑.
    매핑 실패 토큰은 드롭. 결과 probs는 정규화해서 합=1 유지.
    """
    ids, lps = [], []
    for e in entries:
        tok_str = e.get("token") or e.get("text") or ""
        if tok_str is None:
            continue
        # tokenizer.convert_tokens_to_ids 는 "토큰 문자열"을 기대함.
        # vLLM의 token text가 공백/특수문자 포함일 수 있으니 그대로 시도.
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid is None or tid == tokenizer.unk_token_id:
            # 실패하면 한 번 더: 토크나이저를 통해 텍스트 → 토큰화 후 첫 토큰 id 사용
            enc = tokenizer(tok_str, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
            if len(enc["input_ids"]) == 0:
                continue
            tid = enc["input_ids"][0]
        ids.append(int(tid))
        lps.append(float(e.get("logprob", -1e9)))

    if not ids:
        return [], []

    probs = _normalize_probs(lps)
    return ids, probs

# --------- Dataset: teacher logprobs(K) + student labels ----------
class KDJSONLDataset(Dataset):
    """
    JSONL 레코드 예시:
    {
      "prompt": "...",
      "text": "...",             # teacher가 생성한 텍스트
      "logprobs": [              # vLLM top-k per-token (생성 토큰 길이와 동일)
         {
           "top_logprobs": [
              {"token": " The", "logprob": -0.1}, {"token":" A", "logprob":-1.3}, ...
           ]
         },
         ...
      ]
    }
    """
    def __init__(self, path: str, tokenizer, max_len: int):
        with open(path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line.strip()) for line in f if line.strip()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        prompt, gen = ex["prompt"], ex["text"]
        # 입력은 prompt + generation 모두 포함해 LM 표준 방식으로 라벨 = 입력 시프트
        text = f"{prompt}\n{gen}"
        enc = self.tok(
            text, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return {"input_ids": enc["input_ids"][0], "labels": enc["input_ids"][0]}


def collate(batch):
    import torch
    keys = batch[0].keys()
    out = {k: torch.stack([torch.tensor(x[k]) for x in batch]) for k in keys}
    out["labels"] = out["labels"].long()
    return out

# --------- KD Trainer ----------
class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, kd_alpha=0.5, kd_temp=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher_model is not None
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.kd_alpha, self.kd_temp = kd_alpha, kd_temp

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        if not torch.is_tensor(labels):
            raise TypeError("'labels' must be a torch.Tensor")

        outputs_s = model(**inputs)
        logits_s = outputs_s.logits

        with torch.no_grad():
            outputs_t = self.teacher(**{k: v for k, v in inputs.items() if k != "labels"})
            logits_t = outputs_t.logits

        ce_loss = F.cross_entropy(
            logits_s.view(-1, logits_s.size(-1)), labels.view(-1), ignore_index=-100
        )
        t = self.kd_temp
        kd_loss = F.kl_div(
            F.log_softmax(logits_s / t, dim=-1),
            F.softmax(logits_t / t, dim=-1),
            reduction="batchmean",
        ) * (t**2)

        loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * ce_loss
        return (loss, outputs_s) if return_outputs else loss


# --------- 엔트리 ----------
def run_train_kd(cfg: Dict[str, Any]):
    tok = AutoTokenizer.from_pretrained(cfg["student_model"], use_fast=True)
    tok.pad_token = tok.eos_token

    ds = KDJSONLDataset(cfg["data_path"], tok, cfg["max_len"])

    student_model = AutoModelForCausalLM.from_pretrained(
        cfg["student_model"], torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto"
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg["teacher_model"], torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto"
    )

    args = TrainingArguments(
        output_dir=cfg["out_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg["scheduler"],
        warmup_ratio=cfg["warmup"],
        bf16=cfg.get("bf16", True),
        logging_steps=cfg["logging_steps"],
        save_strategy="no",
        remove_unused_columns=False,
        report_to=[],
        no_cuda=True,
    )

    trainer = KDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
        kd_alpha=cfg.get("kd_alpha", 0.5),
        kd_temp=cfg.get("kd_temp", 1.0),
    )
    trainer.train()
    trainer.save_model(cfg["final_dir"])
    tok.save_pretrained(cfg["final_dir"])
