import json, math, torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
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
              {"token": "▁The", "logprob": -0.1}, {"token":"▁A", "logprob":-1.3}, ...
           ]
         },
         ...
      ]
    }
    """
    def __init__(self, path: str, tokenizer, max_len: int):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
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

        # --- 생성 구간 마스크 ---
        # 간단 근사: student 토크나이저로 다시 디코딩해서 prompt 길이 추정
        # 보다 정확히 하려면 prompt만 tokenize해서 길이를 써도 됨.
        enc_prompt = self.tok(prompt + "\n", add_special_tokens=False, return_tensors="pt")
        prompt_len = min(enc_prompt["input_ids"].size(1), enc["input_ids"].size(1))
        seq_len = enc["input_ids"].size(1)
        gen_mask = torch.zeros(seq_len, dtype=torch.bool)
        gen_mask[prompt_len:] = True  # prompt 이후만 생성 구간으로 간주

        # --- teacher sparse top-k 분포 (길이 정렬) ---
        # vLLM의 logprobs 길이 = teacher 생성 토큰 길이
        # student 토큰 경계와 다를 수 있으므로 "최소 길이"만큼만 KL 계산
        teacher_lp = ex.get("logprobs", None)
        if isinstance(teacher_lp, list):
            teacher_k = []
            for step in teacher_lp:
                top = step.get("top_logprobs") or step.get("top") or []
                teacher_k.append(top)
        else:
            teacher_k = None

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "gen_mask": gen_mask,
            "teacher_topk": teacher_k,  # List[List[{token, logprob}]] or None
        }

@dataclass
class CollateOut:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    gen_mask: torch.Tensor
    teacher_topk: List[Optional[List[List[Dict[str, Any]]]]]

class Collator:
    def __init__(self, tok): self.tok = tok
    def __call__(self, batch) -> CollateOut:
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        ids = [b["input_ids"] for b in batch]
        att = [b["attention_mask"] for b in batch]
        gen = [b["gen_mask"] for b in batch]
        max_len = max(x.size(0) for x in ids)

        ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad)
        att = torch.nn.utils.rnn.pad_sequence(att, batch_first=True, padding_value=0)
        gen = torch.nn.utils.rnn.pad_sequence(gen, batch_first=True, padding_value=0)

        labels = ids.clone()  # LM 표준: 입력 전체를 레이블로 (shift는 모델 내부)
        return CollateOut(
            input_ids=ids, attention_mask=att, labels=labels,
            gen_mask=gen, teacher_topk=[b["teacher_topk"] for b in batch]
        )

# --------- KL(KD) 계산 ----------
def kd_sparse_topk_kl(
    student_logits: torch.Tensor,  # [B, L, V]
    teacher_topk_batch: List[Optional[List[List[Dict[str, Any]]]]],
    gen_mask: torch.Tensor,        # [B, L] (생성 구간만 True)
    tokenizer,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    teacher_topk_batch[b] = None 또는 길이 T의 리스트(각 step당 top_k 엔트리 리스트)
    KL은 (teacher 생성토큰 길이)와 (student 시퀀스 길이) 중 최소 길이만 계산.
    """
    B, L, V = student_logits.shape
    # 온도 스케일
    if temperature != 1.0:
        student_logits = student_logits / temperature

    log_softmax = torch.nn.functional.log_softmax(student_logits, dim=-1)
    total = student_logits.new_tensor(0.0)
    count = 0

    for b in range(B):
        topk_steps = teacher_topk_batch[b]
        if not topk_steps:
            continue

        # student 쪽에서 생성영역 인덱스만 취득
        gen_indices = torch.nonzero(gen_mask[b], as_tuple=False).squeeze(-1).tolist()
        # teacher 생성 스텝 길이
        Tt = len(topk_steps)
        # KL 계산 길이는 두 쪽의 최소 길이
        T = min(Tt, len(gen_indices))
        if T <= 0:
            continue

        for t in range(T):
            step_entries = topk_steps[t] or []
            ids, probs = _teacher_dist_to_student_ids(step_entries, tokenizer)
            if not ids:
                continue

            # student 분포에서 해당 ids만 샘플링
            idx = gen_indices[t]
            s_logp = log_softmax[b, idx, ids]   # [k]
            t_p = student_logits.new_tensor(probs)  # [k], 합=1

            # KL(P_t || P_s) = sum_i p_t(i) * (log p_t(i) - log p_s(i))
            t_logp = torch.log(t_p + eps)
            kl = torch.sum(t_p * (t_logp - s_logp))
            total = total + kl
            count += 1

    if count == 0:
        return student_logits.new_tensor(0.0)
    return total / count

# --------- KD Trainer ----------
class KDTrainer(Trainer):
    def __init__(self, *args, alpha_kd: float = 0.1, temperature: float = 1.0, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_kd = alpha_kd
        self.temperature = temperature
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs: CollateOut(dict로 들어옴)
        teacher_topk = inputs.pop("teacher_topk")
        gen_mask = inputs.pop("gen_mask")

        outputs = model(**inputs)
        ce_loss = outputs.loss

        # KD KL
        with torch.no_grad():
            # forward에서 이미 logits 계산됨. (outputs.logits: [B,L,V])
            pass

        kd = kd_sparse_topk_kl(
            outputs.logits, teacher_topk, gen_mask,
            tokenizer=self.tokenizer, temperature=self.temperature
        )
        loss = ce_loss + self.alpha_kd * kd
        return (loss, outputs) if return_outputs else loss

# --------- 엔트리 ----------
def run_train_kd(cfg: Dict[str, Any]):
    tok = AutoTokenizer.from_pretrained(cfg["student_model"], use_fast=True)
    tok.pad_token = tok.eos_token

    ds = KDJSONLDataset(cfg["data_path"], tok, cfg["max_len"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg["student_model"], torch_dtype=torch.bfloat16,
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
        bf16=True,
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=2,
    )

    trainer = KDTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=Collator(tok),
        alpha_kd=cfg.get("alpha_kd", 0.1),
        temperature=cfg.get("temperature", 1.0),
        tokenizer=tok
    )
    trainer.train()
    model.save_pretrained(cfg["final_dir"])
    tok.save_pretrained(cfg["final_dir"])
