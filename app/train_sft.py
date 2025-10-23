import json, torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments)

class JSONLDistill(Dataset):
    def __init__(self, path: str, tok, max_len: int):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tok
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        text = f"{ex['prompt']}\n{ex['text']}"
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0]}

class Collator:
    def __init__(self, tok): self.tok = tok
    def __call__(self, batch):
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        ids = [b["input_ids"] for b in batch]
        att = [b["attention_mask"] for b in batch]
        ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad)
        att = torch.nn.utils.rnn.pad_sequence(att, batch_first=True, padding_value=0)
        return {"input_ids": ids, "attention_mask": att, "labels": ids}

def run_train_sft(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["student_model"], use_fast=True)
    tok.pad_token = tok.eos_token
    ds = JSONLDistill(cfg["data_path"], tok, cfg["max_len"])

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

    trainer = Trainer(
        model=model, args=args, train_dataset=ds, data_collator=Collator(tok)
    )
    trainer.train()
    model.save_pretrained(cfg["final_dir"])
    tok.save_pretrained(cfg["final_dir"])
