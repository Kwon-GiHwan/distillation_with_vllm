from datasets import load_dataset
from typing import List, Dict, Any
from app.hosting import VLLMEngine
from app.utils import jsonl_writer, log

def _alpaca_prompts(n: int) -> List[str]:
    ds = load_dataset("tatsu-lab/alpaca_farm", "alpaca_instructions", split="train")
    ds = ds.select(range(min(n, len(ds))))
    prompts = []
    for ex in ds:
        inst = ex["instruction"]
        inp = ex.get("input", "")
        if inp:
            p = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:"
        else:
            p = f"### Instruction:\n{inst}\n\n### Response:"
        prompts.append(p)
    return prompts

DATASETS = {
    "alpaca": _alpaca_prompts,
    # "flan": lambda n: ...,  # 이후 확장 예정
}

def run_collect(cfg: Dict[str, Any]) -> str:
    make_prompts = DATASETS[cfg["dataset"]["name"]]
    prompts = make_prompts(cfg["dataset"]["n_samples"])

    engine = VLLMEngine(
        model=cfg["hosting"]["model"],
        dtype=cfg["hosting"]["dtype"],
        quantization=cfg["hosting"]["quantization"],
        gpu_memory_utilization=cfg["hosting"]["gpu_memory_utilization"],
    )

    results = engine.generate(
        prompts,
        temperature=cfg["sampling"]["temperature"],
        top_p=cfg["sampling"]["top_p"],
        max_tokens=cfg["sampling"]["max_tokens"],
        logprobs=cfg["sampling"]["logprobs"],
    )

    out_path = cfg["output"]["path"]
    with jsonl_writer(out_path) as f:
        for r in results:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    log(f"[collect] saved -> {out_path}")
    return out_path
