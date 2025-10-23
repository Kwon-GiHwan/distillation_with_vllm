from vllm import LLM, SamplingParams
from typing import List, Dict, Any

class VLLMEngine:
    """vLLM 인프로세스 래퍼: prompts -> [{prompt, text, logprobs}]"""
    def __init__(self, model: str, dtype: str = "auto",
                 quantization: str | None = "bitsandbytes",
                 gpu_memory_utilization: float = 0.9):
        self.llm = LLM(
            model=model,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate(self, prompts: List[str], **sampling) -> List[Dict[str, Any]]:
        sp = SamplingParams(**sampling)  # temperature, top_p, max_tokens, logprobs…
        outs = self.llm.generate(prompts, sp)
        results = []
        for p, o in zip(prompts, outs):
            choice = o.outputs[0]
            results.append({
                "prompt": p,
                "text": choice.text,
                "logprobs": getattr(choice, "logprobs", None),
            })
        return results
