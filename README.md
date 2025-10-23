 🧠 distillation_with_vllm

A complete end-to-end **knowledge distillation pipeline** using [vLLM](https://github.com/vllm-project/vllm).  
This project runs a **large teacher model** to generate synthetic instruction–response data and log-probabilities,  
then trains a **small student model (<1B)** to imitate and distill that teacher’s knowledge efficiently on a single GPU.

---

## 🚀 Overview

The pipeline is divided into three main stages:

1. **collect** – Use a large teacher model (via vLLM) to generate instruction–response pairs and token-level logprobs.  
2. **train_sft** – Fine-tune a small student model using the teacher’s responses as hard labels (Cross-Entropy).  
3. **train_kd** – Perform Knowledge Distillation training with **KL divergence** between teacher and student token distributions.  

After training, two evaluation scripts are provided:

- **eval_bench.py** – benchmark accuracy test (PIQA, HellaSwag, etc.)
- **eval_similarity.py** – teacher/student semantic similarity check (BERTScore)

---

## 🗂 Project Structure

distillation_with_vllm/
├── main.py # Lightweight launcher script
├── requirements.txt
├── configs/
│ ├── collect.yaml # Teacher inference configs
│ ├── train_sft.yaml # Supervised fine-tuning configs
│ ├── train_kd.yaml # Knowledge Distillation configs
├── app/
│ ├── utils.py # Config, JSONL, logging utilities
│ ├── hosting.py # vLLM wrapper for teacher inference
│ ├── collect.py # Data generation pipeline
│ ├── train_sft.py # Student SFT training (CrossEntropy)
│ ├── train_kd.py # Student KD training (CE + KL)
│ └── cli.py # Typer-based optional CLI
├── data/
│ ├── distilled/ # Generated teacher data
│ └── eval/
├── eval_bench.py # Benchmark accuracy evaluation
└── eval_similarity.py # Semantic similarity (BERTScore)

yaml
코드 복사

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Prepare configs
Modify the files under configs/ to adjust:

model names (teacher/student)

dataset (Alpaca, FLAN, etc.)

batch size, epochs, learning rate

🧭 Execution Flow
Step 1: Collect teacher data
bash
코드 복사
python main.py collect
Runs the teacher model via vLLM, generates instruction–response pairs,
and saves results (including logprobs) into data/distilled/.

Step 2: Supervised Fine-tuning (SFT)
bash
코드 복사
python main.py train_sft
Trains the student model on teacher outputs using hard labels (CrossEntropy).

Step 3: Knowledge Distillation (KD)
bash
코드 복사
python main.py train_kd
Refines the student by matching teacher’s token-level probability distributions (KL loss).

🧪 Evaluation
1. Benchmark Accuracy
bash
코드 복사
python eval_bench.py --model artifacts/qwen05b_kd/final --tasks piqa,hellaswag
2. Teacher–Student Semantic Similarity
bash
코드 복사
python eval_similarity.py \
  --teacher meta-llama/Llama-3.2-3B-Instruct \
  --student artifacts/qwen05b_kd/final \
  --sample 500
📈 Expected Results
Metric	Description	Target
Accuracy	Benchmark tasks (PIQA, HellaSwag)	≥ 90% of teacher
BERTScore F1	Semantic similarity with teacher	≥ 0.85
KL loss	Distribution alignment	steadily decreasing
Speedup	vLLM latency (student vs teacher)	≥ 2× faster

🧩 Design Philosophy
Minimal but modular: 5 Python modules cover all core functionality.

Extensible: You can easily replace datasets or add new evaluation benchmarks.

Runnable on consumer GPUs: Designed for RTX 4060 / 4070-class hardware.

Self-contained: No OpenAI API or external inference server required.

🔧 Notes for AI Code Assistants (like Jules / Gemini)
This README intentionally describes:

The role of each file and function,

Execution order (collect → train_sft → train_kd → eval),

Required configs and paths,

CLI commands and expected outputs.

An AI code assistant can safely use this README to:

Understand where the entry point is (main.py).

Map imports like app.collect, app.train_sft, etc.

Verify configuration file locations.

Automatically execute or debug each pipeline stage.

Suggest modifications or improvements (e.g., adding init.py or updating paths).

If you are Jules, read this document fully before editing code.
Your job is to ensure that python main.py commands run successfully for all stages without manual intervention.

🧠 Example: Full Pipeline
bash
코드 복사
# 1. Teacher generation
python main.py collect

# 2. Student SFT
python main.py train_sft

# 3. Knowledge Distillation
python main.py train_kd

# 4. Evaluation
python eval_bench.py --model artifacts/qwen05b_kd/final --tasks piqa,hellaswag
python eval_similarity.py --teacher meta-llama/Llama-3.2-3B-Instruct --student artifacts/qwen05b_kd/final --sample 500
🧾