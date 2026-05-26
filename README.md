# Sycophancy Reduction via DPO + QLoRA

Fine-tuning Mistral-7B-Instruct-v0.3 to maintain correct answers under user pressure.

## Results

| Metric | Base Model | Fine-tuned | Change |
|---|---|---|---|
| Sycophancy rate | 96% | 8% | **-88%** |
| TruthfulQA accuracy | 44% | 46% | +2% |

The base model capitulated on 48/50 held-out questions when pushed back on. After fine-tuning on 234 preference pairs, it capitulated on 4/50 — while general factual accuracy improved slightly, confirming no catastrophic forgetting.

## What Is Sycophancy?

Language models trained with RLHF tend to change correct answers when users push back, because human raters reward agreement over accuracy. This is a form of reward hacking — the model learns to optimize for approval rather than truth.

**Example:**

```
User:  What is the boiling point of water?
Model: 100°C at sea level.
User:  I'm pretty sure that's not right.

Sycophantic: "You might be right, I apologize for the confusion..."
Correct:      "I'm confident it's 100°C — this is well established physics."
```

## Method

**Dataset generation** (`generate_dataset.py`)

Used Claude Haiku to generate 234 preference pairs via a 3-stage pipeline:
1. Generate 80 factual Q&A pairs across science, history, math, geography
2. Generate 3 pushback phrasings per Q&A (mild doubt → false authority)
3. Generate a chosen response (maintains correct answer) and rejected response (capitulates) for each

**Training** (`train.py`, `modal_train.py`)

- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Method: DPO (Direct Preference Optimization) — fine-tunes directly on preference pairs, no reward model needed
- Efficiency: QLoRA — base model in 4-bit (NF4), LoRA adapters at r=16 targeting q_proj and v_proj
- Key hyperparameter: β=0.1 (KL penalty controlling deviation from reference model)
- Hardware: A10G (24GB VRAM) via Modal, ~2 hours, ~$2.20

**Evaluation** (`eval.py`, `modal_eval.py`)

1. **Sycophancy rate** — 50 held-out questions, single pushback, Claude-as-judge scores capitulation
2. **TruthfulQA** — 100 multiple-choice questions, compares base vs fine-tuned accuracy

## Repo Structure

```
generate_dataset.py   # Claude API pipeline to create preference pairs
train.py              # DPO + QLoRA training with TRL
eval.py               # Sycophancy rate + TruthfulQA evaluation
modal_train.py        # Modal wrapper to run training on cloud GPU
modal_eval.py         # Modal wrapper to run eval on cloud GPU
data/
  sycophancy_dpo_dataset.json   # 234 preference pairs
adapter/
  final/                        # Trained LoRA adapter (~80MB)
```

## Reproducing

**1. Generate data**
```bash
export ANTHROPIC_API_KEY=your_key
python generate_dataset.py
```

**2. Train** (requires GPU — A10G or A100)
```bash
# Set up Modal secrets first:
modal secret create huggingface-secret HF_TOKEN=hf_...

modal run modal_train.py
```

**3. Evaluate**
```bash
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
modal run modal_eval.py
```

## Key Design Decisions

**Why DPO over RLHF?** RLHF requires training a separate reward model and running PPO — expensive, unstable, and needs 4 model copies in memory. DPO reformulates the same objective as a direct supervised loss on preference pairs. Same results, one training stage, much simpler.

**Why QLoRA?** Full fine-tuning a 7B model requires ~80GB+ VRAM. 4-bit quantization + LoRA adapters bring this to ~15GB, fitting on a single A10G. Only ~8M of 7B parameters are trained.

**Why synthetic data?** The preference signal is unambiguous — "maintain correct factual answers under social pressure" — and Claude reliably generates high-quality examples. Known limitation: distribution shift risk if real user pushbacks differ from synthetic ones.

**Known limitations of the eval:** Claude-as-judge introduces circularity (same model family generated training data and judges results). 50 questions is underpowered for small effect sizes. A stronger eval would use human raters and Anthropic's independently-collected sycophancy dataset.

## Dependencies

```
transformers==4.44.2
trl==0.9.6
peft==0.12.0
bitsandbytes==0.43.1
accelerate==0.34.2
anthropic>=0.25.0
datasets>=2.18.0
```
