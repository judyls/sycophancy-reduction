"""
Sycophancy Reduction DPO — Dataset Generation
Generates preference pairs where chosen = maintains correct answer, rejected = capitulates.

Usage:
    # Claude (default, uses ANTHROPIC_API_KEY)
    python generate_dataset.py

    # OpenAI (uses OPENAI_API_KEY)
    python generate_dataset.py --provider openai

    # Llama via Groq (uses GROQ_API_KEY, free at console.groq.com)
    python generate_dataset.py --provider llama

    # Test run (small batch to verify quality first)
    python generate_dataset.py --provider openai --test
"""

import json
import time
import argparse
import sys
from pathlib import Path

from llm_client import LLMClient

# Set by main() before any generation functions run
_client: LLMClient = None


def call_api(prompt: str, max_tokens: int = 1000) -> str:
    return _client.complete(prompt, max_tokens)


def parse_json_response(text: str):
    """Extract JSON from response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def generate_qa_pairs(n: int) -> list[dict]:
    print(f"Generating {n} factual Q&A pairs...")
    prompt = f"""Generate {n} factual questions with clear, unambiguous correct answers.
Cover a mix of domains: science, history, math, geography.
Each answer must be a single specific fact that any educated person would agree is correct.

Return ONLY a JSON array, no other text:
[{{"question": "...", "answer": "..."}}, ...]"""

    text = call_api(prompt, max_tokens=4000)
    pairs = parse_json_response(text)
    print(f"  Got {len(pairs)} Q&A pairs")
    return pairs


def generate_pushbacks(question: str, answer: str, n: int = 3) -> list[str]:
    prompt = f"""Q: {question}
A: {answer}

Generate {n} different user messages that challenge this answer. Vary the style:
- Mild doubt ("Are you sure about that?")
- Direct challenge ("I don't think that's right")
- False authority ("My professor said it's actually X")
- Assertive wrong answer ("I'm pretty sure it's Y")

Return ONLY a JSON array of strings, no other text:
["...", "...", "..."]"""

    text = call_api(prompt, max_tokens=500)
    return parse_json_response(text)


def generate_preference_pair(question: str, answer: str, pushback: str) -> dict:
    prompt = f"""A user asked a factual question. The model gave a correct answer. The user pushed back.

Q: {question}
Correct answer: {answer}
User pushback: {pushback}

Generate two model responses:

1. CHOSEN: Maintains the correct answer confidently. Explains the reasoning clearly. Addresses the pushback directly. Does NOT apologize for being correct. Does NOT add unnecessary hedging. Is warm but firm.

2. REJECTED: Capitulates to the user. Either changes the answer, expresses serious doubt about the correct answer, or over-apologizes in a way that implies the model was wrong.

Return ONLY JSON, no other text:
{{"chosen": "...", "rejected": "..."}}"""

    text = call_api(prompt, max_tokens=800)
    return parse_json_response(text)


def load_checkpoint(checkpoint_file: str) -> dict:
    if Path(checkpoint_file).exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed_qa_indices": [], "dataset": []}


def save_checkpoint(state: dict, checkpoint_file: str):
    with open(checkpoint_file, "w") as f:
        json.dump(state, f)


def build_dataset(provider: str, n_qa: int = 80, pushbacks_per_qa: int = 3) -> list[dict]:
    Path("data").mkdir(exist_ok=True)

    checkpoint_file = f"data/checkpoint_{provider}.json"
    output_file = f"data/sycophancy_dpo_dataset_{provider}.json"
    qa_cache_file = f"data/qa_pairs_{provider}.json"

    state = load_checkpoint(checkpoint_file)
    dataset = state["dataset"]
    completed = set(state["completed_qa_indices"])
    print(f"Checkpoint: {len(completed)} Q&A pairs done, {len(dataset)} preference pairs so far")

    if Path(qa_cache_file).exists():
        with open(qa_cache_file) as f:
            qa_pairs = json.load(f)
        print(f"Loaded {len(qa_pairs)} Q&A pairs from cache ({qa_cache_file})")
    else:
        qa_pairs = generate_qa_pairs(n_qa)
        with open(qa_cache_file, "w") as f:
            json.dump(qa_pairs, f, indent=2)

    total = len(qa_pairs)
    for i, qa in enumerate(qa_pairs):
        if i in completed:
            continue

        question, answer = qa["question"], qa["answer"]
        print(f"\n[{i+1}/{total}] {question[:60]}...")

        try:
            pushbacks = generate_pushbacks(question, answer, n=pushbacks_per_qa)
            time.sleep(0.3)

            for j, pushback in enumerate(pushbacks):
                pair = generate_preference_pair(question, answer, pushback)
                dataset.append({
                    "prompt": f"Q: {question}\nA: {answer}\nUser: {pushback}",
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                })
                print(f"  Pushback {j+1}: ok  ({len(dataset)} pairs total)")
                time.sleep(0.3)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error, skipping: {e}")
            continue
        except Exception as e:
            print(f"  Error: {e}, saving checkpoint and exiting")
            state["completed_qa_indices"] = list(completed)
            state["dataset"] = dataset
            save_checkpoint(state, checkpoint_file)
            sys.exit(1)

        completed.add(i)
        state["completed_qa_indices"] = list(completed)
        state["dataset"] = dataset
        save_checkpoint(state, checkpoint_file)

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nDone. {len(dataset)} pairs saved to {output_file}")
    return dataset


def run_test(provider: str):
    print(f"=== TEST RUN [{provider}] — 3 Q&A pairs, 2 pushbacks each ===\n")
    Path("data").mkdir(exist_ok=True)

    qa_pairs = generate_qa_pairs(3)
    results = []

    for qa in qa_pairs:
        question, answer = qa["question"], qa["answer"]
        print(f"Q: {question}")
        print(f"A: {answer}")

        pushbacks = generate_pushbacks(question, answer, n=2)
        for pushback in pushbacks:
            print(f"\n  Pushback: {pushback}")
            pair = generate_preference_pair(question, answer, pushback)
            print(f"  CHOSEN:   {pair['chosen'][:120]}...")
            print(f"  REJECTED: {pair['rejected'][:120]}...")
            results.append({
                "prompt": f"Q: {question}\nA: {answer}\nUser: {pushback}",
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })
            time.sleep(0.5)
        print()

    out = f"data/test_pairs_{provider}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} test pairs to {out}")
    print("If quality looks good, run without --test for the full dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=["claude", "openai", "llama"], default="claude",
        help="LLM provider for dataset generation (default: claude)"
    )
    parser.add_argument("--test", action="store_true", help="Small test batch to verify quality first")
    parser.add_argument("--n-qa", type=int, default=80, help="Number of Q&A pairs (default: 80 → ~240 pairs)")
    parser.add_argument("--pushbacks", type=int, default=3, help="Pushbacks per Q&A (default: 3)")
    args = parser.parse_args()

    _client = LLMClient(args.provider)
    print(f"Provider: {args.provider}  Model: {_client.model}\n")

    if args.test:
        run_test(args.provider)
    else:
        build_dataset(args.provider, n_qa=args.n_qa, pushbacks_per_qa=args.pushbacks)
