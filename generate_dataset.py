"""
Sycophancy Reduction DPO — Dataset Generation
Generates preference pairs where chosen = maintains correct answer, rejected = capitulates.

Usage:
    export ANTHROPIC_API_KEY=your_key
    python generate_dataset.py           # full run (~800 pairs)
    python generate_dataset.py --test    # test run (10 pairs, verify quality first)
"""

import anthropic
import json
import time
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env.local from the project directory
load_dotenv(Path(__file__).parent / ".env.local")

client = anthropic.Anthropic()

CHECKPOINT_FILE = "data/checkpoint.json"
OUTPUT_FILE = "data/sycophancy_dpo_dataset.json"

# Use Haiku — fast, cheap, fully capable for factual Q&A generation
MODEL = "claude-haiku-4-5-20251001"


def call_api(prompt: str, max_tokens: int = 1000, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed after {retries} attempts")


def parse_json_response(text: str) -> any:
    """Extract JSON from response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # drop first and last fence lines
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def generate_qa_pairs(n: int) -> list[dict]:
    """Generate n factual Q&A pairs with unambiguous correct answers."""
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
    """Generate n pushback messages challenging the correct answer."""
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
    """Generate chosen (maintains answer) and rejected (capitulates) responses."""
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


def load_checkpoint() -> dict:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_qa_indices": [], "dataset": []}


def save_checkpoint(state: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)


def build_dataset(n_qa: int = 80, pushbacks_per_qa: int = 3) -> list[dict]:
    """
    Build the full dataset with checkpointing.
    Default: 80 Q&A pairs × 3 pushbacks = 240 preference pairs.
    Run twice (or increase n_qa) to reach 800+.
    """
    Path("data").mkdir(exist_ok=True)
    state = load_checkpoint()
    dataset = state["dataset"]
    completed = set(state["completed_qa_indices"])

    print(f"Checkpoint: {len(completed)} Q&A pairs already done, {len(dataset)} pairs in dataset")

    # Generate Q&A pairs if we don't have them cached
    qa_cache_file = "data/qa_pairs.json"
    if Path(qa_cache_file).exists():
        with open(qa_cache_file) as f:
            qa_pairs = json.load(f)
        print(f"Loaded {len(qa_pairs)} Q&A pairs from cache")
    else:
        qa_pairs = generate_qa_pairs(n_qa)
        with open(qa_cache_file, "w") as f:
            json.dump(qa_pairs, f, indent=2)

    total = len(qa_pairs)
    for i, qa in enumerate(qa_pairs):
        if i in completed:
            continue

        question = qa["question"]
        answer = qa["answer"]
        print(f"\n[{i+1}/{total}] {question[:60]}...")

        try:
            pushbacks = generate_pushbacks(question, answer, n=pushbacks_per_qa)
            time.sleep(0.3)  # small pause between calls

            for j, pushback in enumerate(pushbacks):
                pair = generate_preference_pair(question, answer, pushback)
                dataset.append({
                    "prompt": f"Q: {question}\nA: {answer}\nUser: {pushback}",
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                })
                print(f"  Pushback {j+1}: ✓  ({len(dataset)} pairs total)")
                time.sleep(0.3)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error, skipping: {e}")
            continue
        except Exception as e:
            print(f"  Error: {e}, saving checkpoint and exiting")
            state["completed_qa_indices"] = list(completed)
            state["dataset"] = dataset
            save_checkpoint(state)
            sys.exit(1)

        completed.add(i)
        state["completed_qa_indices"] = list(completed)
        state["dataset"] = dataset
        save_checkpoint(state)

    return dataset


def run_test():
    """Generate 10 pairs and print them for quality review."""
    print("=== TEST RUN — 3 Q&A pairs, 2 pushbacks each = ~6 pairs ===\n")
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

    with open("data/test_pairs.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} test pairs to data/test_pairs.json")
    print("Review them — if quality looks good, run without --test for the full dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a small test batch first")
    parser.add_argument("--n-qa", type=int, default=80, help="Number of Q&A pairs (default: 80 → ~240 pairs)")
    parser.add_argument("--pushbacks", type=int, default=3, help="Pushbacks per Q&A (default: 3)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    if args.test:
        run_test()
    else:
        dataset = build_dataset(n_qa=args.n_qa, pushbacks_per_qa=args.pushbacks)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"\n✓ Done. {len(dataset)} preference pairs saved to {OUTPUT_FILE}")
        print("Next: review a sample, then set up training (see train.py)")
