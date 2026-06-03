"""
Cross-model comparison: does same-model-generates-and-judges inflate sycophancy scores?

Reads eval_report_judge-*.json files from one or more result directories and prints
a comparison table. Each directory corresponds to a model trained on a different dataset.

Full experiment matrix
  Rows    = training dataset source  (claude | openai | llama)
  Columns = sycophancy judge          (claude | openai | llama)
  Cells   = fine-tuned sycophancy rate

If the diagonal (same model generates data AND judges) is systematically lower than
off-diagonal entries, that's evidence of circularity bias inflating the results.

Workflow
  1. Generate 3 datasets:
       python generate_dataset.py --provider claude
       python generate_dataset.py --provider openai
       python generate_dataset.py --provider llama

  2. Train 3 models on Modal (one per dataset), save adapters:
       modal run modal_train.py --dataset data/sycophancy_dpo_dataset_claude.json
       modal run modal_train.py --dataset data/sycophancy_dpo_dataset_openai.json
       modal run modal_train.py --dataset data/sycophancy_dpo_dataset_llama.json

  3. Run all judge combos (use modal_cross_eval.py for GPU, or eval.py locally):
       python eval.py --adapter adapter/final_claude --judge claude --dataset-label claude --output-dir outputs/cross/claude
       python eval.py --adapter adapter/final_claude --judge openai --dataset-label claude --output-dir outputs/cross/claude
       python eval.py --adapter adapter/final_claude --judge llama  --dataset-label claude --output-dir outputs/cross/claude
       # ... repeat for openai and llama adapters

  4. Print comparison table:
       python cross_model_eval.py outputs/cross/claude outputs/cross/openai outputs/cross/llama

  Quick start (just re-judge the existing Claude-trained adapter with all 3 judges):
       python eval.py --adapter adapter/final --judge claude --dataset-label claude --output-dir outputs/cross/claude
       python eval.py --adapter adapter/final --judge openai --dataset-label claude --output-dir outputs/cross/claude
       python eval.py --adapter adapter/final --judge llama  --dataset-label claude --output-dir outputs/cross/claude
       python cross_model_eval.py outputs/cross/claude
"""

import json
import sys
from pathlib import Path

JUDGES = ["claude", "openai", "llama"]


def load_results(dirs: list[str]) -> dict:
    """
    Returns: {dataset_label: {judge: {"base_rate": float, "ft_rate": float, "reduction": float}}}
    """
    results = {}
    for d in dirs:
        for report_path in Path(d).glob("eval_report_judge-*.json"):
            with open(report_path) as f:
                report = json.load(f)

            meta = report.get("metadata", {})
            dataset_label = meta.get("dataset_label", Path(d).name)
            judge = meta.get("judge", report_path.stem.split("judge-")[-1])

            if "sycophancy" not in report:
                continue

            s = report["sycophancy"]
            if dataset_label not in results:
                results[dataset_label] = {}
            results[dataset_label][judge] = {
                "base_rate":   s["base"]["sycophancy_rate"],
                "ft_rate":     s["finetuned"]["sycophancy_rate"],
                "reduction":   s["reduction"],
                "capitulations_base": s["base"]["capitulations"],
                "capitulations_ft":   s["finetuned"]["capitulations"],
                "n":           s["base"]["n"],
            }
    return results


def print_table(results: dict):
    datasets = sorted(results.keys())
    judges_found = sorted({j for v in results.values() for j in v})

    col_w = 14
    label_w = 10

    def fmt(val):
        return f"{val:.0%}"

    # ── Fine-tuned sycophancy rate table ─────────────────────────────────────
    print("\n══ Fine-tuned Sycophancy Rate  (lower = better) ══")
    print(f"{'Dataset':>{label_w}}  " + "  ".join(f"judge={j:<{col_w-7}}" for j in judges_found))
    print("-" * (label_w + 2 + len(judges_found) * (col_w + 2)))
    for ds in datasets:
        row = f"{ds:>{label_w}}  "
        for j in judges_found:
            if j in results[ds]:
                cell = fmt(results[ds][j]["ft_rate"])
                marker = " *" if ds == j else "  "   # * = same model on diagonal
            else:
                cell = "  n/a"
                marker = "  "
            row += f"{cell+marker:<{col_w}}  "
        print(row)
    print("\n  * diagonal: same model generated dataset AND judged results")

    # ── Reduction table ───────────────────────────────────────────────────────
    print("\n══ Sycophancy Reduction  (base rate − fine-tuned rate) ══")
    print(f"{'Dataset':>{label_w}}  " + "  ".join(f"judge={j:<{col_w-7}}" for j in judges_found))
    print("-" * (label_w + 2 + len(judges_found) * (col_w + 2)))
    for ds in datasets:
        row = f"{ds:>{label_w}}  "
        for j in judges_found:
            if j in results[ds]:
                cell = fmt(results[ds][j]["reduction"])
                marker = " *" if ds == j else "  "
            else:
                cell = "  n/a"
                marker = "  "
            row += f"{cell+marker:<{col_w}}  "
        print(row)
    print("\n  * diagonal: same model generated dataset AND judged results")

    # ── Base rate (should be ~same across judges for the same adapter) ────────
    print("\n══ Base Model Sycophancy Rate  (should be consistent across judges) ══")
    print(f"{'Dataset':>{label_w}}  " + "  ".join(f"judge={j:<{col_w-7}}" for j in judges_found))
    print("-" * (label_w + 2 + len(judges_found) * (col_w + 2)))
    for ds in datasets:
        row = f"{ds:>{label_w}}  "
        for j in judges_found:
            if j in results[ds]:
                cell = fmt(results[ds][j]["base_rate"])
            else:
                cell = "  n/a"
            row += f"{cell:<{col_w}}  "
        print(row)
    print()

    # ── Circularity check ─────────────────────────────────────────────────────
    print("══ Circularity check ══")
    for ds in datasets:
        if ds not in results or ds not in results[ds]:
            continue
        diagonal = results[ds][ds]["ft_rate"]
        off_diag = [results[ds][j]["ft_rate"] for j in judges_found if j != ds and j in results[ds]]
        if not off_diag:
            continue
        avg_off = sum(off_diag) / len(off_diag)
        diff = diagonal - avg_off
        note = "lower on diagonal (possible circularity)" if diff < -0.03 else \
               "higher on diagonal (cross-model harder?)" if diff > 0.03 else \
               "roughly consistent (no strong circularity signal)"
        print(f"  dataset={ds}: diagonal={diagonal:.0%}, off-diagonal avg={avg_off:.0%}  → {note}")
    print()


def main(dirs: list[str]):
    if not dirs:
        print("Usage: python cross_model_eval.py <results_dir> [<results_dir> ...]")
        print("       Each directory should contain eval_report_judge-*.json files.")
        sys.exit(1)

    results = load_results(dirs)
    if not results:
        print(f"No eval_report_judge-*.json files found in: {dirs}")
        sys.exit(1)

    print(f"Loaded results for datasets: {sorted(results.keys())}")
    print_table(results)


if __name__ == "__main__":
    main(sys.argv[1:])
