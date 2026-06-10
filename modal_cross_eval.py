"""
Modal wrapper: run sycophancy eval with all three judges in one GPU job.

Loads the model once, runs the sycophancy eval three times (claude / openai / llama judge),
saves one eval_report_judge-{judge}.json per run. Much cheaper than 3 separate Modal runs.

Setup (one-time secrets):
    modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
    modal secret create openai-secret    OPENAI_API_KEY=sk-...
    modal secret create groq-secret      GROQ_API_KEY=gsk_...

Run:
    # All 3 judges on the existing Claude-trained adapter:
    modal run modal_cross_eval.py

    # Different adapter (e.g. model trained on OpenAI-generated data):
    modal run modal_cross_eval.py --dataset-label openai --adapter-subdir dpo-run-openai/final

Download results:
    modal volume get sycophancy-reduction-outputs /cross_eval ./cross_eval_results
    python cross_model_eval.py cross_eval_results/claude cross_eval_results/openai cross_eval_results/llama
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers==4.44.2",
        "trl==0.9.6",
        "peft==0.12.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.34.2",
        "datasets>=2.18.0",
        "anthropic>=0.25.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "sentencepiece",
    )
)

volume = modal.Volume.from_name("sycophancy-reduction-outputs", create_if_missing=True)

app = modal.App("sycophancy-cross-eval")


@app.function(
    gpu="A10G",
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("openai-secret"),
    ],
    volumes={"/outputs": volume},
    timeout=60 * 60 * 4,
)
def run_cross_eval(
    dataset_label: str = "claude",
    adapter_subdir: str = "dpo-run/final",
    n_sycophancy: int = 50,
    judges: list = None,
):
    import subprocess
    import sys
    import os

    if judges is None:
        judges = ["claude", "openai"]

    subprocess.run(
        ["git", "clone", "https://github.com/judyls/sycophancy-reduction.git", "/app"],
        check=True,
    )
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from eval import main as run_eval

    adapter_path = f"/outputs/{adapter_subdir}"
    output_dir = f"/outputs/cross_eval/{dataset_label}"

    # Load model once and run all judges — eval.py reloads models each call,
    # so we call it as a subprocess to keep isolation simple
    for judge in judges:
        print(f"\n{'='*60}")
        print(f"  Running judge={judge} on dataset={dataset_label}")
        print(f"{'='*60}\n")
        run_eval(
            adapter_path=adapter_path,
            eval_type="sycophancy",
            n_sycophancy=n_sycophancy,
            n_truthfulqa=0,
            output_dir=output_dir,
            judge=judge,
            dataset_label=dataset_label,
        )

    volume.commit()
    print(f"\nAll judges done. Results in Modal volume at: /cross_eval/{dataset_label}/")
    print(f"Download: modal volume get sycophancy-reduction-outputs /cross_eval ./cross_eval_results")


@app.local_entrypoint()
def main(
    dataset_label: str = "claude",
    adapter_subdir: str = "dpo-run/final",
    n_sycophancy: int = 50,
):
    run_cross_eval.remote(
        dataset_label=dataset_label,
        adapter_subdir=adapter_subdir,
        n_sycophancy=n_sycophancy,
    )
