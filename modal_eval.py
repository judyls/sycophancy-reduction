"""
Sycophancy DPO — Modal Eval Runner

Setup (one-time):
    modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...

Run:
    modal run modal_eval.py
    modal run modal_eval.py --eval sycophancy
    modal run modal_eval.py --eval truthfulqa

Download results:
    modal volume get sycophancy-reduction-outputs /eval ./eval_results
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
        "python-dotenv>=1.0.0",
        "rich",
        "sentencepiece",
    )
)

volume = modal.Volume.from_name("sycophancy-reduction-outputs", create_if_missing=False)

app = modal.App("sycophancy-reduction-eval")


@app.function(
    gpu="A10G",
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    volumes={"/outputs": volume},
    timeout=60 * 60 * 3,
)
def run_eval(eval_type: str = "both", n_sycophancy: int = 50, n_truthfulqa: int = 100):
    import subprocess
    import sys
    import os

    subprocess.run(
        ["git", "clone", "https://github.com/judyls/sycophancy-reduction.git", "/app"],
        check=True,
    )
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from eval import main
    main(
        adapter_path="/outputs/dpo-run/final",
        eval_type=eval_type,
        n_sycophancy=n_sycophancy,
        n_truthfulqa=n_truthfulqa,
        output_dir="/outputs/eval",
    )

    volume.commit()
    print("\nResults saved to Modal volume: sycophancy-reduction-outputs/eval")
    print("Download with: modal volume get sycophancy-reduction-outputs /eval ./eval_results")


@app.local_entrypoint()
def main(eval: str = "both", n_sycophancy: int = 50, n_truthfulqa: int = 100):
    run_eval.remote(eval_type=eval, n_sycophancy=n_sycophancy, n_truthfulqa=n_truthfulqa)
