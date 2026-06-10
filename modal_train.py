"""
Sycophancy DPO — Modal Training Runner

Setup (one-time):
    modal secret create huggingface-secret HF_TOKEN=hf_...

Run:
    # Train on Claude-generated dataset (original):
    modal run modal_train.py

    # Train on OpenAI-generated dataset:
    modal run modal_train.py --dataset data/sycophancy_dpo_dataset_openai.json --output-subdir dpo-run-openai

Download adapter when done:
    modal volume get sycophancy-reduction-outputs /dpo-run ./adapter
    modal volume get sycophancy-reduction-outputs /dpo-run-openai ./adapter_openai
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
        "python-dotenv>=1.0.0",
        "rich",
        "sentencepiece",
    )
)

volume = modal.Volume.from_name("sycophancy-reduction-outputs", create_if_missing=True)

app = modal.App("sycophancy-reduction-training")


@app.function(
    gpu="A10G",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
    timeout=60 * 60 * 3,
)
def run_training(dataset: str = "data/sycophancy_dpo_dataset.json", output_subdir: str = "dpo-run"):
    import subprocess
    import sys
    import os

    subprocess.run(
        ["git", "clone", "https://github.com/judyls/sycophancy-reduction.git", "/app"],
        check=True,
    )
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from train import main
    main(
        output_dir=f"/outputs/{output_subdir}",
        beta=0.1,
        epochs=3,
        batch_size=2,
        data_file=dataset,
    )

    volume.commit()
    print(f"\nAdapter saved to Modal volume: sycophancy-reduction-outputs/{output_subdir}")
    print(f"Download: modal volume get sycophancy-reduction-outputs /{output_subdir} ./adapter_{output_subdir}")


@app.local_entrypoint()
def main(
    dataset: str = "data/sycophancy_dpo_dataset.json",
    output_subdir: str = "dpo-run",
):
    run_training.remote(dataset=dataset, output_subdir=output_subdir)
