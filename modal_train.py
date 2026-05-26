"""
Sycophancy DPO — Modal Training Runner

Setup (one-time):
    modal secret create huggingface-secret HF_TOKEN=hf_...

Run:
    modal run modal_train.py

Download adapter when done:
    modal volume get sycophancy-reduction-outputs /dpo-run ./adapter
"""

import modal

# PyTorch image with CUDA — bitsandbytes needs this
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

# Persistent volume — adapter survives after the function ends
volume = modal.Volume.from_name("sycophancy-reduction-outputs", create_if_missing=True)

app = modal.App("sycophancy-reduction-training")


@app.function(
    gpu="A10G",                                          # 24GB VRAM, ~$1.10/hr — fits 7B in 4-bit
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
    timeout=60 * 60 * 3,                                # 3hr ceiling
)
def run_training():
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
    main(output_dir="/outputs/dpo-run", beta=0.1, epochs=3, batch_size=2)

    volume.commit()
    print("\nAdapter saved to Modal volume: sycophancy-reduction-outputs/dpo-run")
    print("Download with: modal volume get sycophancy-reduction-outputs /dpo-run ./adapter")


@app.local_entrypoint()
def main():
    run_training.remote()
