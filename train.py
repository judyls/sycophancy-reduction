"""
Sycophancy Reduction DPO — Training
Fine-tunes Mistral-7B-Instruct-v0.2 with DPO + QLoRA to reduce sycophancy.

Usage (on Colab/RunPod A100):
    pip install trl peft transformers bitsandbytes accelerate datasets
    python train.py
    python train.py --output-dir my-run --beta 0.05
"""

import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_DATA_FILE = "data/sycophancy_dpo_dataset.json"


def load_dataset(path: str) -> Dataset:
    with open(path) as f:
        records = json.load(f)
    # DPOTrainer expects columns: prompt, chosen, rejected
    return Dataset.from_list(records)


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,          # alpha=2r is a stable default
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main(output_dir: str, beta: float, epochs: int, batch_size: int, data_file: str = DEFAULT_DATA_FILE):
    print(f"Loading dataset from {data_file}...")
    dataset = load_dataset(data_file)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    print(f"  Train: {len(train_data)}  Eval: {len(eval_data)}")

    print(f"Loading {BASE_MODEL} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for decoder-only DPO

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=beta,
        max_length=512,
        max_prompt_length=256,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,         # TRL creates a frozen copy automatically with QLoRA
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}/final/")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/dpo-run", help="Where to save checkpoints and final adapter")
    parser.add_argument("--dataset", default=DEFAULT_DATA_FILE, help="Path to preference pairs JSON")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size (default: 2)")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_file=args.dataset,
    )
