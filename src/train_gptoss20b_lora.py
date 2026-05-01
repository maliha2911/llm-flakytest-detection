import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model


# ---------- Utility functions ----------

def normalize_language(value: str) -> str:
    value = str(value).strip()
    if value.lower() == "go":
        return "Go"
    return value


def prepare_dataframe(df: pd.DataFrame, args) -> pd.DataFrame:
    required = [args.language_column, args.label_column, args.text_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[[args.language_column, args.label_column, args.text_column]].copy()
    data.columns = ["Language", "label", "text"]

    data["Language"] = data["Language"].astype(str).map(normalize_language)
    data = data[data["Language"] != "PHP"].copy()
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str)

    def map_label(x):
        s = str(x).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        if s in {"1", "flaky", "true", "yes"}:
            return 1
        if s in {"0", "nonflaky", "false", "no"}:
            return 0
        try:
            numeric = int(float(s))
            if numeric in (0, 1):
                return numeric
        except Exception:
            pass
        raise ValueError(f"Unsupported label value: {x!r}")

    data["label"] = data["label"].map(map_label)
    data = data[data["Language"] == normalize_language(args.language)].reset_index(drop=True)

    if args.max_samples is not None:
        data = data.sample(
            n=min(args.max_samples, len(data)),
            random_state=args.random_state,
        ).reset_index(drop=True)

    if data.empty:
        raise ValueError(f"No rows found for language={args.language!r}")

    return data


def make_prompt(code_text: str) -> str:
    return (
        "You are a software testing assistant.\n"
        "Task: classify the following test case as exactly one of these labels:\n"
        "- Flaky\n"
        "- Non-Flaky\n\n"
        "Definition:\n"
        "- Flaky: the test may pass or fail intermittently without relevant code changes.\n"
        "- Non-Flaky: the test is stable and deterministic.\n\n"
        "Rules:\n"
        "1. Respond with only one label.\n"
        "2. Output exactly either Flaky or Non-Flaky.\n"
        "3. Do not explain.\n\n"
        f"Test case:\n{code_text}\n\n"
        "Label:"
    )


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    def label_to_str(x):
        return "Flaky" if int(x) == 1 else "Non-Flaky"

    texts = []
    for code, label in zip(df["text"].tolist(), df["label"].tolist()):
        prompt = make_prompt(code)
        full = prompt + " " + label_to_str(label)
        texts.append(full)

    df = df.copy()
    df["full_text"] = texts
    return df


def balance_classes_for_mcc(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    if len(counts) != 2:
        return df

    min_count = counts.min()
    balanced_parts = []
    for label_value in counts.index:
        subset = df[df["label"] == label_value]
        if len(subset) > min_count:
            subset = subset.sample(min_count, random_state=random_state)
        balanced_parts.append(subset)

    balanced_df = pd.concat(balanced_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced_df


# ---------- Argument parsing ----------

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of GPT-OSS-20B for flaky test classification (MCC-optimized).")
    parser.add_argument("--data", required=True, help="Path to flaky_data.xlsx")
    parser.add_argument("--model-path", required=True, help="Local path to GPT-OSS-20B model folder")
    parser.add_argument("--language", default="Python", help="Language filter")
    parser.add_argument("--text-column", default="test case content")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--language-column", default="Language")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=66)
    parser.add_argument("--output-dir", default="./gptoss20b_flaky_lora")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Main ----------

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LoRA fine-tuning GPT-OSS-20B for flaky test classification (MCC-optimized)")
    print("=" * 80)
    print(f"Data       : {args.data}")
    print(f"Model path : {args.model_path}")
    print(f"Language   : {args.language}")
    print(f"Output dir : {output_dir.resolve()}")
    print(f"CUDA avail : {torch.cuda.is_available()}")

    # 1) Load and filter data
    df_raw = pd.read_excel(args.data, engine="openpyxl")
    df = prepare_dataframe(df_raw, args)

    print("Label counts before balancing:")
    print(df["label"].value_counts().sort_index())

    df = balance_classes_for_mcc(df, random_state=args.random_state)

    print("Label counts after balancing:")
    print(df["label"].value_counts().sort_index())

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.random_state,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Filtered rows (balanced) : {len(df)}")
    print(f"Train size               : {len(train_df)}")
    print(f"Val size                 : {len(val_df)}")

    # 2) Build training text
    train_df = build_text_column(train_df)
    val_df = build_text_column(val_df)

    train_ds = Dataset.from_pandas(train_df[["full_text"]])
    val_ds = Dataset.from_pandas(val_df[["full_text"]])

    # 3) Load tokenizer & model (GPT-OSS requires trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()

    # 4) LoRA config for GPT-NeoX architecture
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5) Tokenization
    def tokenize_fn(example):
        out = tokenizer(
            example["full_text"],
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["full_text"])
    val_ds_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["full_text"])

    # 6) TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tok,
        eval_dataset=val_ds_tok,
    )

    trainer.train()

    # 7) Save LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nSaved LoRA adapter + tokenizer to:", output_dir)


if __name__ == "__main__":
    main()
