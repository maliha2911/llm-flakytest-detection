#!/usr/bin/env python3
"""
Offline zero-shot evaluation script for Mistral 7B on an Excel dataset.

What it does
------------
- Loads a locally downloaded Mistral 7B Instruct model (no internet required)
- Reads an .xlsx dataset
- Filters to a selected programming language
- Creates a stratified train/test split
- Uses the test set only for evaluation (no fine-tuning)
- Predicts labels via prompt-based generation: "Flaky" or "Non-Flaky"
- Computes accuracy, balanced accuracy, precision, recall, f1, MCC
- Saves detailed predictions and summary metrics to CSV/JSON

Example
-------
python offline_mistral_eval.py \
  --data /home/miftahul/projects/scripts/FlakyTest_Detection_LLM/datasets/flaky_data.xlsx \
  --model-path /home/miftahul/projects/models/mistral-7b \
  --language Python \
  --output-dir ./results \
  --test-size 0.2 \
  --max-samples 100

Notes
-----
- The train split is created for parity with your notebook and reproducibility, but
  it is not used to update the model because this script performs zero-shot inference only.
- Recommended to run on a GPU node.
"""

import argparse
import json
import os
import random
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Mistral 7B zero-shot evaluator for flaky test prediction.")
    parser.add_argument("--data", required=True, help="Path to flaky_data.xlsx")
    parser.add_argument("--model-path", required=True, help="Local path to downloaded Mistral model folder")
    parser.add_argument("--language", default="Python", help="Language filter: Java | Python | Go | C++ | JS")
    parser.add_argument("--text-column", default="test case content", help="Column containing test code/text")
    parser.add_argument("--label-column", default="label", help="Column containing label")
    parser.add_argument("--language-column", default="Language", help="Column containing language name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=66, help="Random seed for reproducibility")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer truncation length for the prompt")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Generated tokens for label output")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature; 0.0 is greedy")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts per batch")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap after language filtering")
    parser.add_argument("--output-dir", default="./offline_eval_results", help="Directory for outputs")
    parser.add_argument("--seed", type=int, default=42, help="Python/Numpy/Torch seed")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    # Normalize labels to binary integer 0/1
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
        data = data.sample(n=min(args.max_samples, len(data)), random_state=args.random_state).reset_index(drop=True)

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


def parse_generated_label(text: str) -> int:
    cleaned = text.strip()
    lower = cleaned.lower()

    # Strongest matches first
    if "non-flaky" in lower or "non flaky" in lower:
        return 0

    # Standalone flaky
    if re.search(r"\bflaky\b", lower):
        return 1

    # Conservative fallback: if neither exact phrase found, default to Non-Flaky
    # to avoid falsely inflating flaky predictions from malformed outputs.
    return 0


def load_model_and_tokenizer(model_path: str):
    model_path = str(model_path)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
        device_map=None,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()
    return model, tokenizer


def predict_batch(model, tokenizer, prompts, args):
    messages = [[{"role": "user", "content": p}] for p in prompts]

    rendered = [
        tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msg in messages
    ]

    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if args.temperature and args.temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for row in output_ids:
        gen_only = row[prompt_len:]
        text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        results.append(text)

    return results


def run_inference(model, tokenizer, test_df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    total = len(test_df)

    for start in range(0, total, args.batch_size):
        batch = test_df.iloc[start : start + args.batch_size].copy()
        prompts = [make_prompt(t) for t in batch["text"].tolist()]
        raw_outputs = predict_batch(model, tokenizer, prompts, args)
        preds = [parse_generated_label(x) for x in raw_outputs]

        batch["raw_model_output"] = raw_outputs
        batch["pred"] = preds
        rows.append(batch)

        print(f"Processed {min(start + args.batch_size, total)}/{total}")

    return pd.concat(rows, ignore_index=True)


def compute_scores(y_true, y_pred):
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Non-Flaky", "Flaky"],
            zero_division=0,
            output_dict=True,
        ),
    }
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Offline Mistral 7B zero-shot flaky test evaluator")
    print("=" * 80)
    print(f"Data       : {args.data}")
    print(f"Model path : {args.model_path}")
    print(f"Language   : {args.language}")
    print(f"Output dir : {output_dir.resolve()}")
    print(f"CUDA avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"GPU count  : {torch.cuda.device_count()}")

    df_raw = pd.read_excel(args.data, engine="openpyxl")
    df = prepare_dataframe(df_raw, args)

    if df["label"].nunique() < 2:
        raise ValueError("Need at least 2 classes after filtering to compute metrics.")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Filtered rows : {len(df)}")
    print(f"Train size    : {len(train_df)}")
    print(f"Test size     : {len(test_df)}")
    print("Label counts (full set):")
    print(df["label"].value_counts().sort_index())

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    predictions_df = run_inference(model, tokenizer, test_df, args)

    y_true = predictions_df["label"].astype(int).tolist()
    y_pred = predictions_df["pred"].astype(int).tolist()

    scores = compute_scores(y_true, y_pred)

    print("\n" + "=" * 80)
    print("Evaluation scores")
    print("=" * 80)
    for key in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc"]:
        print(f"{key:20s}: {scores[key]:.4f}")

    pred_out = output_dir / "test_predictions.csv"
    metrics_out = output_dir / "metrics.json"
    split_out = output_dir / "data_split_summary.json"

    predictions_df.to_csv(pred_out, index=False)

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    split_summary = {
        "language": args.language,
        "filtered_rows": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "label_counts_full": df["label"].value_counts().sort_index().to_dict(),
        "label_counts_train": train_df["label"].value_counts().sort_index().to_dict(),
        "label_counts_test": test_df["label"].value_counts().sort_index().to_dict(),
    }

    with open(split_out, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    print("\nSaved:")
    print(f"- Predictions : {pred_out}")
    print(f"- Metrics     : {metrics_out}")
    print(f"- Split info  : {split_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nERROR:\n")
        traceback.print_exc()
        raise
