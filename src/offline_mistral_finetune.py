#!/usr/bin/env python3
"""
Offline fine-tuning script for flaky-test classification with a local Mistral model.

What it does
------------
1. Reads an XLSX dataset.
2. Filters to one programming language.
3. Splits into stratified train/test sets.
4. Fine-tunes a sequence-classification head + LoRA adapters.
5. Saves the tuned adapter/tokenizer and evaluation outputs.
6. Reports accuracy, balanced accuracy, precision, recall, F1, MCC.

Notes
-----
- This script is fully offline: it loads the model/tokenizer from a local directory only.
- It uses PEFT/LoRA to keep fine-tuning practical on shared GPU systems.
- It does NOT download from Hugging Face Hub.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Mistral fine-tuning for flaky-test classification"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to XLSX dataset")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local model directory (already downloaded)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save model, metrics, and predictions",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Python",
        choices=["Java", "Python", "Go", "C++", "JS"],
        help="Language subset to train/evaluate on",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=66)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target modules",
    )
    parser.add_argument("--bf16", action="store_true", help="Use bf16 if supported")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument(
        "--save-merged-model",
        action="store_true",
        help="Try to merge LoRA weights into the base model and save them too",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_label(x) -> int:
    s = str(x).strip().lower()
    if s in {"1", "flaky", "true", "yes"}:
        return 1
    if s in {"0", "non-flaky", "non flaky", "nonflaky", "false", "no"}:
        return 0
    try:
        numeric = int(float(s))
        if numeric in (0, 1):
            return numeric
    except Exception:
        pass
    raise ValueError(f"Unsupported label value: {x!r}")


def load_dataframe(data_path: str, language: str) -> pd.DataFrame:
    df = pd.read_excel(data_path, index_col=False)
    required = ["Language", "label", "test case content"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df.rename(columns={"test case content": "text"}, inplace=True)
    df.drop(df[df["Language"] == "PHP"].index, inplace=True)
    df.dropna(subset=["Language", "label", "text"], inplace=True)
    df["label"] = df["label"].apply(map_label)
    df["Language"] = df["Language"].replace({"go": "Go"})
    df = df[df["Language"] == language].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No rows found for language={language!r}")
    if df["label"].nunique() < 2:
        raise ValueError("Need at least two label classes after filtering.")

    return df


def split_dataset(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    def preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )
    tokenized = dataset.map(preprocess, batched=True)
    return tokenized


def compute_metrics_from_arrays(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }


def trainer_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics_from_arrays(labels, preds)


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("=" * 80)
    print("Offline Mistral fine-tuning for flaky-test classification")
    print("=" * 80)
    print(f"Data       : {args.data}")
    print(f"Model path : {args.model_path}")
    print(f"Language   : {args.language}")
    print(f"Output dir : {args.output_dir}")
    print(f"CUDA avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count  : {torch.cuda.device_count()}")
        print(f"GPU name   : {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset does not exist: {args.data}")

    print("\n[1/7] Loading dataset...")
    df = load_dataframe(args.data, args.language)
    train_df, test_df = split_dataset(df, args.test_size, args.random_state)
    print(f"Total rows : {len(df)}")
    print(f"Train rows : {len(train_df)}")
    print(f"Test rows  : {len(test_df)}")
    print("Train label distribution:")
    print(train_df["label"].value_counts(dropna=False).sort_index())
    print("Test label distribution:")
    print(test_df["label"].value_counts(dropna=False).sort_index())

    print("\n[2/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[3/7] Tokenizing dataset...")
    tokenized = build_datasets(train_df, test_df, tokenizer, args.max_length)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in {"input_ids", "attention_mask", "label"}]
    )

    print("[4/7] Loading local base model...")
    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[5/7] Preparing trainer...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # evaluation_strategy became eval_strategy in newer releases; handle both.
    training_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 1),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(args.eval_strategy != "no" and args.save_strategy != "no"),
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    if args.bf16 or (dtype == torch.bfloat16):
        training_kwargs["bf16"] = True
    elif dtype == torch.float16:
        training_kwargs["fp16"] = True

    try:
        training_args = TrainingArguments(
            eval_strategy=args.eval_strategy,
            save_strategy=args.save_strategy,
            **training_kwargs,
        )
    except TypeError:
        training_args = TrainingArguments(
            evaluation_strategy=args.eval_strategy,
            save_strategy=args.save_strategy,
            **training_kwargs,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=trainer_compute_metrics,
    )

    print("[6/7] Fine-tuning...")
    train_result = trainer.train()

    print("[7/7] Evaluating on test split...")
    eval_metrics = trainer.evaluate(tokenized["test"])

    raw_preds = trainer.predict(tokenized["test"])
    pred_logits = raw_preds.predictions
    pred_labels = np.argmax(pred_logits, axis=-1)

    metrics = compute_metrics_from_arrays(test_df["label"].to_numpy(), pred_labels)
    metrics["eval_loss"] = float(raw_preds.metrics.get("test_loss", np.nan))

    class_report = classification_report(
        test_df["label"].to_numpy(),
        pred_labels,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(test_df["label"].to_numpy(), pred_labels).tolist()

    print("\nFinal test metrics:")
    for k, v in metrics.items():
        print(f"{k:>18}: {v:.6f}" if isinstance(v, float) and not np.isnan(v) else f"{k:>18}: {v}")

    print("\nSaving artifacts...")
    # Save LoRA adapter + tokenizer
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Optional merged model save
    if args.save_merged_model:
        try:
            merged_dir = output_dir / "merged_model"
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            print(f"Merged model saved to: {merged_dir}")
        except Exception as e:
            print(f"Could not save merged model: {e}")

    predictions_df = test_df.copy()
    predictions_df["pred_label"] = pred_labels
    predictions_df["pred_text"] = predictions_df["pred_label"].map({0: "non-flaky", 1: "flaky"})
    predictions_df["true_text"] = predictions_df["label"].map({0: "non-flaky", 1: "flaky"})
    predictions_df["logit_0"] = pred_logits[:, 0]
    predictions_df["logit_1"] = pred_logits[:, 1]
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "train_runtime_metrics": train_result.metrics,
                "trainer_eval_metrics": eval_metrics,
                "final_test_metrics": metrics,
                "classification_report": class_report,
                "confusion_matrix": conf_mat,
            },
            f,
            indent=2,
        )

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": args.language,
                "n_total": int(len(df)),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "label_counts_total": df["label"].value_counts().sort_index().to_dict(),
                "label_counts_train": train_df["label"].value_counts().sort_index().to_dict(),
                "label_counts_test": test_df["label"].value_counts().sort_index().to_dict(),
            },
            f,
            indent=2,
        )

    print(f"\nDone. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
