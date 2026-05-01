import argparse
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from train_mistral_flaky_lora import prepare_dataframe, make_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True, help="LoRA adapter dir")
    parser.add_argument("--base-model-path", required=True, help="Base Mistral path")
    parser.add_argument("--language", default="Python")
    parser.add_argument("--text-column", default="test case content")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--language-column", default="Language")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=66)
    return parser.parse_args()


def main():
    args = parse_args()

    df_raw = pd.read_excel(args.data, engine="openpyxl")
    class DummyArgs:
        language_column = args.language_column
        label_column = args.label_column
        text_column = args.text_column
        language = args.language
        max_samples = args.max_samples
        random_state = args.random_state

    df = prepare_dataframe(df_raw, DummyArgs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = model.from_pretrained(args.model_path)  # load LoRA adapter
    model.eval()

    y_true = []
    y_pred = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        code = row["text"]
        label = int(row["label"])
        prompt = make_prompt(code)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
            )

        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = gen.strip().lower()

        if "flaky" in gen and "non" not in gen:
            pred = 1
        elif "non-flaky" in gen or "non flaky" in gen or "nonflaky" in gen:
            pred = 0
        else:
            # fallback: treat unknown as majority class or 0
            pred = 0

        y_true.append(label)
        y_pred.append(pred)

    mcc = matthews_corrcoef(y_true, y_pred)
    print("MCC:", mcc)


if __name__ == "__main__":
    main()
