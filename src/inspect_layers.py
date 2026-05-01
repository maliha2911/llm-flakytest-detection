import torch
from transformers import AutoModelForCausalLM, AutoConfig


def main():
    # TODO: change this if your model is in a different folder
    model_path = "/home/miftahul/models/gpt-oss-20b"

    print(f"Loading config from: {model_path}")
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"Model class: {config.__class__.__name__}")

    print(f"\nLoading model on CPU just for inspection...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},   # keep it simple, no GPU needed
    )

    print("\nListing Linear layers (name -> class):\n")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name)
            count += 1

    print(f"\nTotal Linear layers found: {count}")


if __name__ == "__main__":
    main()
