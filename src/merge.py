import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_base(adapter_path, override):
    if override:
        return override
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        return json.load(f)["base_model_name_or_path"]


def merge(base_model_name, adapter_path, output_path):
    print(f"Base model:    {base_model_name}")
    print(f"Adapter path:  {adapter_path}")
    print(f"Output path:   {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=None,
        help="base model name (default: read from adapter_config.json)",
    )
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="output path")
    args = parser.parse_args()
    base = resolve_base(args.adapter, args.base)
    merge(base, args.adapter, args.output)
