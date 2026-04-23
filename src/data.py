import json
from dataclasses import dataclass

import torch
from datasets import Dataset


@dataclass
class CausalLMCollator:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lbl = f["labels"]
            am = f.get("attention_mask", [1] * len(ids))
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lbl + [self.label_pad_token_id] * pad)
            attn.append(am + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _encode(tokenizer, messages, add_generation_prompt):
    # Depending on transformers version, apply_chat_template(tokenize=True) may
    # return a list[int] or a BatchEncoding dict. Normalize to list[int].
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def build_tokenize_fn(tokenizer, max_length):
    def tokenize(example):
        messages = example["messages"]
        input_ids = _encode(tokenizer, messages, add_generation_prompt=False)
        # Only train on assistant turns: mask everything else with -100.
        labels = [-100] * len(input_ids)
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prefix_before = _encode(tokenizer, messages[:i], add_generation_prompt=True)
            prefix_after = _encode(tokenizer, messages[: i + 1], add_generation_prompt=False)
            start = len(prefix_before)
            end = min(len(prefix_after), len(input_ids))
            for j in range(start, end):
                labels[j] = input_ids[j]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    return tokenize


def build_dataset(train_file, eval_file, tokenizer, max_length):
    tokenize = build_tokenize_fn(tokenizer, max_length)

    train_ds = Dataset.from_list(load_jsonl(train_file))
    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)

    eval_ds = None
    if eval_file:
        eval_ds = Dataset.from_list(load_jsonl(eval_file))
        eval_ds = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)

    return train_ds, eval_ds
