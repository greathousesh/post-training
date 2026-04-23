import json

from datasets import Dataset


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_tokenize_fn(tokenizer, max_length):
    def tokenize(example):
        messages = example["messages"]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        # Only train on assistant turns: mask everything else with -100.
        labels = [-100] * len(input_ids)
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prefix_before = tokenizer.apply_chat_template(
                messages[:i], tokenize=True, add_generation_prompt=True
            )
            prefix_after = tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=True, add_generation_prompt=False
            )
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
