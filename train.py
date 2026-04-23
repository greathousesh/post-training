import argparse

import yaml
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import build_dataset
from src.model import load_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    set_seed(train_cfg.get("seed", 42))

    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"], cfg["quantization"], cfg["lora"])

    train_ds, eval_ds = build_dataset(
        cfg["data"]["train_file"],
        cfg["data"].get("eval_file"),
        tokenizer,
        cfg["model"]["max_length"],
    )

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=train_cfg["optim"],
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 42),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 200) if eval_ds is not None else None,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])


if __name__ == "__main__":
    main()
