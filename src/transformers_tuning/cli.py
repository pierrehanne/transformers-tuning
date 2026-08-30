"""Command-line fine-tuning entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from transformers_tuning.methods import METHODS, build_peft_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), default="lora")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", default="Abirate/english_quotes")
    parser.add_argument("--text-column", default="quote")
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/adapter"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset, split=f"train[:{args.max_samples}]")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[args.text_column], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    targets = [target.strip() for target in args.target_modules.split(",") if target.strip()]
    model = get_peft_model(model, build_peft_config(args.method, targets))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
