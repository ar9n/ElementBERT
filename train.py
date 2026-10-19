import os
import json
import logging
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "bert-base-uncased"
data_path = "../datasets/Elements"
output_dir = "./elementbert-mlm"
max_length = 512


# ---------------------------------------------------------------------------
# Step 1: Extract abstracts from all JSON metadata files in data_path
# ---------------------------------------------------------------------------
def extract_abstracts_from_json(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return []

    abstracts = []
    for paper in data.get("papers", []):
        abstract = paper.get("abstract")
        if abstract and abstract.strip():
            abstracts.append(abstract.strip())
    return abstracts


def load_abstracts(data_dir: str) -> list[str]:
    texts = []
    json_files = sorted(Path(data_dir).rglob("*.json"))
    logger.info("Found %d JSON files in %s (including subfolders)", len(json_files), data_dir)

    for json_file in json_files:
        abstracts = extract_abstracts_from_json(str(json_file))
        if abstracts:
            texts.extend(abstracts)
        else:
            logger.warning("No abstracts found in %s", json_file)

    logger.info("Extracted %d abstracts from %d files", len(texts), len(json_files))
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    logger.info("Loading abstracts from %s", data_path)
    raw_texts = load_abstracts(data_path)

    if not raw_texts:
        raise RuntimeError(f"No usable abstracts extracted from JSON files in {data_path}")

    logger.info("Creating dataset from %d abstracts", len(raw_texts))

    dataset = Dataset.from_dict({"text": raw_texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    NUM_EPOCHS = 50
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=NUM_EPOCHS,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
