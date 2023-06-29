#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    $ spark-submit --deploy-mode client _.py
"""
from argparse import ArgumentParser
import csv
from enum import Enum
import functools
import os
import sys
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    pipeline,
    Pipeline,
    TrainingArguments,
    Trainer,
)


class EntailmentCategory(Enum):
    ENTAILMENT = 0
    NEUTRAL = 1
    CONTRADICTION = 2


def create_preprocess_function(tokenizer):
    # TODO: review what's the best way to include premise and the hypothesis
    def preprocess_function(examples):
        print(examples)
        entries_to_remove: List[str] = [
            "promptID",
            "pairID",
            "premise_binary_parse",
            "premise_parse",
            "hypothesis_binary_parse",
            "hypothesis_parse",
            "genre",
        ]
        for entry in entries_to_remove:
            examples.pop(entry, None)

        print(examples["premise"])
        # Truncate context
        examples["premise"] = [
            tokenizer(example, truncation=True) for example in examples["premise"]
        ]
        print(examples["premise"])
        examples["hypothesis"] = [
            tokenizer(example, truncation=True) for example in examples["hypothesis"]
        ]
        return examples

    return preprocess_function


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def main(is_full: bool, is_final: bool) -> None:
    """Main routine"""
    run_zero_shot()


def train_model() -> None:
    print("Running entailment training")

    # Preprocess helpers
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("roberta-base")
    data_collator: DataCollatorWithPadding = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # Dataset recommended by Will Merrill
    dataset: Dataset = load_dataset("multi_nli")

    print(dataset)
    print(dataset.keys())

    # Preprocess data
    preprocess_function = create_preprocess_function(tokenizer)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    print(tokenized_dataset)

    accuracy = evaluate.load("accuracy")

    id2label: Dict[int, str] = {i.value: i.name for i in EntailmentCategory}
    label2id: Dict[str, int] = {i.name: i.value for i in EntailmentCategory}

    model: AutoModelForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli",
            num_labels=len(id2label.keys()),
            id2label=id2label,
            label2id=label2id,
        )
    )

    training_args: TrainingArguments = TrainingArguments(
        output_dir="entailment_classifier_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
    )

    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation_matched"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()


def run_zero_shot() -> None:
    print("Running research project")

    # Model source: https://huggingface.co/roberta-large-mnli
    # Truncate sentences (context) to under 512 characters for roberta limit
    classifier: Pipeline = pipeline(
        "text-classification",
        model="roberta-large-mnli",
        tokenizer="roberta-large-mnli",
        max_length=512,
        truncation=True,
    )

    # Test zero-shot NLI classification
    print(
        classifier(
            "A soccer game with multiple males playing. Some men are playing a sport."
        )
    )
    # [{'label': 'ENTAILMENT', 'score': 0.98}]

    open_web_text(classifier)


def open_web_text(classifier: Pipeline) -> None:
    # Dataset source: https://huggingface.co/datasets/openwebtext
    dataset: Dataset = load_dataset("openwebtext")

    # Get entailment examples
    results: Dict[EntailmentCategory, List[str]] = {
        "CONTRADICTION": [],
        "ENTAILMENT": [],
        "NEUTRAL": [],
    }
    label: str = ""
    score: float = 0.0

    print(f"{len(dataset['train'])} training examples")
    # print(f'{dataset["train"][0]}')

    # Write results in csv
    csv_writer = csv.writer(sys.stdout)

    for data in dataset["train"]:
        res: Dict[str, Union[str, float]] = classifier(data["text"])[0]
        label: str = res["label"]
        score: float = res["score"]
        # print(f"label: {label}; score: {score}")
        if float(score) > 0.5:
            results[label].append(f'{data["text"]}')

            csv_writer.writerow([label, score, data["text"]])


def mnli_test(classifier: Pipeline) -> None:
    # Dataset recommended by Will Merrill
    dataset: Dataset = load_dataset("multi_nli")

    # Get entailment examples
    results: Dict[EntailmentCategory, List[str]] = {
        "CONTRADICTION": [],
        "ENTAILMENT": [],
        "NEUTRAL": [],
    }
    label: str = ""
    score: float = 0.0

    print(f"{len(dataset['train'])} training examples")
    # print(f'{dataset["train"][0]}')
    # Write results in csv
    csv_writer = csv.writer(sys.stdout)

    for data in dataset["train"]:
        # print(f'{data["premise"] + data["hypothesis"]} {classifier(data["premise"] + data["hypothesis"])}')
        res: Dict[str, Union[str, float]] = classifier(
            data["premise"] + data["hypothesis"]
        )[0]
        label: str = res["label"]
        score: float = res["score"]
        # print(f"label: {label}; score: {score}")
        if float(score) > 0.5:
            results[label].append(f'{data["premise"]}, {data["hypothesis"]}')

            csv_writer.writerow([label, data["premise"], data["hypothesis"]])

            if label == "ENTAILMENT":
                pass
                # print(f"Entailment: {data}")


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--full", dest="is_full", action="store_true", help="Run on full dataset"
    )
    parser.add_argument(
        "--final",
        dest="is_final",
        action="store_true",
        help="Run on final datasets, train/test",
    )

    args = parser.parse_args()
    print(f"Using args: {args}")

    # Create the spark session object
    # spark = SparkSession.builder.appName("final_project").getOrCreate()

    # Call our main routine
    # main(spark, args.is_full, args.is_final)
    main(args.is_full, args.is_final)
