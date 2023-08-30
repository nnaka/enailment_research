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
from typing import Dict, Generator, List, Optional, Tuple

from datasets import Dataset, load_dataset
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
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
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2


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


def main(is_full: bool, is_final: bool, output_path: str = None) -> None:
    """Main routine"""
    # run_zero_shot(output_path)
    run_zero_shot_nli(output_path)


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


def run_zero_shot(output: str = None) -> None:
    print("Running research project")

    # Model source: https://huggingface.co/roberta-large-mnli
    # Pipeline-less Roberta model based on https://github.com/facebookresearch/fairseq/tree/main/examples/roberta#use-roberta-for-sentence-pair-classification-tasks
    # Download RoBERTa already finetuned for MNLI
    # force_reload=True based on https://github.com/facebookresearch/fairseq/issues/2678
    roberta = torch.hub.load(
        "pytorch/fairseq:main", "roberta.large.mnli", force_reload=True
    )

    roberta.eval()  # disable dropout for evaluation
    roberta.cuda()  # run on gpu

    # Test
    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode(
        "Roberta is a heavily optimized version of BERT.",
        "Roberta is not very optimized.",
    )
    print(f"TEST 1:")
    print(roberta.predict("mnli", tokens).argmax())  # 0: contradiction

    # Encode another pair of sentences
    tokens = roberta.encode(
        "Roberta is a heavily optimized version of BERT.", "Roberta is based on BERT."
    )
    print(f"TEST 2:")
    print(roberta.predict("mnli", tokens).argmax())  # 2: entailment

    # Experiment
    DEFUALT_OUTPUT_CSV_FILE_PATH: str = "/scratch/nn1331/entailment/data.csv"
    output_path: str = DEFUALT_OUTPUT_CSV_FILE_PATH if output is None else output

    # TODO: vanilla entailment classification on premise + hypothesis in data
    classify_open_web_text(roberta, output_path)


def run_zero_shot_nli(output: str = None) -> None:
    # "pytorch/fairseq:main", "roberta.large.mnli", force_reload=True
    nli_model = pipeline(
        "zero-shot-classification", model="google/t5_xxl_true_nli_mixture"
    )

    """
    nli_model = torch.hub.load(
        model="google/t5_xxl_true_nli_mixture", force_reload=True
    )
    """

    nli_model.eval()  # disable dropout for evaluation
    nli_model.cuda()  # run on gpu

    # Test
    premise: str = "This model is a heavily optimized version of BERT."
    hypothesis: str = "This model is not very optimized."
    model_input: str = f"premise: {premise} hypothesis: {hypothesis}"

    print(f"TEST 1:")
    print(nli_model.predict(model_input).argmax())  # 0: not entailment

    hypothesis = "Roberta is based on BERT."
    model_input = f"premise: {premise} hypothesis: {hypothesis}"

    print(f"TEST 2:")
    print(nli_model.predict(model_input).argmax())  # 1: entailment

    # Experiment
    DEFUALT_OUTPUT_CSV_FILE_PATH: str = "/scratch/nn1331/entailment/data.csv"
    output_path: str = DEFUALT_OUTPUT_CSV_FILE_PATH if output is None else output
    # entailment classification for summarized groups of n sentences of data
    # classify_summarized_text(roberta, output_path)


def get_premise_and_hypothesis(
    document: str, n: int
) -> Generator[List[str], None, None]:
    """premise is last n sentences before hypothesis"""
    sentences: List[str] = sent_tokenize(document)
    premise: str = ""
    hypothesis: str = ""
    for i, sentence in enumerate(sentences):
        if i >= n + 1:
            premise = "".join(sentences[i - n : i])
            hypothesis = sentences[i]
            yield premise, hypothesis


def get_csv_writer(file_path: str = None):
    """Either write to file path or to stdout"""
    if file_path is not None:
        csv_file = open(file_path, "w", newline="")
        return csv.writer(csv_file)
    else:
        return csv.writer(sys.stdout)


def classify_open_web_text(roberta: Pipeline, file_path: str = None) -> None:
    """Classification using roberta pipeline classifier (i.e. pair input)"""
    # Dataset source: https://huggingface.co/datasets/openwebtext
    dataset: Dataset = load_dataset("openwebtext")

    # Get entailment examples
    results: Dict[EntailmentCategory, List[str]] = {
        "CONTRADICTION": [],
        "ENTAILMENT": [],
        "NEUTRAL": [],
    }

    print(f"{len(dataset['train'])} training examples")

    # NLTK package for splitting sentences
    nltk.download("punkt")

    # Write results in csv
    csv_writer = get_csv_writer(file_path)

    id2label: Dict[int, str] = {i.value: i.name for i in EntailmentCategory}
    premise: str = ""
    hypothesis: str = ""

    for data in dataset["train"]:
        # Split into predicate + hypothesis and try every n-previous + sentence window in document
        # Doc: https://github.com/facebookresearch/fairseq/tree/main/examples/roberta#use-roberta-for-sentence-pair-classification-tasks
        # Make sure the tokenization is within the 512-token limit
        for (premise, hypothesis) in get_premise_and_hypothesis(data["text"], 5):
            tokens = roberta.encode(premise, hypothesis)
            print(f"{premise} {hypothesis}; TOKENS: {tokens}; size: {tokens.size()}")

            if tokens.size(dim=0) > 512:
                # raise ValueError("Input exceeds the 512-token limit.")
                print("Input exceeds the 512-token limit.")
            else:
                label: str = id2label[roberta.predict("mnli", tokens).argmax().item()]
                results[label].append(f'{data["text"]}')

                csv_writer.writerow([label, premise, hypothesis])


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


def summarize_text(s: str) -> str:
    classifier: Pipeline = pipeline("summarization")
    return classifier(s)[0]["summary_text"]


def get_n_sentences(s: str, n: int) -> Generator[List[str], None, None]:
    """Return groups of n sentences of s"""
    sentences: List[str] = sent_tokenize(s)
    premise: str = ""
    hypothesis: str = ""
    for i in range(0, len(sentences), n):
        yield " ".join(sentences[i : i + n])


def classify_summarized_text(roberta: Pipeline, file_path: str = None) -> None:
    """Classification using roberta pipeline classifier (i.e. pair input)"""
    # Dataset source: https://huggingface.co/datasets/openwebtext
    dataset: Dataset = load_dataset("openwebtext")

    # Get entailment examples
    results: Dict[EntailmentCategory, List[str]] = {
        "CONTRADICTION": [],
        "ENTAILMENT": [],
        "NEUTRAL": [],
    }

    print(f"{len(dataset['train'])} training examples")

    # NLTK package for splitting sentences
    nltk.download("punkt")

    # Write results in csv
    csv_writer = get_csv_writer(file_path)

    id2label: Dict[int, str] = {i.value: i.name for i in EntailmentCategory}
    premise: str = ""
    hypothesis: str = ""

    # Summarization model
    classifier: Pipeline = pipeline("summarization")

    for data in dataset["train"]:
        # Split into predicate + hypothesis and try every n-previous + sentence window in document
        # Doc: https://github.com/facebookresearch/fairseq/tree/main/examples/roberta#use-roberta-for-sentence-pair-classification-tasks
        # Make sure the tokenization is within the 512-token limit
        for premise in get_n_sentences(data["text"], 20):
            # Summarize premise
            try:
                print(f"HERE: {classifier(premise)}")
                print(f"HERE2: {classifier(premise)[0]['summary_text']}")

                hypothesis: str = classifier(premise)[0]["summary_text"]
            except IndexError as e:
                print(f"ERROR: {e}")
                continue
            tokens = roberta.encode(premise, hypothesis)
            print(
                f"PREMISE: {premise}; HYPOTHESIS: {hypothesis}; TOKENS: {tokens}; size: {tokens.size()}"
            )

            if tokens.size(dim=0) > 512:
                # raise ValueError("Input exceeds the 512-token limit.")
                print("Input exceeds the 512-token limit.")
            else:
                label: str = id2label[roberta.predict("mnli", tokens).argmax().item()]
                results[label].append(f'{data["text"]}')

                csv_writer.writerow([label, premise, hypothesis])


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
    parser.add_argument(
        "--out",
        dest="output",
        type=str,
        help="Output CSV file path",
    )

    args = parser.parse_args()
    print(f"Using args: {args}")

    # Create the spark session object
    # spark = SparkSession.builder.appName("final_project").getOrCreate()

    # Call our main routine
    # main(spark, args.is_full, args.is_final)
    main(args.is_full, args.is_final, args.output)
