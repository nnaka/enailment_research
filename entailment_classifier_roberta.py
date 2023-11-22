#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import csv
from enum import Enum
import functools
import os
import sys
from typing import Dict, Generator, List, Optional, Tuple, Union

# Must be called before the import of transformers etc to properly set the .cache dir
def setup_env(path: str) -> None:
    """Modifying where the .cache directory is getting stored"""
    os.environ["HF_HOME"] = path
    os.environ["TORCH_HOME"] = path
    os.environ["TRANSFORMERS_CACHE"] = path
    print(
        f"Environment variables set TORCH_HOME = {os.environ['TORCH_HOME']}; HF_HOME={os.environ['HF_HOME']}; TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}"
    )


setup_env("/scratch/nn1331/entailment/.cache")

from datasets import Dataset, load_dataset
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    pipeline,
    Pipeline,
    TrainingArguments,
    Trainer,
)

NUM_SENTENCES_IN_PRIOR: int = 5


class EntailmentCategory(Enum):
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2


def main(is_full: bool, is_final: bool, output_path: str = None) -> None:
    """Main routine"""
    run_zero_shot_nli(output_path)


def run_zero_shot_nli(output: str = None) -> None:
    nli_model = torch.hub.load("pytorch/fairseq", "roberta.large.mnli")
    tokenizer = nli_model
    """
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-large-mnli",
        device_map="auto",
        offload_folder="offload_folder",
        torch_dtype="auto",
        offload_state_dict=True,
    )
    """

    nli_model.eval()  # disable dropout for evaluation
    assert torch.cuda.is_available()
    # Move model to GPU (currently errors out due to model being both on CPU / GPU)
    # nli_model.cuda()  # run on gpu

    # Test
    premise: str = "This model is a heavily optimized version of BERT."
    hypothesis: str = "This model is not very optimized."
    model_input: List[int] = tokenizer.encode(premise, hypothesis)  # .cuda()

    print(f"TEST 1:")
    print(nli_model.predict("mnli", model_input).argmax())  # 0: not entailment

    hypothesis = "Roberta is based on BERT."
    model_input = tokenizer.encode(premise, hypothesis)  # .cuda()

    print(f"TEST 2:")
    print(nli_model.predict("mnli", model_input).argmax())  # 1: entailment

    # Experiment
    # TODO (nnaka): follow up tests of Pile data (https://arxiv.org/pdf/2101.00027.pdf) subsets
    # Dataset source: https://huggingface.co/datasets/monology/pile-uncopyrighted
    # https://huggingface.co/datasets/EleutherAI/pile doesn't work due to
    # https://huggingface.co/datasets/EleutherAI/pile/discussions/15
    # dataset: Dataset = load_dataset("monology/pile-uncopyrighted", split="test[:50%]")

    # DATASET_NAME: str = "suolyer/pile_freelaw"
    # DATASET_NAME: str = "suolyer/pile_arxiv"
    # DATASET_NAME: str = "suolyer/pile_youtubesubtitles"

    # DATASET_NAME: str = "suolyer/pile_books3"
    # DATASET_NAME: str = "suolyer/pile_wikipedia"
    # DATASET_NAME: str = "yelp_review_full"
    # dataset: Dataset = load_dataset(DATASET_NAME, split="test")["text"]

    # DATASET_NAME: str = "multi_news"
    # dataset: Dataset = load_dataset(DATASET_NAME, split="test")["document"]

    DATASET_NAME: str = "reuters21578"
    dataset: Dataset = load_dataset(DATASET_NAME, "ModHayes", split="test")["text"]

    # DATASET_NAME: str = "yahoo_answers_topics"
    # dataset: Dataset = load_dataset(DATASET_NAME, split="test")["best_answer"]

    DEFUALT_OUTPUT_CSV_FILE_PATH: str = (
        f"/scratch/nn1331/entailment/data-roberta-{DATASET_NAME.split('/')[-1]}.csv"
    )
    output_path: str = DEFUALT_OUTPUT_CSV_FILE_PATH if output is None else output

    # import pdb; pdb.set_trace()
    classify_dataset_text(nli_model, tokenizer, dataset, output_path)


def get_premise_and_hypothesis(
    document: str, n: int
) -> Generator[Tuple[str, str], None, None]:
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
        return csv.writer(csv_file), csv_file
    else:
        return csv.writer(sys.stdout), csv_file


def get_n_sentences(s: str, n: int) -> Generator[str, None, None]:
    """Return groups of n sentences of s"""
    sentences: List[str] = sent_tokenize(s)
    premise: str = ""
    hypothesis: str = ""
    for i in range(0, len(sentences), n):
        yield " ".join(sentences[i : i + n])


def classify_dataset_text(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    file_path: str = None,
) -> None:
    """Classification (i.e. pair input)"""
    # Get entailment examples
    results: Dict[str, List[str]] = {
        EntailmentCategory.CONTRADICTION.name: [],
        EntailmentCategory.ENTAILMENT.name: [],
        EntailmentCategory.NEUTRAL.name: [],
    }
    id2label: Dict[int, str] = {i.value: i.name for i in EntailmentCategory}

    print(f"{len(dataset)} examples")

    # NLTK package for splitting sentences
    nltk.download("punkt")

    # Write results in csv
    csv_writer, csv_file = get_csv_writer(file_path)

    premise: str = ""
    hypothesis: str = ""

    for data in dataset:
        # Split into predicate + hypothesis and try every n-previous + sentence window in document
        # Make sure the tokenization is within the 512-token limit
        for i, (premise, hypothesis) in enumerate(
            get_premise_and_hypothesis(data, NUM_SENTENCES_IN_PRIOR)
        ):
            # Hypotheses should be somewhat substantial
            if len(hypothesis.split()) < 5:
                print(
                    f"Skipping iteration {i} for PREMISE: {premise}; HYPOTHESIS: {hypothesis}; since hypothesis is too short"
                )
                continue
            tokens: torch.Tensor = tokenizer.encode(premise, hypothesis)  # .cuda()

            print(
                f"PREMISE: {premise}; HYPOTHESIS: {hypothesis}; TOKENS: {tokens}; size: {tokens.size()}"
            )

            if tokens.size(dim=0) > 512:
                # raise ValueError("Input exceeds the 512-token limit.")
                print("Input exceeds the 512-token limit.")
            else:
                label: str = id2label[model.predict("mnli", tokens).argmax().item()]

                print(f"Writing result #{i} to csv at path {file_path}")
                csv_writer.writerow([label, premise, hypothesis])
                csv_file.flush()


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
