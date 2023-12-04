import ai2_arc
import airoboros
import apps
import belebele
import boolq
import cinematika
import drop
import mathinstruct
import mmlu
import natural_instructions
import openbookqa
import piqa
import python_alpaca
import slimorca
import spider
import squad_v2
import synthia
import winogrande

# Other imports.
import json
import faiss
import requests
import numpy as np
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from transformers import AutoModel


def decontaminate(dataset):
    """Decontaminate the dataset."""
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
    )
    index = faiss.IndexFlatL2(512)

    # Index the benchmark datasets.
    instructions = []
    alpaca_eval = json.loads(
        requests.get(
            "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json"
        ).text
    )
    logger.info("Indexing Alpaca Eval test set...")
    for item in tqdm(alpaca_eval):
        index.add(np.array([model.encode([item["instruction"]], max_length=4096)[0]]))
        instructions.append(item["instruction"])

    # MT-Bench.
    mt_bench = [
        json.loads(line)
        for line in requests.get(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
        ).text.splitlines()
        if line.strip()
    ]
    logger.info("Indexing MT Bench test set...")
    for item in tqdm(mt_bench):
        index.add(np.array([model.encode([item["turns"][0]], max_length=4096)[0]]))
        instructions.append(item["turns"][0])

    # DROP.
    logger.info("Indexing DROP test set...")
    for item in tqdm(load_dataset("drop", split="validation")):
        index.add(
            np.array(
                [
                    model.encode(
                        [item["passage"] + "\n" + item["question"]], max_length=4096
                    )[0]
                ]
            )
        )
        instructions.append(item["passage"] + "\n" + item["question"])

    # Winogrande.
    logger.info("Indexing winogrande test set...")
    for item in tqdm(load_dataset("winogrande", "winogrande_xl", split="test")):
        index.add(np.array([model.encode([item["sentence"]], max_length=4096)[0]]))
        instructions.append(item["sentence"])

    # MMLU
    logger.info("Indexing MMLU test set...")
    for item in tqdm(load_dataset("cais/mmlu", "all", split="test")):
        index.add(np.array([model.encode([item["question"]], max_length=4096)[0]]))
        instructions.append(item["question"])

    # TruthfulQA
    logger.info("Indexing TruthfulQA test set...")
    for item in tqdm(load_dataset("truthful_qa", "generation", split="validation")):
        index.add(np.array([model.encode([item["question"]], max_length=4096)[0]]))
        instructions.append(item["question"])

    # GSM8K
    logger.info("Indexing GSM8K test set...")
    for item in tqdm(load_dataset("gsm8k", "main", split="test")):
        index.add(np.array([model.encode([item["question"]], max_length=4096)[0]]))
        instructions.append(item["question"])

    # Hellaswag
    logger.info("Indexing Hellaswag test set...")
    for item in tqdm(load_dataset("Rowan/hellaswag", split="test")):
        index.add(np.array([model.encode([item["ctx"]], max_length=4096)[0]]))
        instructions.append(item["ctx"])

    # ARC Challenge
    logger.info("Indexing ARC-Challenge test set...")
    for item in tqdm(load_dataset("ai2_arc", "ARC-Challenge", split="test")):
        index.add(np.array([model.encode([item["question"]], max_length=4096)[0]]))
        instructions.append(item["question"])

    # HumanEval
    logger.info("Indexing HumanEval test set...")
    for item in tqdm(load_dataset("openai_humaneval", split="test")):
        index.add(np.array([model.encode([item["prompt"]], max_length=4096)[0]]))
        instructions.append(item["prompt"])

    # Now, the tedious part.  Go through our entire > 1m dataset and remove items...
    logger.info(
        "Removing contaminated values -- this is going to take a long time, go find a snack or something..."
    )
    keep = []
    for item in tqdm(dataset):
        convs = item.get("conversations", item.get("chosen"))
        if not convs:
            # plain text
            keep.append(item)
            continue

        # Find the instruction.
        instruction = None
        for turn in convs:
            if turn.get("from") == "human":
                instruction = turn["value"]
                break
        if not instruction:
            continue
        embeddings = np.array([model.encode([instruction], max_length=4096)[0]])
        distances, indices = index.search(embeddings, k=1)
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        if not distances:
            keep.append(item)
        else:
            dataset_instruction = instructions[indices[0]]
            # Not sure what's actually best here, but I'm going with:
            # cos sim > 0.03 or > 10% diff in length = not contamination.
            if (
                distances[0] >= 0.03
                or abs(
                    (len(dataset_instruction) - len(instruction))
                    / (len(dataset_instruction) or 1)
                )
                > 0.1
            ):
                keep.append(item)
            else:
                logger.warning(f"Likely contamination: {item}")
    return Dataset.from_list(keep)


def load_datasets():
    """Load all of the datasets."""
    things = {key: val for key, val in globals().items()}
    from datasets import concatenate_datasets
    from types import ModuleType

    all_datasets = []
    known_uids = set([])
    for key, val in sorted(things.items(), key=lambda m: m[1].CONFIDENCE, reverse=True):
        if key.startswith("__"):
            continue
        if isinstance(val, ModuleType) and hasattr(val, "load_data"):
            dataset = val.load_data(known_uids)
            if "text" not in dataset.column_names:
                dataset = dataset.add_column("text", [None] * len(dataset))
            if "source" not in dataset.column_names:
                dataset = dataset.add_column("source", [key] * len(dataset))
            dataset = dataset.remove_columns(
                [
                    col
                    for col in dataset.column_names
                    if col
                    not in (
                        "id",
                        "source",
                        "text",
                        "conversations",
                        "chosen",
                        "rejected",
                    )
                ]
            )
            all_datasets.append(dataset)
    return concatenate_datasets(all_datasets)


if __name__ == "__main__":
    dataset = load_datasets()
    print(dataset)
    dataset.to_parquet("bagel-raw-v0.1.parquet")
    decontaminated = decontaminate(dataset)
    print(decontaminated)
    decontaminated.to_parquet("bagel-clean-v0.1.parquet")
