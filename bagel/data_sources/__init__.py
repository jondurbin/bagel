from . import ai2_arc
from . import airoboros
from . import apps
from . import belebele
from . import boolq
from . import cinematika
from . import drop
from . import gutenberg
from . import helpsteer
from . import lmsys_chat_1m
from . import mathinstruct
from . import mmlu
from . import natural_instructions
from . import openbookqa
from . import orca_dpo_pairs
from . import piqa
from . import python_alpaca
from . import rosetta_code
from . import slimorca
from . import spider
from . import squad_v2
from . import synthia
from . import ultrafeedback
from . import winogrande

# Other imports.
import json
import faiss
import requests
import numpy as np
from tqdm import tqdm
from loguru import logger
from types import ModuleType
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModel


def decontaminate(dataset):
    """Decontaminate the dataset."""
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True, device_map="auto"
    )
    index = faiss.IndexFlatL2(512)
    lengths = []

    # Index the benchmark datasets.
    logger.info("Indexing Alpaca Eval test set...")
    alpaca_eval = json.loads(
        requests.get(
            "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json"
        ).text
    )
    for item in tqdm(alpaca_eval):
        text = item["instruction"]
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # MT-Bench.
    logger.info("Indexing MT Bench test set...")
    mt_bench = [
        json.loads(line)
        for line in requests.get(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
        ).text.splitlines()
        if line.strip()
    ]
    for item in tqdm(mt_bench):
        text = item["turns"][0]
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # DROP.
    logger.info("Indexing DROP test set...")
    for item in tqdm(load_dataset("drop", split="validation")):
        text = "\n".join([item["passage"], item["question"]])
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # Winogrande.
    logger.info("Indexing winogrande test set...")
    for item in tqdm(load_dataset("winogrande", "winogrande_xl", split="validation")):
        text = item["sentence"]
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # MMLU
    logger.info("Indexing MMLU test set...")
    for item in tqdm(load_dataset("cais/mmlu", "all", split="test")):
        text = item["question"]
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # TruthfulQA
    logger.info("Indexing TruthfulQA test set...")
    for item in tqdm(load_dataset("truthful_qa", "generation", split="validation")):
        text = item["question"]
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # GSM8K
    logger.info("Indexing GSM8K test set...")
    for item in tqdm(load_dataset("gsm8k", "main", split="test")):
        text = item["question"]
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # ARC Challenge
    logger.info("Indexing ARC-Challenge test set...")
    for item in tqdm(load_dataset("ai2_arc", "ARC-Challenge", split="validation")):
        # We'll add two versions of the data here, one multiple-choice, one plain.
        instruction = "\n".join(
            [
                item["question"],
                "\n".join(
                    [
                        f"{item['choices']['label'][idx]}. {item['choices']['text'][idx]}"
                        for idx in range(len(item["choices"]["label"]))
                    ]
                ),
            ]
        )
        index.add(np.array([model.encode([instruction], max_length=4096)[0]]))
        lengths.append(len(instruction))

        # Plain.
        text = "\n".join(
            [
                item["question"],
                item["choices"]["text"][
                    item["choices"]["label"].index(item["answerKey"])
                ],
            ]
        )
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # HumanEval - 2nd pass -- the bulk of our coding instructions are from
    # the python_alpaca module, which has it's own filtering.
    logger.info("Indexing HumanEval test set...")
    for item in tqdm(load_dataset("openai_humaneval", split="test")):
        text = item["prompt"]
        index.add(np.array([model.encode([text], max_length=4096)[0]]))
        lengths.append(len(text))

    # Now, the tedious part.  Go through our entire > 1m dataset and remove items...
    logger.info(
        "Removing contaminated values -- this is going to take a long time, go find a snack or something..."
    )
    index = faiss.index_cpu_to_all_gpus(index)

    # Filter for contamination, in batches -- if we don't use batches, the
    # performance of faiss is quite slow, particularly on CPU.
    remove = set([])
    batch = []
    dataset_size = len(dataset)
    for dataset_idx in tqdm(range(dataset_size)):
        item = dataset[dataset_idx]
        if item.get("text"):
            continue
        prompt = None
        if item.get("chosen"):
            prompt = item["prompt"]
        elif item.get("conversations"):
            for turn in item["conversations"]:
                if turn["from"] == "human":
                    prompt = turn["value"]
                    break
        if not prompt:
            continue

        # Queue item in batch.
        batch.append(
            {
                "text": prompt,
                "id": item["id"],
            }
        )
        if len(batch) == 1024 or dataset_idx == dataset_size - 1:
            embeddings = np.array(
                [
                    model.encode([batch_item["text"]], max_length=4096)[0]
                    for batch_item in batch
                ]
            )
            distances, indices = index.search(embeddings, k=1)
            for idx in range(len(batch)):
                if not len(distances[idx]):
                    continue
                distance = distances[idx][0]
                found_index = indices[idx][0]
                # Not sure what's actually best here, but I'm going with:
                # cos sim > 0.05 or > 20% diff in length = not contamination.
                length_delta = abs(
                    (len(batch[idx]["text"]) - lengths[found_index])
                    / (len(batch[idx]["text"]) or 1)
                )
                if distance <= 0.05 and length_delta <= 0.20:
                    logger.warning(f"Likely contamination: {batch[idx]['id']}")
                    remove.add(batch[idx]["id"])
            batch = []
    filtered = dataset.filter(lambda item: item["id"] not in remove)
    logger.success(
        f"Original size: {dataset_size}, decontaminated size: {len(filtered)}"
    )
    return filtered


def load_datasets():
    """Load all of the datasets."""
    things = {
        key: val
        for key, val in globals().items()
        if isinstance(val, ModuleType)
        and hasattr(val, "load_data")
        and not key.startswith("__")
    }
    all_datasets = []
    known_uids = set([])
    for key, val in sorted(things.items(), key=lambda m: m[1].CONFIDENCE, reverse=True):
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
                    "prompt",
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
