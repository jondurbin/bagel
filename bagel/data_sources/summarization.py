import json
import requests
import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from .util import as_conversation

PRIORITY = 3


def load_data(known_uids=set([])):
    """Summarization dataset."""
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/mistral-7b-v0.1", use_fast=True
    )
    combined_items = []
    for file in [
        "expert_summarization_4096-booksum-train.jsonl",
        "expert_summarization_4096-cnndm-summaries.jsonl",
        "expert_summarization_4096-mediasum-summaries.jsonl",
        "expert_summarization_4096-samsum-summaries.jsonl",
        "expert_summarization_4096-xsum-summaries.jsonl",
    ]:
        source_name = file.replace("expert_summarization_4096-", "").replace(
            ".jsonl", ""
        )
        logger.info(f"Loading airoboros-summarization/{source_name}...")
        dataset = (
            load_dataset(
                "mattpscott/airoboros-summarization", data_files=[file], split="train"
            )
            .shuffle(seed=42)
            .train_test_split(train_size=2000)["train"]
        )
        for item in tqdm(dataset):
            instruction = item["instruction"]
            re_match = re.search(
                "(BEGINCONTEXT\n*(.*)ENDCONTEXT\n*)ENDINPUT",
                instruction,
                re.DOTALL | re.MULTILINE,
            )
            if not re_match:
                logger.warning("Unexpected format, skipping...")
                continue
            fixed = instruction.replace(re_match.group(1), re_match.group(2))
            total = len(tokenizer(fixed + item["response"]).input_ids)
            if total > 4000:
                logger.warning("Context too large, skipping...")
                continue
            combined_items.append(as_conversation(fixed, item["response"].strip()))
            combined_items[-1]["source"] = f"airoboros-summarization-{source_name}"
    return Dataset.from_list(combined_items)


if __name__ == "__main__":
    print(load_data())
