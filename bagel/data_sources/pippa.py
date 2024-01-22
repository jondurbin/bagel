import json
import random
import requests
from loguru import logger
from datasets import Dataset
from .util import has_refusal

PRIORITY = 1


def load_data(known_uids=set([]), **_):
    """Pippa dataset, filtered."""
    logger.info("Loading PIPPA dataset...")
    raw_data = [
        json.loads(line)
        for line in requests.get(
            "https://huggingface.co/datasets/kingbri/PIPPA-shareGPT/resolve/main/pippa_sharegpt_trimmed.jsonl?download=true"
        ).text.splitlines()
    ]
    keep = []
    for item in raw_data:
        if len(item["conversations"]) < 3:
            continue
        if has_refusal("\n".join([conv["value"] for conv in item["conversations"]])):
            continue

        # If the conversation starts with GPT, we'll add the first message to system prompt, otherwise it messes up llama-2 chat.
        if (
            item["conversations"][0]["from"] == "system"
            and item["conversations"][1]["from"] == "gpt"
        ):
            item["conversations"][0]["value"] = (
                "First message: " + item["conversations"][1]["value"]
            )
            del item["conversations"][1]
        elif item["conversations"][0]["from"] == "gpt":
            continue
        if random.random() <= 0.25:
            keep.append({"id": item["id"], "conversations": item["conversations"]})
    return Dataset.from_list(keep)


if __name__ == "__main__":
    print(load_data())
