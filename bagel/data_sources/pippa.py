import json
import requests
from loguru import logger
from datasets import Dataset
from .util import has_refusal

CONFIDENCE = 1


def load_data(known_uids=set([])):
    """Pippa dataset, filtered."""
    logger.info("Loading PIPPA dataset...")
    raw_data = [
        json.loads(line)
        for line in requests.get(
            "https://huggingface.co/datasets/kingbri/PIPPA-shareGPT/resolve/main/pippa_sharegpt_trimmed.jsonl?download=true"
        ).text.splitlines()
    ]
    return Dataset.from_list(
        [
            {"id": item["id"], "conversations": item["conversations"]}
            for item in raw_data
            if not has_refusal(
                "\n".join([conv["value"] for conv in item["conversations"]])
            )
        ]
    )


if __name__ == "__main__":
    print(load_data())
