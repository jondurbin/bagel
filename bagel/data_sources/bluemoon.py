import json
import requests
import uuid
from loguru import logger
from datasets import Dataset

PRIORITY = 1


def load_data(known_uids=set([])):
    """Bloomoon fandom RP dataset."""
    logger.info("Loading Bluemoon Fandom RP dataset...")
    raw_data = json.loads(
        requests.get(
            "https://huggingface.co/datasets/Squish42/bluemoon-fandom-1-1-rp-cleaned/resolve/main/data/pruned/bluemoon.train.json?download=true"
        ).text
    )
    return Dataset.from_list(
        [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_OID, str(item["id"]))),
                "conversations": item["conversations"],
            }
            for item in raw_data
        ]
    )


if __name__ == "__main__":
    print(load_data())
