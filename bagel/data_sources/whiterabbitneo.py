import re
from tqdm import tqdm
from loguru import logger
from datasets import concatenate_datasets, load_dataset, Dataset
from .util import get_uid, has_refusal, as_conversation

PRIORITY = 1


def load_data(known_uids=set([]), **_):
    """WhiteRabbitNeo dataset."""
    logger.info("Loading WhiteRabbitNeo dataset...")
    dataset = (
        concatenate_datasets(
            [
                load_dataset("WhiteRabbitNeo/WRN-Chapter-1", split="train"),
                load_dataset("WhiteRabbitNeo/WRN-Chapter-2", split="train"),
            ]
        )
        .filter(lambda item: not has_refusal(item["response"]))
        .map(lambda item: as_conversation(item["instruction"], item["response"]))
        .filter(lambda item: item["id"] not in known_uids)
    )
    for item in dataset:
        known_uids.add(item["id"])
    return dataset


if __name__ == "__main__":
    print(load_data())
