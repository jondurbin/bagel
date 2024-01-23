import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import as_conversation

PRIORITY = 1


def load_data(known_uids=set([]), **_):
    """LoLLMs question answer dataset."""
    logger.info("Loading LoLLMs-QNA dataset...")
    dataset = load_dataset("ParisNeo/lollms_aware_dataset", split="train")
    data = []
    for item in tqdm(dataset):
        as_conv = as_conversation(item["question"], item["answer"])
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        as_conv["source"] = "lollms"
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
