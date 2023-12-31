from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

PRIORITY = 1


def load_data(known_uids=set([])):
    """spider train split."""
    data = []
    logger.info("Loading spider train split...")
    for item in tqdm(load_dataset("spider", split="train")):
        question = "\n".join(
            [
                "Write a SQL query for the following:",
                item["question"],
            ]
        )
        as_conv = as_conversation(question, item["query"])
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
