import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

PRIORITY = 3


def load_data(known_uids=set([]), **_):
    """ROPES train split."""
    data = []
    logger.info("Loading ROPES train split...")
    dataset = load_dataset("ropes", split="train")
    for item in tqdm(dataset):
        question = "\n".join(
            [
                "Read the following background information, then apply the knowledge to the question that follows.",
                "",
                "Background:",
                item["background"],
                "",
                "Question:",
                item["situation"],
                item["question"],
            ]
        )
        answer = item["answers"]["text"][0]
        as_conv = as_conversation(
            question,
            answer,
            system="You are a helpful assistant that answers questions concisely.",
        )
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
