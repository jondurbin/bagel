import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

PRIORITY = 2


def load_data(known_uids=set([])):
    """DROP training split."""
    logger.info("Loading DROP train split...")
    data = []
    for item in tqdm(
        load_dataset("drop", split="train").shuffle(seed=42).select(range(5000))
    ):
        ordered = [item["passage"], item["question"]]
        if random.random() <= 0.5:
            ordered.reverse()
        instruction = "\n".join(ordered)
        answer = ", ".join([val for val in item["answers_spans"]["spans"]])
        as_conv = as_conversation(
            instruction,
            answer,
            system="You are a helpful assistant who answers as briefly as possible.",
        )
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
