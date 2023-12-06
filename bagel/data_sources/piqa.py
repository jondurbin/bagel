import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

CONFIDENCE = 3


def load_data(known_uids=set([])):
    """piqa training split."""
    logger.info("Loading piqa train split...")
    data = []
    for item in tqdm(load_dataset("piqa", split="train")):
        instruction = "\n".join(
            [
                item["goal"],
                f"A. {item['sol1']}",
                f"B. {item['sol2']}",
            ]
        )
        coin = random.random()
        answer = "A" if item["label"] == 0 else "B"
        if coin <= 0.25:
            instruction += "\nOnly output the the letter of the correct answer."
        else:
            answer = (
                f'A. {item["sol1"]}' if item["label"] == 0 else f'B. {item["sol2"]}'
            )
        as_conv = as_conversation(instruction, answer)
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
