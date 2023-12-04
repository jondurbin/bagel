import json
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 2


def load_data(known_uids=set([])):
    """APPS training split."""
    logger.info("Loading APPS train split...")
    data = []
    for item in tqdm(load_dataset("codeparrot/apps", "all", split="train")):
        instruction = "\n".join(
            [
                "Provide a python solution to the following:",
                item["question"],
            ]
        )
        solution = json.loads(item["solutions"])[0]
        as_conv = as_conversation(instruction, solution)
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
