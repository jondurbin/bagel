from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

PRIORITY = 3


def load_data(known_uids=set([]), **_):
    """MathInstruct training split."""
    logger.info("Loading MathInstruct train split...")
    data = []
    dataset = (
        load_dataset("TIGER-Lab/MathInstruct", split="train")
        .class_encode_column("source")
        .train_test_split(train_size=75000, stratify_by_column="source")["train"]
    )
    for item in tqdm(dataset):
        as_conv = as_conversation(item["instruction"], item["output"])
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
