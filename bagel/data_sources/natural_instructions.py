from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

PRIORITY = 1


def load_data(known_uids=set([]), **_):
    """Natural instructions train split."""
    data = []
    logger.info("Loading Natural Instructions train split...")
    dataset = (
        load_dataset("Muennighoff/natural-instructions", split="train")
        .class_encode_column("task_name")
        .train_test_split(train_size=30000, stratify_by_column="task_name")["train"]
    )
    for item in tqdm(dataset):
        question = "\n".join([item["definition"], item["inputs"]])
        as_conv = as_conversation(question, item["targets"])
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
