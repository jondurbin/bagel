import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import as_conversation, has_refusal


PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """Various Camel-AI datasets."""
    data = []
    for expert in ["chemistry", "physics", "math", "biology"]:
        logger.info(f"Loading camel-ai {expert} dataset...")
        dataset = (
            load_dataset(f"camel-ai/{expert}", split="train")
            .filter(lambda item: not has_refusal(item["message_2"]))
            .class_encode_column("topic;")
            .train_test_split(train_size=5000, stratify_by_column="topic;")["train"]
        )
        for item in tqdm(dataset):
            as_conv = as_conversation(item["message_1"], item["message_2"])
            if as_conv["id"] in known_uids:
                continue
            data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
