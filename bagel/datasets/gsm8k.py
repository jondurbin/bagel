from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 3


def load_data():
    """GSM8K training split."""
    logger.info("Loading GSM8K train split...")
    data = []
    for item in tqdm(load_dataset("gsm8k", "main", split="train")):
        data.append(as_conversation(item["question"], item["answer"]))
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
