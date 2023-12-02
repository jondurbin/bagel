from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 3


def load_data():
    """Winogrande train split."""
    data = []
    logger.info("Loading winogrande train split...")
    for item in tqdm(load_dataset("winogrande", "winogrande_xl", split="train")):
        answer = item["option1"] if str(item["answer"]) == "1" else item["option2"]
        data.append(as_conversation(item["sentence"], answer))
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
