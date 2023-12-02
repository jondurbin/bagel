from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 1


def load_data():
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
        data.append(as_conversation(question, item["query"]))
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
