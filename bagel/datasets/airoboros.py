from loguru import logger
from datasets import load_dataset

CONFIDENCE = 1


def load_data():
    """Airoboros 3.1 dataset (with unalignment)."""
    logger.info("Loading airoboros-3.1 dataset...")
    dataset = load_dataset(
        "unalignment/spicy-3.1", data_files=["conversations-no-mathjson.json"]
    )["train"]
    dataset = dataset.add_column(
        "source", [f"airoboros_{item['category']}" for item in dataset]
    )
    dataset = dataset.remove_columns(["category"])
    return dataset


if __name__ == "__main__":
    print(load_data()[0])
