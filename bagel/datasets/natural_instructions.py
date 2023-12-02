from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 1


def load_data():
    """Natural instructions train split."""
    data = []
    logger.info("Loading Natural Instructions train split...")
    dataset = load_dataset(
        "Muennighoff/natural-instructions", split="train"
    ).class_encode_column("task_name")
    dataset = dataset.train_test_split(
        train_size=100000, stratify_by_column="task_name"
    )["train"]
    for item in tqdm(dataset):
        question = "\n".join([item["definition"], item["inputs"]])
        data.append(as_conversation(question, item["targets"]))
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
