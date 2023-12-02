from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation, has_refusal

CONFIDENCE = 1


def load_data():
    """Python alpaca."""
    data = []
    logger.info("Loading python alpaca...")
    for item in tqdm(load_dataset("Vezora/Tested-188k-Python-Alpaca", split="train")):
        if has_refusal(item["output"]):
            continue
        data.append(as_conversation(item["instruction"], item["output"]))
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
