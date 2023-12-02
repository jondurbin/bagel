import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 2


def load_data():
    """DROP training split."""
    logger.info("Loading DROP train split...")
    data = []
    for item in tqdm(load_dataset("drop", split="train")):
        ordered = [item["passage"], item["question"]]
        if random.random() <= 0.5:
            ordered.reverse()
        instruction = "\n".join(ordered)
        answer = ", ".join([val for val in item["answers_spans"]["spans"]])
        data.append(
            as_conversation(
                instruction,
                answer,
                system="You are a helpful assistant who answers as briefly as possible.",
            )
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
