from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

PRIORITY = 2


def load_data(known_uids=set([])):
    """SQL Create Context dataset."""
    data = []
    logger.info("Loading sql-create-context...")
    for item in tqdm(load_dataset("b-mc2/sql-create-context", split="train")):
        question = "\n".join(
            [
                "Using the context provided, please generate a SQL query to answer the question.",
                f"Context: {item['context']}",
                f"Question: {item['question']}",
            ]
        )
        as_conv = as_conversation(question, item["answer"])
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return (
        Dataset.from_list(data)
        .shuffle(seed=42)
        .train_test_split(train_size=10000)["train"]
    )


if __name__ == "__main__":
    print(load_data())
