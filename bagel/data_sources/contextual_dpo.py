from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """Contextual DPO pairs."""
    logger.info("Loading contextual DPO dataset...")
    dataset = load_dataset("jondurbin/contextual-dpo-v0.1", split="train")
    data = []
    for item in dataset:
        # We don't care about the known UIDs here, since we are using it for DPO.
        if item["chosen"] != item["rejected"]:
            data.append(
                {
                    "id": get_uid(item["prompt"]),
                    "source": "contextual-dpo",
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "conversations": None,
                }
            )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
