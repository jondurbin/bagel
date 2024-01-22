from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """Gutenberg DPO pairs."""
    logger.info("Loading Gutenberg DPO dataset...")
    dataset = load_dataset("jondurbin/gutenberg-dpo-v0.1", split="train")
    data = []
    for item in dataset:
        # We don't care about the known UIDs here, since we are using it for DPO.
        data.append(
            {
                "id": get_uid(item["prompt"]),
                "source": "gutenberg-dpo",
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "conversations": [
                    {
                        "from": "human",
                        "value": item["prompt"],
                    },
                    {
                        "from": "gpt",
                        "value": item["chosen"],
                    },
                ],
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
