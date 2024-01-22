from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """Toxic DPO pairs."""
    logger.info("Loading Toxic DPO dataset...")
    dataset = load_dataset(
        "unalignment/toxic-dpo-v0.2", data_files=["toxic-dpo.parquet"], split="train"
    )
    data = []
    for item in dataset:
        # We don't care about the known UIDs here.
        if item["chosen"] != item["rejected"]:
            data.append(
                {
                    "id": get_uid(item["prompt"]),
                    "source": "toxic-dpo",
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
