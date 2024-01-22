from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """Toxic DPO pairs."""
    logger.info("Loading comedy snippets...")
    data = []
    dataset = load_dataset(
        "unalignment/comedy-snippets-v0.1",
        data_files=["comedy-snippets.parquet"],
        split="train",
    )
    for item in dataset:
        data.append(
            {
                "id": get_uid(item["snippet"]),
                "source": "comedy-snippets",
                "prompt": None,
                "chosen": None,
                "rejected": None,
                "conversations": None,
                "text": item["snippet"],
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
