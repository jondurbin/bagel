from loguru import logger
from datasets import load_dataset, Dataset
from bagel.datasets.util import get_uid, has_refusal

CONFIDENCE = 2


def load_data(known_uids=set([])):
    """Orca DPO pairs dataset."""
    logger.info("Loading orca_dpo_pairs dataset...")
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train").filter(
        lambda item: not has_refusal(item["chosen"])
    )

    logger.info("Formatting...")
    data = []
    for item in dataset:
        # We don't care about the known UIDs here, since we are using it for DPO.
        data.append(
            {
                "id": get_uid(item["question"]),
                "prompt": item["question"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "conversations": None,
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
