from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid, has_refusal

PRIORITY = 2


def load_data(known_uids=set([])):
    """Orca DPO pairs dataset."""
    logger.info("Loading distilabel_orca_dpo_pairs dataset...")
    dataset = load_dataset(
        "argilla/distilabel-intel-orca-dpo-pairs", split="train"
    ).filter(
        lambda item: not has_refusal(item["chosen"])
        and item["status"] != "tie"
        and item["chosen_score"] >= 8
        and not item["in_gsm8k_train"]
        and item["chosen"] != item["rejected"]
    )
    data = []
    for item in dataset:
        # We don't care about the known UIDs here, since we are using it for DPO.
        data.append(
            {
                "id": get_uid(item["input"]),
                "source": "orca_dpo_pairs",
                "prompt": item["input"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "conversations": None,
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
