from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid, has_refusal

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """ultrafeedback dataset for DPO."""
    logger.info("Loading ultrafeedback dataset...")
    dataset = load_dataset(
        "allenai/ultrafeedback_binarized_cleaned", split="train_gen"
    ).filter(
        lambda item: item["score_chosen"] >= 8
        and len(item["chosen"]) == 2
        and item["chosen"][1]["content"].strip()
        and item["chosen"][1]["role"] == "assistant"
        and not has_refusal(item["chosen"][1]["content"])
        and item["chosen"][1]["content"].lower().strip()
        != item["rejected"][1]["content"].lower().strip()
    )

    logger.info("Formatting...")
    data = []
    for item in dataset:
        uid = get_uid(item["chosen"][0]["content"])
        if uid in known_uids:
            continue
        known_uids.add(uid)
        data.append(
            {
                "id": uid,
                "source": "ultrafeedback",
                "prompt": item["prompt"],
                "chosen": item["chosen"][1]["content"],
                "rejected": item["rejected"][1]["content"],
                "conversations": None,
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
