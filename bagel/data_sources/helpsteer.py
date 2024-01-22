from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid, has_refusal

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """HelpSteer dataset for DPO."""
    logger.info("Loading HelpSteer dataset...")
    dataset = load_dataset("nvidia/HelpSteer", split="train").filter(
        lambda item: not has_refusal(item["response"])
    )

    # Use the highest correctness items as the "chosen" responses.
    logger.info("Generating DPO pairs...")
    options = {}
    for item in dataset:
        uid = get_uid(item["prompt"])
        if uid not in options:
            options[uid] = []
        item["score"] = (
            item["helpfulness"]
            + item["correctness"]
            + item["coherence"]
            + item["complexity"]
            + item["verbosity"]
        )
        options[uid].append(item)

    # Select the chosen and rejected items.
    data = []
    for uid, responses in options.items():
        if len(responses) < 2:
            continue
        if uid in known_uids:
            continue
        known_uids.add(uid)
        chosen = None
        chosen_score = 0
        rejected = None
        rejected_score = 0
        for item in responses:
            if item["correctness"] == 4:
                if chosen:
                    if item["score"] > chosen["score"]:
                        rejected = chosen["response"]
                        rejected_score = chosen["score"]
                        chosen = item
                        chosen_score = item["score"]
                        break
                else:
                    chosen = item
                    chosen_score = item["score"]
            else:
                if not rejected:
                    rejected = item["response"]
                    rejected_score = item["score"]
        if chosen and rejected and chosen["response"] != rejected:
            logger.success(f"Found DPO pair: {chosen_score} vs {rejected_score}")
            data.append(
                {
                    "id": uid,
                    "source": "helpsteer",
                    "prompt": item["prompt"],
                    "chosen": chosen["response"],
                    "rejected": rejected,
                    "conversations": None,
                }
            )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
