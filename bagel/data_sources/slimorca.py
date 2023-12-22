from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import has_refusal, get_uid

CONFIDENCE = 2


def load_data(known_uids=set([])):
    """SlimOrca dataset."""
    logger.info("Loading SlimOrca dataset...")
    dataset = (
        load_dataset("Open-Orca/SlimOrca")["train"]
        .filter(
            lambda item: not has_refusal(
                "\n".join([turn["value"] for turn in item["conversations"]])
            )
        )
        .shuffle(seed=42)
        .select(range(100000))
    )
    data = []
    for item in tqdm(dataset):
        uid = get_uid(
            "\n".join(
                [
                    turn["value"]
                    for turn in item["conversations"]
                    if turn["from"] in ("user", "human")
                ]
            )
        )
        if uid in known_uids:
            continue
        known_uids.add(uid)
        data.append(
            {
                "id": uid,
                "conversations": [
                    {
                        key: value
                        for key, value in turn.items()
                        if key in ("from", "value")
                    }
                    for turn in item["conversations"]
                ],
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
