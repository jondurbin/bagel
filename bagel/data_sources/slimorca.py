from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import has_refusal, get_uid

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """SlimOrca dataset."""
    logger.info("Loading SlimOrca dataset...")

    # Airoboros 3.2 includes some ~11k slimorca examples that were extended to have
    # multiple, related turns.  We'll exclude that set here.
    skip = set(
        [
            item["conversations"][1]["value"].strip()
            for item in load_dataset("jondurbin/airoboros-3.2", split="train").filter(
                lambda item: item["category"] == "slimorca_multiturn"
            )
        ]
    )

    def _should_skip(item):
        offset = 0
        if item["conversations"][0]["from"] == "system":
            offset = 1
        return item["conversations"][offset]["value"].strip() in skip

    dataset = (
        load_dataset("Open-Orca/SlimOrca", split="train")
        .filter(
            lambda item: not has_refusal(
                "\n".join([turn["value"] for turn in item["conversations"]])
            )
            and not _should_skip(item)
        )
        .shuffle(seed=42)
        .select(range(200000))
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
