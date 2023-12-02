from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from bagel.datasets.util import has_refusal, get_uid

CONFIDENCE = 2


def load_data():
    """SlimOrca dataset."""
    logger.info("Loading SlimOrca dataset...")
    dataset = load_dataset("Open-Orca/SlimOrca")["train"].filter(
        lambda item: not has_refusal(
            "\n".join([turn["value"] for turn in item["conversations"]])
        )
    )
    data = []
    for item in tqdm(dataset):
        data.append(
            {
                "conversations": [
                    {
                        key: value
                        for key, value in turn.items()
                        if key in ("from", "value")
                    }
                    for turn in item["conversations"]
                ]
            }
        )
        data[-1]["id"] = get_uid(
            "\n".join(
                [
                    turn["value"]
                    for turn in data[-1]["conversations"]
                    if turn["from"] == "human"
                ]
            )
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
