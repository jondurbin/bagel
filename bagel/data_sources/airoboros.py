import json
import requests
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid

PRIORITY = 1


def load_data(known_uids=set([])):
    """Airoboros 3.2 dataset."""
    logger.info("Loading airoboros-3.2 dataset...")
    dataset = load_dataset("jondurbin/airoboros-3.2", split="train")
    dataset = dataset.add_column(
        "source", [f"airoboros_{item['category']}" for item in dataset]
    )

    # We can also look back into earlier versions of airoboros datasets to find
    # writing examples that were rewritten to be longer/less cliche.
    old_dataset = [
        json.loads(line)
        for line in requests.get(
            "https://huggingface.co/datasets/jondurbin/airoboros-gpt4-m2.0/resolve/main/instructions.jsonl?download=true"
        ).text.splitlines()
    ]
    by_instruction = {
        item["instruction"].lower().strip(): item
        for item in old_dataset
        if item.get("category") in ("roleplay", "writing")
    }

    # Differentiate DPO pairs.
    data = []
    logger.info("Finding DPO pairs...")
    for item in tqdm(dataset):
        new_id = get_uid(
            "\n".join(
                [
                    turn["value"]
                    for turn in item["conversations"]
                    if turn["from"] in ("human", "system")
                ]
            )
        )
        if new_id in known_uids:
            continue
        known_uids.add(new_id)
        if (
            item.get("category") in ("roleplay", "writing")
            and item["conversations"][1]["from"] == "human"
        ):
            instruction = item["conversations"][1]["value"]
            key = instruction.lower().strip()
            if key in by_instruction:
                new_length = len(item["conversations"][2]["value"])
                old_length = len(by_instruction[key]["response"])
                if new_length > old_length:
                    logger.success(
                        f"Found rewritten response: {new_length} vs {old_length}"
                    )
                    data.append(
                        {
                            "id": new_id,
                            "source": item["source"],
                            "prompt": item["conversations"][1]["value"],
                            "chosen": item["conversations"][2]["value"],
                            "rejected": by_instruction[key]["response"],
                            "conversations": item["conversations"],
                        }
                    )
                    continue
        data.append(item)
        data[-1].update(
            {
                "id": new_id,
                "prompt": None,
                "rejected": None,
                "chosen": None,
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
