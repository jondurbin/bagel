import csv
import json
import requests
import uuid
from loguru import logger
from datasets import Dataset

CONFIDENCE = 3


def load_data(known_uids=set([])):
    """EmoBank dataset."""
    logger.info("Loading EmoBank dataset...")
    raw_data = requests.get(
        "https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv"
    ).text
    csv_reader = csv.DictReader(raw_data.splitlines())
    data = []
    for item in csv_reader:
        if item["split"] != "train":
            continue
        text = item["text"].strip().rstrip('"').rstrip()
        data.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_OID, item["id"])),
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Please assign a Valence-Arousal-Dominance (VAD) score in JSON format to the following message:\n{text}",
                    },
                    {
                        "from": "gpt",
                        "value": json.dumps(
                            {"V": item["V"], "A": item["A"], "D": item["D"]}, indent=2
                        ),
                    },
                ],
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
