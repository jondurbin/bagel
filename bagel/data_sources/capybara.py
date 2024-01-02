from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import get_uid


PRIORITY = 2


def load_data(known_uids=set([])):
    """Capybara training split."""
    logger.info("Loading Capybara train split...")
    data = []
    for item in tqdm(load_dataset("ldjnr/capybara", split="train")):
        data.append(
            {
                "id": get_uid(
                    "\n".join([conv["input"] for conv in item["conversation"]])
                ),
                "conversations": [],
            }
        )
        for conv in item["conversation"]:
            data[-1]["conversations"] += [
                {
                    "from": "human",
                    "value": conv["input"],
                },
                {
                    "from": "gpt",
                    "value": conv["output"],
                },
            ]
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
