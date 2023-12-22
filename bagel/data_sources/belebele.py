from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

CONFIDENCE = 2


def load_data(known_uids=set([])):
    """belebele training split."""
    data = []
    dataset = load_dataset("facebook/belebele")
    for split in dataset:
        logger.info(f"Loading belebele train split -- {split}")
        for item in tqdm(dataset[split].shuffle(seed=42).select(range(300))):
            instruction = "\n".join(
                [
                    item["flores_passage"],
                    item["question"],
                ]
            )
            as_conv = as_conversation(
                instruction, item[f"mc_answer{item['correct_answer_num']}"]
            )
            if as_conv["id"] in known_uids:
                continue
            known_uids.add(as_conv["id"])
            data.append(as_conv)
            data[-1]["category"] = split
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
