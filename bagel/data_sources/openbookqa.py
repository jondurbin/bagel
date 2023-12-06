import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation


CONFIDENCE = 3


def load_data(known_uids=set([])):
    """OpenBookQA training split."""
    logger.info("Loading OpenBookQA train split...")
    data = []
    for item in tqdm(load_dataset("openbookqa", "main", split="train")):
        instruction = "\n".join(
            [
                item["question_stem"],
                "\n".join(
                    [
                        f"{item['choices']['label'][idx]}. {item['choices']['text'][idx]}"
                        for idx in range(len(item["choices"]["label"]))
                    ]
                ),
            ]
        )
        coin = random.random()
        answer = item["answerKey"]
        if coin <= 0.5:
            instruction += "\nOnly output the the letter of the correct answer."
        else:
            idx = item["choices"]["label"].index(item["answerKey"])
            answer = item["answerKey"] + " " + item["choices"]["text"][idx]
        as_conv = as_conversation(instruction, answer)
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
