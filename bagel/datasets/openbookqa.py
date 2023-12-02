import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation


CONFIDENCE = 3


def load_data():
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
        data.append(as_conversation(instruction, answer))
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
