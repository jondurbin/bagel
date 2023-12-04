import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import as_conversation

CONFIDENCE = 3
BAD = [
    "IRRELEVANT",
    "DO NOT KNOW",
    "SKIP",
    "I don't know.",
    "The passage does not appear to provide an answer to the question.",
]


def load_data(known_uids=set([])):
    """SquadV2 train split."""
    data = []
    logger.info("Loading SQuAD2.0 train split...")
    for item in tqdm(load_dataset("squad_v2", split="train")):
        bad = random.choice(BAD)
        question = "\n".join(
            [
                f'Read the passage of text provided below, then concisely answer the question.  If you don\'t know, respond with "{bad}"',
                "",
                item["context"],
                "",
                item["question"],
            ]
        )
        answer = item["answers"]["text"][0] if item["answers"]["text"] else bad
        as_conv = as_conversation(question, answer)
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
