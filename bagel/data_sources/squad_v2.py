import random
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset, concatenate_datasets
from .util import as_conversation

PRIORITY = 3
BAD = [
    "IRRELEVANT",
    "DO NOT KNOW",
    "SKIP",
    "I don't know.",
    "The passage does not appear to provide an answer to the question.",
]


def load_data(known_uids=set([]), **_):
    """SquadV2 train split."""
    data = []
    logger.info("Loading SQuAD2.0 train split...")
    dataset = load_dataset("squad_v2", split="train")
    not_answered = dataset.filter(lambda item: not item["answers"]["text"])
    answered = dataset.filter(lambda item: item["answers"]["text"])
    dataset = concatenate_datasets(
        [
            not_answered.shuffle(seed=42).select(range(500)),
            answered.shuffle(seed=42).select(range(2500)),
        ]
    ).shuffle(seed=42)
    for item in tqdm(dataset):
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
