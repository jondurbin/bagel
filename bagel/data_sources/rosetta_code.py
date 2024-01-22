from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """RosettaCode data."""
    logger.info("Loading RosettaCode train split...")
    data = []
    # Make sure we filter out humaneval entrites...
    human_eval = [
        item["canonical_solution"]
        for item in load_dataset("openai_humaneval", split="test")
    ]
    for item in tqdm(
        load_dataset("cakiki/rosetta-code", split="train").train_test_split(
            train_size=75000
        )["train"]
    ):
        instruction = "\n".join(
            [
                f"Create a solution in {item['language_name']} to the following:",
                item["task_description"],
            ]
        )
        code = item["code"]
        if any([solution in code for solution in human_eval]) or any(
            [code in solution for solution in human_eval]
        ):
            logger.warning("Rejecting humaneval contamination...")
            continue
        as_conv = as_conversation(instruction, code)
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
