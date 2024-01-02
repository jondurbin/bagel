import json
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation

PRIORITY = 2


def load_data(known_uids=set([])):
    """APPS training split."""
    logger.info("Loading APPS train split...")
    data = []
    # Make sure we filter out humaneval entrites...
    human_eval = [
        item["canonical_solution"]
        for item in load_dataset("openai_humaneval", split="test")
    ]
    for item in tqdm(load_dataset("codeparrot/apps", "all", split="train")):
        instruction = "\n".join(
            [
                "Provide a python solution to the following:",
                item["question"],
            ]
        )
        first_solution = json.loads(item["solutions"])[0]
        if any([solution in first_solution for solution in human_eval]) or any(
            [first_solution in solution for solution in human_eval]
        ):
            logger.warning("Rejecting humaneval contamination...")
            continue
        as_conv = as_conversation(instruction, first_solution)
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
