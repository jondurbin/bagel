from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from .util import as_conversation, has_refusal

CONFIDENCE = 1


def load_data(known_uids=set([])):
    """Python alpaca."""
    data = []
    logger.info("Loading python alpaca...")
    # Make sure we filter out humaneval entrites...
    human_eval = [
        item["canonical_solution"]
        for item in load_dataset("openai_humaneval", split="test")
    ]
    for item in tqdm(load_dataset("Vezora/Tested-22k-Python-Alpaca", split="train")):
        if has_refusal(item["output"]):
            continue
        if any([solution in item["output"] for solution in human_eval]) or any(
            [item["output"] in solution for solution in human_eval]
        ):
            logger.warning("Rejecting humaneval contamination...")
            continue
        as_conv = as_conversation(item["instruction"], item["output"])
        if as_conv["id"] in known_uids:
            continue
        known_uids.add(as_conv["id"])
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
