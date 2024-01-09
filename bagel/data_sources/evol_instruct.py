import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import as_conversation, has_refusal

PRIORITY = 2


def load_data(known_uids=set([])):
    """WizardLM evol instruct dataset"""
    logger.info("Loading WizardLM evol instruct 70k dataset...")
    dataset = load_dataset("WizardLM/WizardLM_evol_instruct_70k")
    data = []
    for item in tqdm(dataset["train"]):
        if has_refusal(item["output"]):
            continue
        as_conv = as_conversation(item["instruction"], item["output"])
        if as_conv["id"] in known_uids:
            continue
        data.append(as_conv)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
