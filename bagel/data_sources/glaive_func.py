import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid, has_refusal

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """Glaive function calling v2 dataset."""
    logger.info("Loading glaive function calling dataset...")
    dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    data = []
    for item in tqdm(dataset):
        parts = re.findall(
            "(USER|ASSISTANT|FUNCTION RESPONSE):(.*?)(?=(?:USER|ASSISTANT|FUNCTION RESPONSE):|$)",
            item["chat"],
            re.DOTALL,
        )
        if not parts:
            continue
        conversations = [
            {"from": "system", "value": item["system"].replace("SYSTEM: ", "")}
        ]
        for role, response in parts:
            if role == "USER":
                conversations.append(
                    {
                        "from": "human",
                        "value": response.strip(),
                    }
                )
            elif role == "ASSISTANT":
                if "<functioncall>" in response:
                    response = response.replace(
                        "<functioncall>", "<|begin_func|>"
                    ).replace("<|endoftext|>", "<|end_func|>")
                else:
                    response = response.replace("<|endoftext|>", "")
                conversations.append(
                    {
                        "from": "gpt",
                        "value": response.strip(),
                    }
                )
            elif role == "FUNCTION RESPONSE":
                conversations.append(
                    {
                        "from": "human",
                        "value": f"<|begin_func_response|>{response.strip()}<|end_func_response|>",
                    }
                )
        if (
            len(conversations) < 3
            or conversations[1]["from"] != "human"
            or conversations[-1]["from"] != "gpt"
        ):
            logger.warning("Unexpected format, skipping...")
            continue
        data.append(
            {
                "id": get_uid(item["system"] + item["chat"]),
                "source": "glaive-func",
                "conversations": conversations,
            }
        )
    return Dataset.from_list(data).shuffle(seed=42).select(range(10000))


if __name__ == "__main__":
    print(load_data())
