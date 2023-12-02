import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from bagel.datasets.util import get_uid, has_refusal

CONFIDENCE = 1


def load_data():
    """Synthia v1.3 dataset."""
    logger.info("Loading Synthia-v1.3 dataset...")
    dataset = load_dataset("migtissera/Synthia-v1.3")
    data = []
    for item in tqdm(dataset["train"]):
        # Skip items that specifically reference vicuna format, too much work to properly parse...
        if "USER" in item.get("system") or re.search(
            r"[\"'](?:USER|ASSISTANT):?[\"']", item["instruction"]
        ):
            logger.warning(f"Skipping: {item['instruction']}")
            continue

        # Differentiate multi-turn.
        conv = []
        if item.get("system").strip():
            conv.append({"from": "system", "value": item["system"]})
        if "USER:" not in item["instruction"]:
            conv += [
                {"from": "human", "value": item["instruction"]},
                {"from": "gpt", "value": item["response"]},
            ]
        else:
            turns = re.findall(
                r"(^(?:(?:USER|ASSISTANT))?:?|(?:USER|ASSISTANT):)\s*(.*?)(?=(?:ASSISTANT|USER):|$)",
                item["instruction"],
                re.DOTALL,
            )
            for role, content in turns:
                role = "human" if not role.strip() or "USER" in role else "gpt"
                conv.append({"from": role, "value": content})
            conv.append({"from": "gpt", "value": item["response"]})
        data.append(
            {
                "id": get_uid(
                    "\n".join(
                        [turn["value"] for turn in conv if turn["from"] == "human"]
                    )
                ),
                "conversations": conv,
            }
        )
    return Dataset.from_list(data).filter(
        lambda item: not has_refusal(
            "\n".join([turn["value"] for turn in item["conversations"]])
        )
    )


if __name__ == "__main__":
    print(load_data())
