import re
import hashlib
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset
from bagel.datasets.util import map_conv_format, get_uid, has_refusal

CONFIDENCE = 2


def load_data(known_uids=set([])):
    """lmsys 1 million chat dataset."""
    logger.info("Loading lmsys-chat-1m gpt-4 dataset...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

    # Calculate a simple digest of the inputs, so we can find the same prompts sent to multiple models.
    digests = [
        hashlib.md5(
            "\n".join(
                [
                    turn["content"]
                    for turn in item["conversation"]
                    if turn["role"] == "user"
                ]
            ).encode()
        ).hexdigest()
        for item in dataset
    ]
    dataset = dataset.add_column("digest", digests)

    # Now, we can try to find GPT-4 pairs to use for DPO.
    digests = {}
    gpt4_items = []
    logger.info("Looking for DPO pairs...")
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        if item["model"] == "gpt-4" and not has_refusal(
            "\n".join(
                [
                    turn["content"]
                    for turn in item["conversation"]
                    if turn["role"] == "assistant"
                ]
            )
        ):
            gpt4_items.append(item)
            continue

        # We'll use the largest size model's response as the rejected value.
        m = re.search("-([0-9]{2})b", item["model"])
        if not m:
            continue
        size = int(m.group(1))

        # Only keep the largest model's output for the DPO pair.
        keep = False
        if item["digest"] in digests:
            old_size = digests[item["digest"]]["size"]
            if size > old_size:
                keep = True
        else:
            keep = True
        if keep:
            digests[item["digest"]] = {
                "model": item["model"],
                "idx": idx,
                "size": size,
            }

    # Now we can actually generate the DPO pairs.
    dpo_pairs = []
    gpt4_data = []
    for item in tqdm(gpt4_items):
        inputs = "\n".join(
            [turn["content"] for turn in item["conversation"] if turn["role"] == "user"]
        )
        uid = get_uid(inputs)
        if uid in known_uids:
            continue
        known_uids.add(uid)
        save_item = map_conv_format({"id": uid, "conversation": item["conversation"]})
        save_item["prompt"] = None
        save_item["chosen"] = None
        save_item["rejected"] = None
        if item["digest"] not in digests or len(item["conversation"]) != 2:
            gpt4_data.append(save_item)
            continue
        responses = "\n".join(
            [
                turn["content"]
                for turn in item["conversation"]
                if turn["role"] == "assistant"
            ]
        )
        alt_responses = "\n".join(
            [
                turn["content"]
                for turn in dataset[digests[item["digest"]]["idx"]]["conversation"]
                if turn["role"] == "assistant"
            ]
        )
        if alt_responses == responses:
            logger.warning("Same response from rejected...")
            gpt4_data.append(save_item)
            continue
        logger.success(f"Found alternative: {digests[item['digest']]}")
        prompt = save_item["conversations"][0]["value"]
        response = save_item.pop("conversations")[-1]["value"]
        save_item["prompt"] = prompt
        save_item["chosen"] = response
        save_item["rejected"] = dataset[digests[item["digest"]]["idx"]]["conversation"][
            -1
        ]["content"]
        dpo_pairs.append(save_item)
    logger.success(f"Found {len(dpo_pairs)} DPO samples and {len(gpt4_data)} SFT items")
    return Dataset.from_list(gpt4_data + dpo_pairs)


if __name__ == "__main__":
    print(load_data())
