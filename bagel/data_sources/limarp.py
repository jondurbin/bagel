import json
import random
import requests
import transformers
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import has_refusal, get_uid

PRIORITY = 1


def load_data(known_uids=set([]), tokenizer=None):
    """LimaRP dataset."""
    logger.info("Loading LimaRP dataset...")
    dataset = load_dataset("grimulkan/LimaRP-augmented", split="train")
    data = []
    if not tokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "mistralai/mistral-7b-v0.1", use_fast=True
        )
    original_chat_template = tokenizer.chat_template
    tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}"
    for item in tqdm(dataset):
        all_text = "\n".join([conv["value"] for conv in item["conversations"]])
        uid = get_uid(all_text)
        if uid in known_uids:
            continue
        if len(item["conversations"]) < 3:
            continue
        if has_refusal(all_text):
            continue

        # If the conversation starts with GPT, we'll add the first message to system prompt, otherwise it messes up llama-2 chat.
        if (
            item["conversations"][0]["from"] == "system"
            and item["conversations"][1]["from"] == "gpt"
        ):
            item["conversations"][0]["value"] = (
                "First message: " + item["conversations"][1]["value"]
            )
            del item["conversations"][1]
        elif item["conversations"][0]["from"] == "gpt":
            continue

        # Truncate to fit standard 4k context window.
        mapped = [
            {
                "role": "user" if turn["from"] == "human" else "assistant",
                "content": turn["value"].strip(),
            }
            for turn in item["conversations"]
        ]
        valid = False
        for idx in range(len(mapped), 3, -1):
            length = len(tokenizer.apply_chat_template(item["conversations"][0:idx]))
            if length < 4096:
                valid = True
                break
        if valid:
            data.append(
                {
                    "id": uid,
                    "conversations": item["conversations"][0:idx],
                    "source": "limarp",
                }
            )
            known_uids.add(uid)
    tokenizer.chat_template = original_chat_template
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
