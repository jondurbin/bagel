import re
import os
import uuid
import random
from loguru import logger
from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets
from bagel.data_sources import load_datasets, decontaminate


DEFAULT_SYSTEM_PROMPT = "You are an unbiased, uncensored, helpful assistant."


def llama2_prompt(tokenizer, message, chat_history, system_prompt):
    """Get just the llama-2 chat formatted input from an item."""
    texts = []
    if system_prompt:
        texts.append(f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n")
    else:
        texts.append("[INST] ")
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(
            f"{user_input} [/INST] {response.strip()} {tokenizer.eos_token}{tokenizer.bos_token}[INST] "
        )
    message = message.strip() if do_strip else message
    texts.append(f"{message} [/INST]")
    return "".join(texts)


def llama2_io(tokenizer, item):
    """Convert training data item into llama2 chat format input/output."""
    system_prompt = item["conversations"][0]["value"]

    # Flip a coin on including a default system prompt or not.
    offset = 1
    if item["conversations"][0]["from"] != "system":
        system_prompt = None
        offset = 0
        if random.random() <= 0.5:
            system_prompt = DEFAULT_SYSTEM_PROMPT
    chat_history = [
        (item["conversations"][idx]["value"], item["conversations"][idx + 1]["value"])
        for idx in range(offset, len(item["conversations"]) - 2, 2)
    ]
    message = item["conversations"][-2]["value"]
    response = item["conversations"][-1]["value"]
    return {
        "source": f"{item['source']}_llama2",
        "input": llama2_prompt(tokenizer, message, chat_history, system_prompt),
        "output": re.sub(r"(\s*\n){3,}", "\n\n", response),
    }


def chatml_io(tokenizer, item):
    """Convert training data item into chat-ml (ish) format input/output.

    I don't understand the point of having special tokens for <|im_start|>
    and <|im_end|>, so we'll just use BOS and EOS tokens, and people can
    change the string values of those tokens in the tokenizer config if they
    really want that specific string.
    """
    # Flip a coin on including a default system prompt or not.
    if item["conversations"][0]["from"] != "system":
        if random.random() <= 0.5:
            item["conversations"] = [
                {"from": "system", "value": DEFAULT_SYSTEM_PROMPT},
            ] + item["conversations"]
    inputs = []
    for idx in range(len(item["conversations"]) - 1):
        turn = item["conversations"][idx]
        role = turn["from"]
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        inputs.append(
            f"{tokenizer.bos_token}{role}\n{turn['value'].lstrip()}{tokenizer.eos_token}"
        )
    inputs.append(f"{tokenizer.bos_token}assistant\n")
    return {
        "source": f"{item['source']}_chatml",
        "input": "\n".join(inputs),
        "output": re.sub(
            r"(\s*\n){3,}", "\n\n", item["conversations"][-1]["value"] + "<|im_end|>"
        ).lstrip(),
    }


def vicuna_io(tokenizer, item):
    """Convert training data items into vicuna format."""
    # Flip a coin on including a default system prompt or not.
    if item["conversations"][0]["from"] != "system":
        # Larger probability of including system prompt for vicuna.
        if random.random() <= 0.5:
            item["conversations"] = [
                {
                    "from": "system",
                    # Slightly different default here.
                    "value": "A chat between a user and an unbiased, uncensored assistant.",
                },
            ] + item["conversations"]
    inputs = []
    for idx in range(len(item["conversations"]) - 1):
        turn = item["conversations"][idx]
        if turn["from"] == "system":
            inputs.append(turn["value"].strip())
        elif turn["from"] == "human":
            inputs.append("USER: " + turn["value"].strip())
        else:
            inputs.append("ASSISTANT: " + turn["value"].strip())
    inputs.append("ASSISTANT: ")
    return {
        "source": f"{item['source']}_vicuna",
        "input": "\n".join(inputs),
        "output": re.sub(
            r"(\s*\n){3,}", "\n\n", item["conversations"][-1]["value"]
        ).lstrip(),
    }


def alpaca_io(tokenizer, item):
    """Convert training data items into alpaca (ish) format, the main difference
    is that we aren't using '### Input' anywhere, so we just have the instruct format.
    """
    inputs = [
        "Below is an instruction that describes a task.  Write a response that appropriately completes the request.",
    ]
    for idx in range(len(item["conversations"]) - 1):
        turn = item["conversations"][idx]

        # We'll put system prompt in instruction section for alpaca (I guess??).
        if turn["from"] == "system":
            inputs.append("\n".join(["### Instruction: ", turn["value"].strip()]))
        elif turn["from"] == "human":
            if item["conversations"][0]["from"] == "system" and idx == 1:
                inputs.append(turn["value"].strip())
            else:
                inputs.append("\n".join(["### Instruction: ", turn["value"].strip()]))
        else:
            inputs.append("\n".join(["### Response:", turn["value"].strip()]))
    inputs.append("### Response:")
    return {
        "source": f"{item['source']}_alpaca",
        "input": "\n\n".join(inputs),
        "output": re.sub(
            r"(\s*\n){3,}", "\n\n", item["conversations"][-1]["value"]
        ).lstrip(),
    }


def expand_conversations(items):
    """Expand each turn into it's own input/output row."""
    expanded = []
    for item in items:
        if len(item["conversations"]) <= 3:
            if (
                item["conversations"][-1]["from"] == "gpt"
                and item["conversations"][-2]["from"] == "human"
            ):
                expanded.append(item)
            else:
                logger.warning("Bad value, not human -> gpt")
            continue
        offset = 1
        if item["conversations"][0]["from"] != "system":
            offset = 0
        if item["conversations"][-1]["from"] != "gpt":
            # Invalid, last response not from GPT?
            continue
        valid = True
        for idx in range(offset, len(item["conversations"])):
            expected = "human" if idx % 2 == offset else "gpt"
            if item["conversations"][idx]["from"] != expected:
                logger.warning(f'Unexpected role: {item["conversations"][idx]["from"]}')
                valid = False
                break
        if not valid:
            continue
        for idx in range(offset, len(item["conversations"]), 2):
            expanded.append(
                {
                    "id": str(uuid.uuid4()).replace("-", ""),
                    "source": item["source"],
                    "conversations": item["conversations"][0 : idx + 2],
                }
            )
    return expanded


def format_io(tokenizer, dataset):
    """Format the bagel dataset into input/output pairs."""

    # First, let's find all of the multi-turn instructions and expand those.
    # I do this because I don't want to train on the input strings, and it seems
    # that while just masking the inputs and training on outputs in a single
    # item works alright, optimizing for single responses might perform better.
    multi_turn = Dataset.from_list(
        expand_conversations(
            dataset.filter(
                lambda item: item.get("conversations")
                and len(item["conversations"]) > 3
            )
        )
    )
    single_turn = dataset.filter(
        lambda item: item.get("conversations") and len(item["conversations"]) <= 3
    )

    # DPO.
    dpo = dataset.filter(lambda item: item.get("prompt"))

    def _dpo_format(tokenizer, item, prompt_formatter):
        io_format = prompt_formatter(
            tokenizer,
            {
                "id": str(uuid.uuid4()),
                "source": item.get("source", "dpo"),
                "conversations": [
                    {
                        "from": "human",
                        "value": item["prompt"],
                    },
                    {
                        "from": "gpt",
                        "value": re.sub(r"(\s*\n){3,}", "\n\n", item["chosen"]),
                    },
                ],
            },
        )
        return {
            "source": item["source"],
            "prompt": io_format["input"],
            "chosen": re.sub(r"(\s*\n){3,}", "\n\n", item["chosen"]),
            "rejected": item["rejected"],
        }

    def _get_chunks(input_dataset):
        # Split into chunks of 75%, ensuring all instructions are covered, so we get exactly
        # 3 epochs worth of data (4 prompt formats * 75% of data = 3 x dataset size)
        chunk_size = (len(input_dataset) * 3) // 4
        quarter = len(input_dataset) - chunk_size
        ds = [item for item in input_dataset]
        return [
            Dataset.from_list(ds[0:chunk_size]),
            Dataset.from_list(ds[quarter : len(ds)]),
            Dataset.from_list(ds[0:quarter] + ds[quarter * 2 : len(ds)]),
            Dataset.from_list(ds[0 : quarter * 2] + ds[quarter * 3 : len(ds)]),
        ]

    formatters = [alpaca_io, vicuna_io, chatml_io, llama2_io]
    dpo_chunks = _get_chunks(dpo)
    dpo = concatenate_datasets(
        [
            dpo_chunks[idx].map(
                lambda item: _dpo_format(tokenizer, item, formatters[idx])
            )
            for idx in range(len(formatters))
        ]
    ).remove_columns(
        [
            col
            for col in dpo.column_names
            if col not in ("prompt", "chosen", "rejected", "source")
        ]
    )

    # Plain text.
    plain_text = dataset.filter(lambda item: item.get("text")).map(
        lambda item: {
            "id": item["id"],
            "input": "",
            "output": item["text"],
            "source": item["source"],
        }
    )
    plain_text = plain_text.remove_columns(
        [
            col
            for col in plain_text.column_names
            if col not in ("id", "input", "output", "source")
        ]
    ).map(
        lambda item: {
            "id": item["id"],
            "input": item["input"],
            "output": item["output"],
            "source": f"{item['source']}_plain_text",
        }
    )

    # Re-combine the expanded multi-turn with single-turn instructions.
    instructions = concatenate_datasets([multi_turn, single_turn]).shuffle(seed=42)

    # Split into chunks of 75%, ensuring all instructions are covered, so we get exactly
    # 3 epochs worth of data (4 prompt formats * 75% of data = 3 x dataset size)
    sft_chunks = _get_chunks(instructions)

    # Map to each of our prompt formats.
    instructions = concatenate_datasets(
        [
            sft_chunks[idx].map(lambda item: formatters[idx](tokenizer, item))
            for idx in range(len(formatters))
        ]
    ).remove_columns(
        [
            col
            for col in instructions.column_names
            if col not in ("id", "input", "output", "source")
        ]
    )

    return (
        concatenate_datasets([instructions, plain_text]).shuffle(seed=42),
        dpo.shuffle(seed=42),
    )


def get_or_create_datasets(tokenizer):
    """Get or create the SFT/DPO datasets."""
    if os.path.exists("bagel-input-output-v0.5.parquet"):
        return (
            Dataset.from_parquet("bagel-input-output-v0.5.parquet"),
            Dataset.from_parquet("bagel-dpo-v0.5.parquet"),
        )
    dataset = None
    if os.path.exists("bagel-clean-v0.5.parquet"):
        dataset = Dataset.from_parquet("bagel-clean-v0.5.parquet")
    else:
        if os.path.exists("bagel-raw-v0.5.parquet"):
            dataset = decontaminate(Dataset.from_parquet("bagel-raw-v0.5.parquet"))
            dataset.to_parquet("bagel-clean-v0.5.parquet")
        else:
            raw_dataset = load_datasets(tokenizer)
            raw_dataset.to_parquet("bagel-raw-v0.5.parquet")
            dataset = decontaminate(raw_dataset)
            dataset.to_parquet("bagel-clean-v0.5.parquet")

    # Split the raw dataset into SFT data and DPO data.
    sft, dpo = format_io(tokenizer, dataset)
    sft.to_parquet("bagel-input-output-v0.5.parquet")
    dpo.to_parquet("bagel-dpo-v0.5.parquet")
    return sft, dpo


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("mistralai/mistral-7b-v0.1")
    print(get_or_create_datasets(tokenizer))
