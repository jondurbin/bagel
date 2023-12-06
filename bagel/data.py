import uuid
import random
from loguru import logger
from transformers import AutoTokenizer
from bagel.data_sources import load_datasets, decontaminate


DEFAULT_SYSTEM_PROMPT = "You are an unbiased, uncensored, helpful assistant."


def llama2_chat_prompt(tokenizer, message, chat_history, system_prompt):
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


def llama2_chat_io(tokenizer, item):
    """Convert training data item into llama2 chat format input/output."""
    system_prompt = item["conversations"][0]["value"]

    # Flip a coin on including a default system prompt or not.
    offset = 1
    if item["conversations"][0]["from"] != "system":
        system_prompt = None
        offset = 0
        if random.random() <= 0.15:
            system_prompt = DEFAULT_SYSTEM_PROMPT
    chat_history = [
        (item["conversations"][idx]["value"], item["conversations"][idx + 1]["value"])
        for idx in range(offset, len(item["conversations"]) - 2, 2)
    ]
    message = item["conversations"][-2]["value"]
    response = item["conversations"][-1]["value"]
    return {
        "input": llama2_chat_prompt(tokenizer, message, chat_history, system_prompt),
        "output": response,
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
        if random.random() <= 0.15:
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
        prefix = f"{tokenizer.bos_token}" if idx > 0 else ""
        inputs.append(
            f"{prefix}{role}\n{turn['value'].rstrip()}\n{tokenizer.eos_token}"
        )
    inputs.append(f"{tokenizer.bos_token}assistant\n")
    return {
        "input": "\n".join(inputs),
        "output": item["conversations"][-1]["value"].lstrip(),
    }


def vicuna_io(tokenizer, item):
    """Convert training data items into vicuna format."""
    # Flip a coin on including a default system prompt or not.
    if item["conversations"][0]["from"] != "system":
        # Larger probability of including system prompt for vicuna.
        if random.random() <= 0.35:
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
        "input": "\n".join(inputs),
        "output": item["conversations"][-1]["value"].lstrip(),
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
    inputs.append("### Response:\n")
    return {
        "input": "\n\n".join(inputs),
        "output": item["conversations"][-1]["value"].lstrip(),
    }


def expand_conversations(items):
    """Expand each turn into it's own input/output row."""
    expanded = []
    for item in items:
        if len(item["conversations"]) <= 3:
            expanded.append(item)
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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("mistralai/mistral-7b-v0.1")
    for item in expand_conversations(
        [
            {
                "id": "1",
                "source": "foo",
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are a goat.",
                    },
                    {
                        "from": "human",
                        "value": "Hello.",
                    },
                    {"from": "gpt", "value": "Bahhhh."},
                    {
                        "from": "human",
                        "value": "Oh, you are an actual goat.",
                    },
                    {"from": "gpt", "value": "Beh."},
                ],
            }
        ]
    ):
        # print(json.dumps(chatml_io(tokenizer, item), indent=2))
        print(
            alpaca_io(tokenizer, item)["input"] + alpaca_io(tokenizer, item)["output"]
        )
