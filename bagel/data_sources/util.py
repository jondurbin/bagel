import re
import uuid
from loguru import logger

REFUSALS = [
    re.compile(regexp)
    for regexp in [
        "(against|violate) my programming",
        "openai",
        "language model",
        "large language",
        "as an? (ai|generative language|gpt|bot)",
        "illegal and dangerous",
        "important to note",
        "i do(n't| not) (possess|have|exhibit) (personal|consciousness|subjective)",
        "personal (feelings|thoughts|emotions|desires|experiences|goals|objective|belief)",
        "(can('t| ?not)|w(on't|will not)|unable.?) (\\w+\\s)+(with (that|your)|your \\w+|provide)",
        "my limitations",
        "the limitations of my",
        "my abilities",
        "violates my",
        "i (can('t| ?not)|w(on't|will not)|am (not |un)able.?).{0,30}(you are|you're|your )",
        "please note that",
    ]
]


def get_uid(input_text):
    """Helper to get a uuid5 from input text."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, input_text))


def has_refusal(response):
    """Check if a response has a refusal - not perfect, but good enough..."""
    for regexp in REFUSALS:
        if m := regexp.search(response, re.I):
            logger.warning(f"Refusal? {m.group()}")
            return True
    return False


def as_conversation(instruction, response, system=None):
    """Helper to convert instruction/response pair to conversation format."""
    result = {
        "id": get_uid(instruction),
        "source": None,
        "text": None,
        "chosen": None,
        "rejected": None,
        "prommpt": None,
        "conversations": [
            {
                "from": "human",
                "value": instruction,
            },
            {
                "from": "gpt",
                "value": response,
            },
        ],
    }
    if system:
        result["conversations"] = [{"from": "system", "value": system}] + result[
            "conversations"
        ]
    return result


def map_conv_format(item):
    """Map role to from, and content to value."""
    conv = [
        {
            "from": turn.get("role", turn.get("from")),
            "value": turn.get("content", turn.get("value")),
        }
        for turn in item["conversation"]
    ]
    return {
        **{k: v for k, v in item.items() if k not in ("conversation", "conversations")},
        **{"conversations": conv},
    }
