import re
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, concatenate_datasets, Dataset
from .util import as_conversation, get_uid

PRIORITY = 3


def load_data(known_uids=set([]), **_):
    """Cinematika v0.1 dataset."""
    data = []

    def _cleanup_setting(text):
        return re.sub(r"\[scene\]\s*", "", text)

    # Support creating character cards from prompts.
    segment = "prompt_to_character_card"
    logger.info(f"Loading Cinematika v0.1 dataset -- {segment}...")
    dataset = load_dataset(
        "jondurbin/cinematika-v0.1", data_files=[f"{segment}.parquet"]
    )
    for item in tqdm(dataset["train"]):
        as_conv = as_conversation(
            _cleanup_setting(item["input"]), _cleanup_setting(item["output"])
        )
        known_uids.add(as_conv["id"])
        as_conv["source"] = f"cinematika_{segment}"
        data.append(as_conv)

    # Roleplay (scene-by-scene)
    logger.info("Loading Cinematika v0.1 dataset -- scene_by_scene...")
    dataset = load_dataset(
        "jondurbin/cinematika-v0.1", data_files=["scene_by_scene.parquet"]
    )
    for item in tqdm(dataset["train"]):
        uid = get_uid(item["scene_by_scene"])
        if uid in known_uids:
            continue
        known_uids.add(uid)
        data.append(
            {
                "source": "cinematika_scenes",
                "id": uid,
                "text": re.sub(r"\[scene\]\s*", "", item["scene_by_scene"]),
            }
        )

    # Memories (for long-term memory in chat).
    logger.info("Loading Cinematika v0.1 dataset -- memories...")
    memories = load_dataset(
        "jondurbin/cinematika-v0.1", data_files=["memories.parquet"], split="train"
    ).map(
        lambda item: {
            "id": item["id"],
            "source": "cinematika_memories",
            "conversations": item["conversations"],
            "text": None,
        }
    )
    return concatenate_datasets([Dataset.from_list(data), memories])


if __name__ == "__main__":
    print(load_data())
