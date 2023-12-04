from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from bagel.datasets.util import as_conversation, get_uid

CONFIDENCE = 3


def load_data(known_uids=set([])):
    """Cinematika v0.1 dataset."""
    data = []
    for segment in (
        "prompt_to_character_card",
        "rp_to_character_card",
        "scene_summary",
        "scene_enhancement",
    ):
        logger.info(f"Loading Cinematika v0.1 dataset -- {segment}...")
        dataset = load_dataset(
            "jondurbin/cinematika-v0.1", data_files=[f"{segment}.parquet"]
        )
        for item in tqdm(dataset["train"]):
            as_conv = as_conversation(item["input"], item["output"])
            known_uids.add(as_conv["id"])
            data.append(as_conv)
    logger.info("Loading Cinematika v0.1 dataset -- scene_by_scene...")
    dataset = load_dataset(
        "jondurbin/cinematika-v0.1", data_files=["scene_by_scene.parquet"]
    )
    for item in tqdm(dataset["train"]):
        uid = get_uid(item["scene_by_scene"])
        if uid in known_uids:
            continue
        known_uids.add(uid)
        data.append({"id": uid, "text": item["scene_by_scene"]})
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
