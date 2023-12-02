import ai2_arc
import airoboros
import apps
import belebele
import boolq
import cinematika
import drop
import gsm8k
import mmlu
import natural_instructions
import openbookqa
import piqa
import python_alpaca
import slimorca
import spider
import squad_v2
import synthia
import winogrande


def load_datasets():
    """Load all of the datasets."""
    things = {key: val for key, val in globals().items()}
    from datasets import concatenate_datasets
    from types import ModuleType

    all_datasets = []
    for key, val in things.items():
        if key.startswith("__"):
            continue
        if isinstance(val, ModuleType) and hasattr(val, "load_data"):
            dataset = val.load_data()
            if "text" not in dataset.column_names:
                dataset = dataset.add_column("text", [None] * len(dataset))
            if "source" not in dataset.column_names:
                dataset = dataset.add_column("source", [key] * len(dataset))
            dataset = dataset.remove_columns(
                [
                    col
                    for col in dataset.column_names
                    if col not in ("id", "source", "text", "conversations")
                ]
            )
            all_datasets.append(dataset)
    return concatenate_datasets(all_datasets)


if __name__ == "__main__":
    print(load_datasets())
