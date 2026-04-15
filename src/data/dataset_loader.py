#src/data/dataset_loader.py
"""
Dataset Loader.

Loads HuggingFace datasets with robust split handling.
"""

from datasets import load_dataset


class DatasetLoader:
    """Loads datasets safely with fallback split handling."""

    def __init__(self, config):
        self.config = config

    def load(self):
        """Load dataset and return correct split."""

        ds = load_dataset(
            self.config.hf_dataset_id,
            self.config.hf_subset,
        )

        # Smart split resolution
        if self.config.test_split in ds:
            split = self.config.test_split
        elif "validation" in ds:
            split = "validation"
        elif "test" in ds:
            split = "test"
        elif "train" in ds:
            split = "train"
        else:
            raise ValueError(f"No usable split in {list(ds.keys())}")

        data = ds[split]

        print(f"Loaded {self.config.hf_dataset_id} [{split}] → {len(data)} samples")

        return data