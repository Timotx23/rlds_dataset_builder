from pathlib import Path

BASE_DIR = Path("rlds_dataset_builder/dummy_dataset2")

for p in BASE_DIR.rglob("*"):

    print(p)