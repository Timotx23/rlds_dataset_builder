import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import tqdm


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_datasets as tfds

# Make imports relative to this file, not the current shell directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transform import transform


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="name of the dataset to test")
parser.add_argument("--split", default="train", help="dataset split to load")
parser.add_argument("--num_episodes", type=int, default=50, help="number of episodes to test")
args = parser.parse_args()


TARGET_SPEC = {
    "observation": {
        "state": {
            "shape": (7,),
            "dtype": np.float32,
            "range": None,
        }
    },
    "action": {
        "shape": (7,),
        "dtype": np.float32,
        "range": None,
    },
    "discount": {
        "shape": (),
        "dtype": np.float32,
        "range": (0, 1),
    },
    "reward": {
        "shape": (),
        "dtype": np.float32,
        "range": None,
    },
    "is_first": {
        "shape": (),
        "dtype": np.bool_,
        "range": None,
    },
    "is_last": {
        "shape": (),
        "dtype": np.bool_,
        "range": None,
    },
    "is_terminal": {
        "shape": (),
        "dtype": np.bool_,
        "range": None,
    },
    "language_instruction": {
        "shape": (),
        "dtype": str,
        "range": None,
    },
}


def _import_dataset_module(dataset_name: str) -> None:
    """
    Import the local TFDS builder module so the dataset gets registered.
    """
    candidate_modules = [
        dataset_name,
        f"{dataset_name}.{dataset_name}",
        "test_episode.test_episode_dataset_builder",
    ]

    tried = []

    for module_name in candidate_modules:
        try:
            importlib.import_module(module_name)
            return
        except ModuleNotFoundError as e:
            tried.append((module_name, str(e)))

    tried_str = "\n".join(f"  - {name}: {err}" for name, err in tried)
    raise ModuleNotFoundError(
        f"Could not import dataset module for '{dataset_name}'. Tried:\n{tried_str}"
    )


def _dtype_matches(value, expected_dtype) -> bool:
    if expected_dtype is str:
        return isinstance(value, (str, bytes, np.bytes_, np.str_))
    value_arr = np.asarray(value)
    return value_arr.dtype == np.dtype(expected_dtype)


def _shape_matches(value, expected_shape) -> bool:
    value_arr = np.asarray(value)
    return tuple(value_arr.shape) == expected_shape


def check_elements(target, values, prefix=""):
    """Recursively checks that elements in `values` match `target`."""
    for elem, spec in target.items():
        full_name = f"{prefix}.{elem}" if prefix else elem

        if elem not in values:
            raise KeyError(f"Missing key in transformed output: {full_name}")

        if isinstance(spec, dict) and "shape" not in spec:
            if not isinstance(values[elem], dict):
                raise TypeError(f"{full_name} should be a dict but is {type(values[elem])}")
            check_elements(spec, values[elem], prefix=full_name)
            continue

        value = values[elem]

        if spec["shape"] is not None:
            if not _shape_matches(value, spec["shape"]):
                raise ValueError(
                    f"Shape of {full_name} should be {spec['shape']} but is {np.asarray(value).shape}"
                )

        if not _dtype_matches(value, spec["dtype"]):
            actual_dtype = np.asarray(value).dtype if spec["dtype"] is not str else type(value)
            raise ValueError(
                f"Dtype of {full_name} should be {spec['dtype']} but is {actual_dtype}"
            )

        if spec["range"] is not None:
            value_arr = np.asarray(value)
            vmin, vmax = spec["range"]
            if not (np.all(value_arr >= vmin) and np.all(value_arr <= vmax)):
                raise ValueError(
                    f"{full_name} is out of range. Should be in {spec['range']} but is {value_arr}."
                )


dataset_name = args.dataset_name
print(f"Testing transformed data from dataset: {dataset_name}")

# Ensure the local custom dataset is registered with TFDS
_import_dataset_module(dataset_name)

ds = tfds.load(dataset_name, split=args.split)
ds = ds.shuffle(100)

num_checked = 0

for episode in tqdm.tqdm(ds.take(args.num_episodes), desc="Episodes"):
    steps = tfds.as_numpy(episode["steps"])

    for step in steps:
        transformed_step = transform.transform_step(step)
        check_elements(TARGET_SPEC, transformed_step)

    num_checked += 1

print(f"Test passed! Checked {num_checked} episode(s). You're ready to submit!")