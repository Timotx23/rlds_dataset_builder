import argparse
import importlib
import tqdm
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow as tf
import tensorflow_datasets as tfds
sys.path.append(os.path.abspath(".."))
from transform import transform


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='name of the dataset to visualize')
args = parser.parse_args()


TARGET_SPEC: dict["name": "Numpy value": dict["shape": set, "dtype": np.dtype, "ranges": set]] = {
    'observation': {
        'state': {'shape': (7,),
                  'dtype': np.float32,
                  'range': None}
    },
    'action': {'shape': (7,),
               'dtype': np.float32,
               'range': None},
    'discount': {'shape': (),
                 'dtype': np.float32,
                 'range': (0, 1)},
    'reward': {'shape': (),
               'dtype': np.float32,
               'range': None},
    'is_first': {'shape': (),
                 'dtype': np.bool_,
                 'range': None},
    'is_last': {'shape': (),
                'dtype': np.bool_,
                'range': None},
    'is_terminal': {'shape': (),
                    'dtype': np.bool_,
                    'range': None},
    'language_instruction': {'shape': (),
                             'dtype': str,
                             'range': None},
}


def check_elements(target: TARGET_SPEC, values:transform.transform_step):
    """Recursively checks that elements in `values` match the TARGET_SPEC."""
    for elem in target:
        if isinstance(values[elem], dict):
            check_elements(target[elem], values[elem])
        else:
            if target[elem]['shape']:
                if tuple(values[elem].shape) != target[elem]['shape']:
                    raise ValueError(
                        f"Shape of {elem} should be {target[elem]['shape']} but is {tuple(values[elem].shape)}")
            if not isinstance(values[elem], bytes) and values[elem].dtype != target[elem]['dtype']:
                raise ValueError(f"Dtype of {elem} should be {target[elem]['dtype']} but is {values[elem].dtype}")
            if target[elem]['range'] is not None:
                if isinstance(target[elem]['range'], list):
                    for vmin, vmax, val in zip(target[elem]['range'][0],
                                               target[elem]['range'][1],
                                               values[elem]):
                        if not (val >= vmin and val <= vmax):
                            raise ValueError(
                                f"{elem} is out of range. Should be in {target[elem]['range']} but is {values[elem]}.")
                else:
                    if not (np.all(values[elem] >= target[elem]['range'][0])
                            and np.all(values[elem] <= target[elem]['range'][1])):
                        raise ValueError(
                            f"{elem} is out of range. Should be in {target[elem]['range']} but is {values[elem]}.")


# create TF dataset
dataset_name = args.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, split='train')
ds = ds.shuffle(100)

for episode in tqdm.tqdm(ds.take(50)):
    steps = tfds.as_numpy(episode['steps'])
    for step in steps:
        transformed_step = transform.transform_step(step)
        check_elements(TARGET_SPEC, transformed_step)

print("Test passed! You're ready to submit!")