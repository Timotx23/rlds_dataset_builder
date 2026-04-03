from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub



class TestEpisode(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...).
        Might need to add wrist_image too if needed but for now optimized based on sample dataset
        """
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'arm_angles': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Observed robot arm joint angles.',
                        ),
                        'gripper': tfds.features.Scalar(
                            dtype=np.uint8,
                            doc='Observed gripper state.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action: [6x arm angles, 1x gripper command].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount factor for the step.',
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward for the step.',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.',
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.',
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if the last step is terminal.',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction. Use a dummy string if none is available.',
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.',
                    ),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/test_episode.hdf5'),
            #'val': self._generate_examples(path='data/val/episode_*.hdf5'), -> wasn't given a validation set so for now not needed
        }


    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        import h5py
        import numpy as np
        import os
        import glob

        def _parse_example(episode_path: str):
            # Load arrays from HDF5
            with h5py.File(episode_path, "r") as f:
                try:
                    obs_arm = f["observations"]["arm_angles"][:]
                    obs_grip = f["observations"]["gripper"][:]
                    act_arm = f["actions"]["arm_angles"][:]
                    act_grip = f["actions"]["gripper"][:]
                    rewards = f["rewards"][:]
                    discounts = f["discounts"][:]
                    is_first = f["is_first"][:]
                    is_last = f["is_last"][:]
                    is_terminal = f["is_terminal"][:]
                except KeyError as e:
                    raise KeyError(
                        f"Missing expected dataset/key in HDF5 file {episode_path}: {e}"
                    ) from e

            # Basic shape validation
            T = int(len(rewards))
            if not all(len(arr) == T for arr in [
                obs_grip, act_grip, discounts, is_first, is_last, is_terminal
            ]) or obs_arm.shape[0] != T or act_arm.shape[0] != T:
                raise ValueError(f"Inconsistent timestep lengths in file: {episode_path}")

            episode = []
            for i in range(T):
                action = np.concatenate(
                    [
                        act_arm[i].astype(np.float32),
                        np.array([act_grip[i]], dtype=np.float32),
                    ],
                    axis=0,
                )

                episode.append(
                    {
                        "observation": {
                            "arm_angles": obs_arm[i].astype(np.float32),
                            "gripper": np.uint8(obs_grip[i]),
                        },
                        "action": action,
                        "discount": np.float32(discounts[i]),
                        "reward": np.float32(rewards[i]),
                        "is_first": bool(is_first[i]),
                        "is_last": bool(is_last[i]),
                        "is_terminal": bool(is_terminal[i]),
                        "language_instruction": "move robot arm",
                    }
                )

            sample = {
                "steps": episode,
                "episode_metadata": {
                    "file_path": str(episode_path),
                },
            }

            # TFDS keys should be stable strings (not full paths).
            key = os.path.splitext(os.path.basename(episode_path))[0]
            return key, sample

        # Yield each episode exactly once
        for episode_path in sorted(glob.glob(path)):
            yield _parse_example(episode_path)

    

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

