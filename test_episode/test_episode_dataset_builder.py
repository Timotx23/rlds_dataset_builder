from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class TestEpisode(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for HDF5 robot episodes."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "arm_angles": tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc="Observed robot arm joint angles.",
                        ),
                        "gripper": tfds.features.Scalar(
                            dtype=np.uint8,
                            doc="Observed gripper state.",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc="Robot action: [6x arm angles, 1x gripper command].",
                    ),
                    "discount": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Discount factor for the step.",
                    ),
                    "reward": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Reward for the step.",
                    ),
                    "is_first": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on first step of the episode.",
                    ),
                    "is_last": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on last step of the episode.",
                    ),
                    "is_terminal": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True if the last step is terminal.",
                    ),
                    "language_instruction": tfds.features.Text(
                        doc="Language instruction.",
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_index": tfds.features.Scalar(
                        dtype=np.int64,
                        doc="Stable numeric index of the episode within the sorted file list.",
                    ),
                    "file_path": tfds.features.Text(
                        doc="Path to the original data file.",
                    ),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(path="data/train/*.hdf5"),
        }

    def _generate_examples(self, path: str) -> Iterator[Tuple[str, Any]]:
        import h5py

        episode_paths = sorted(glob.glob(path))
        if not episode_paths:
            raise FileNotFoundError(f"No HDF5 files matched pattern: {path}")

        for episode_index, episode_path in enumerate(episode_paths):
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

            if obs_arm.ndim != 2 or obs_arm.shape[1] != 6:
                raise ValueError(
                    f"Expected observations/arm_angles shape (T, 6) in {episode_path}, got {obs_arm.shape}"
                )

            if act_arm.ndim != 2 or act_arm.shape[1] != 6:
                raise ValueError(
                    f"Expected actions/arm_angles shape (T, 6) in {episode_path}, got {act_arm.shape}"
                )

            T = len(rewards)

            arrays_to_check = [
                obs_grip, act_grip, discounts, is_first, is_last, is_terminal
            ]
            if not all(len(arr) == T for arr in arrays_to_check):
                raise ValueError(f"Inconsistent timestep lengths in file: {episode_path}")

            if obs_arm.shape[0] != T or act_arm.shape[0] != T:
                raise ValueError(f"Inconsistent timestep lengths in file: {episode_path}")

            episode = []
            for i in range(T):
                action = np.concatenate([
                    act_arm[i].astype(np.float32),
                    np.array([act_grip[i]], dtype=np.float32),
                ])

                episode.append({
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
                })

            sample = {
                "steps": episode,
                "episode_metadata": {
                    "episode_index": np.int64(episode_index),
                    "file_path": str(episode_path),
                },
            }

            key = f"episode_{episode_index:06d}"
            yield key, sample