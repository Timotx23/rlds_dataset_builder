from pathlib import Path
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow_datasets as tfds


class Cobot280PiDataset(tfds.core.GeneratorBasedBuilder):
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
                        "cam_external": tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                        ),
                        "cam_wrist": tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
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
        base_dir = Path(__file__).resolve().parent / "Cobot280PiDataset"

        if not base_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"path": base_dir},
            ),
        ]

    def _generate_examples(self, path: Path) -> Iterator[Tuple[str, Any]]:
        import h5py

        base_dir = Path(path)

        episode_paths = sorted(
            p for p in base_dir.rglob("*")
            if p.is_file() and p.suffix in {".h5", ".hdf5"}
        )

        if not episode_paths:
            raise FileNotFoundError(f"No HDF5 files found under: {base_dir}")

        image_shape = (480, 640, 3)

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
                    language_instruction: str = f.attrs["task"]
                except KeyError as e:
                    raise KeyError(
                        f"Missing expected dataset/key in HDF5 file {episode_path}: {e}"
                    ) from e

                T = len(rewards)

                if obs_arm.ndim != 2 or obs_arm.shape != (T, 6):
                    raise ValueError(
                        f"Expected observations/arm_angles shape ({T}, 6) in {episode_path}, got {obs_arm.shape}"
                    )

                if act_arm.ndim != 2 or act_arm.shape != (T, 6):
                    raise ValueError(
                        f"Expected actions/arm_angles shape ({T}, 6) in {episode_path}, got {act_arm.shape}"
                    )

                arrays_to_check = [
                    obs_grip,
                    act_grip,
                    discounts,
                    is_first,
                    is_last,
                    is_terminal,
                ]
                if not all(len(arr) == T for arr in arrays_to_check):
                    raise ValueError(f"Inconsistent timestep lengths in file: {episode_path}")

                if "cam_external" in f["observations"]:
                    obs_cam_external = f["observations"]["cam_external"][:]
                    if obs_cam_external.shape != (T, *image_shape):
                        raise ValueError(
                            f"Expected observations/cam_external shape ({T}, {image_shape[0]}, {image_shape[1]}, {image_shape[2]}) "
                            f"in {episode_path}, got {obs_cam_external.shape}"
                        )
                    obs_cam_external = obs_cam_external.astype(np.uint8)
                else:
                    obs_cam_external = np.zeros((T, *image_shape), dtype=np.uint8)

                if "cam_wrist" in f["observations"]:
                    obs_cam_wrist = f["observations"]["cam_wrist"][:]
                    if obs_cam_wrist.shape != (T, *image_shape):
                        raise ValueError(
                            f"Expected observations/cam_wrist shape ({T}, {image_shape[0]}, {image_shape[1]}, {image_shape[2]}) "
                            f"in {episode_path}, got {obs_cam_wrist.shape}"
                        )
                    obs_cam_wrist = obs_cam_wrist.astype(np.uint8)
                else:
                    obs_cam_wrist = np.zeros((T, *image_shape), dtype=np.uint8)

            episode = []
            for i in range(T):
                action = np.concatenate([
                    act_arm[i].astype(np.float32),
                    np.array([act_grip[i]], dtype=np.float32),
                ])

                step = {
                    "observation": {
                        "arm_angles": obs_arm[i].astype(np.float32),
                        "gripper": np.uint8(obs_grip[i]),
                        "cam_wrist": obs_cam_wrist[i],
                        "cam_external": obs_cam_external[i],
                    },
                    "action": action,
                    "discount": np.float32(discounts[i]),
                    "reward": np.float32(rewards[i]),
                    "is_first": bool(is_first[i]),
                    "is_last": bool(is_last[i]),
                    "is_terminal": bool(is_terminal[i]),
                    "language_instruction": language_instruction,
                }
                episode.append(step)

            sample = {
                "steps": episode,
                "episode_metadata": {
                    "episode_index": np.int64(episode_index),
                    "file_path": str(episode_path),
                    "language_instruction": language_instruction
                },
            }

            key = f"episode_{episode_index:06d}"
            yield key, sample