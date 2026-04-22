from pathlib import Path
from typing import Any, Iterator, Tuple
import cv2
import numpy as np
import tensorflow_datasets as tfds
import h5py
import random

class Cobot280PiDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for HDF5 robot episodes."""

    VERSION = tfds.core.Version("0.3.0")
    RELEASE_NOTES = {
        "0.3.0": "22 april scuffed uni monitor table recording",
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
                            shape=(224, 224, 3), 
                            dtype=np.uint8,
                        ),
                        "cam_wrist": tfds.features.Image(
                            shape=(224, 224, 3), 
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
                    "language_instruction": tfds.features.Text(
                        doc="Language instruction.",
                    ),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        base_dir = Path(__file__).resolve().parent / "Cobot280PiDataset"

        if not base_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

        # Collect all episode files (same logic as in _generate_examples)
        episode_paths = sorted(
            p for p in base_dir.rglob("*")
            if p.is_file() and p.suffix in {".h5", ".hdf5"}
        )

        if not episode_paths:
            raise FileNotFoundError(f"No HDF5 files found under: {base_dir}")

        rng = random.Random(42)
        rng.shuffle(episode_paths)

        split_ratio = 0.9
        split_idx = int(len(episode_paths) * split_ratio)

        train_paths = episode_paths[:split_idx]
        val_paths = episode_paths[split_idx:]

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"episode_paths": train_paths},
            ),
            tfds.core.SplitGenerator(
                name="val",
                gen_kwargs={"episode_paths": val_paths},
            ),
        ]

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:


        target_image_shape = (224, 224, 3)

        for episode_index, episode_path in enumerate(episode_paths):

            print(f"Episode: {episode_path}")
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

                target_h, target_w = 224, 224
                target_image_shape = (target_h, target_w, 3)

                # --- Process External Camera ---
                if "cam_external" in f["observations"]:
                    raw_cam_external = f["observations"]["cam_external"][:]
                    
                    # raw_cam_external shape is expected to be (T, H, W, C)
                    current_h, current_w = raw_cam_external.shape[1], raw_cam_external.shape[2]
                    
                    if (current_h, current_w) == (target_h, target_w):
                        # Skip resizing, just copy the array
                        obs_cam_external = raw_cam_external
                    else:
                        # Pre-allocate and resize
                        obs_cam_external = np.empty((T, *target_image_shape), dtype=np.uint8)
                        for i in range(T):
                            obs_cam_external[i] = cv2.resize(
                                raw_cam_external[i], (target_w, target_h), interpolation=cv2.INTER_AREA
                            )
                            
                    # IMPORTANT: If your HDF5 images are saved in BGR format, uncomment the loop below:
                    # for i in range(T):
                    #     obs_cam_external[i] = cv2.cvtColor(obs_cam_external[i], cv2.COLOR_BGR2RGB)
                else:
                    obs_cam_external = np.zeros((T, *target_image_shape), dtype=np.uint8)

                # --- Process Wrist Camera ---
                if "cam_wrist" in f["observations"]:
                    raw_cam_wrist = f["observations"]["cam_wrist"][:]
                    
                    current_h, current_w = raw_cam_wrist.shape[1], raw_cam_wrist.shape[2]
                    
                    if (current_h, current_w) == (target_h, target_w):
                        obs_cam_wrist = raw_cam_wrist
                    else:
                        obs_cam_wrist = np.empty((T, *target_image_shape), dtype=np.uint8)
                        for i in range(T):
                            obs_cam_wrist[i] = cv2.resize(
                                raw_cam_wrist[i], (target_w, target_h), interpolation=cv2.INTER_AREA
                            )
                            
                    # IMPORTANT: If your HDF5 images are saved in BGR format, uncomment the loop below:
                    # for i in range(T):
                    #     obs_cam_wrist[i] = cv2.cvtColor(obs_cam_wrist[i], cv2.COLOR_BGR2RGB)
                else:
                    obs_cam_wrist = np.zeros((T, *target_image_shape), dtype=np.uint8)

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
