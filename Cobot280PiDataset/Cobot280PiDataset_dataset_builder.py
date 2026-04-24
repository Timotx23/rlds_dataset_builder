from pathlib import Path
from typing import Any, Iterator, Tuple
from collections import defaultdict
import cv2
import numpy as np
import tensorflow_datasets as tfds
import h5py
import random

class Cobot280PiDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for HDF5 robot episodes."""

    VERSION = tfds.core.Version("0.4.1")
    RELEASE_NOTES = {
        "0.3.1": "22 april scuffed uni monitor table recording - PROPER CENTER CROP",
        "0.3.2": "22 april rewrite task",
        "0.4.0": "24 april library left and right",
        "0.4.1": "24 april library left and right color fix"
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

        # Collect all episode files
        episode_paths = sorted(
            p for p in base_dir.rglob("*")
            if p.is_file() and p.suffix in {".h5", ".hdf5"}
        )

        if not episode_paths:
            raise FileNotFoundError(f"No HDF5 files found under: {base_dir}")

        # 1. Group files by their task attribute
        task_to_paths = defaultdict(list)
        for path in episode_paths:
            try:
                with h5py.File(path, "r") as f:
                    # Fallback to 'unknown_task' just in case an episode is missing the attribute
                    task = f.attrs.get("task", "unknown_task") 
                    task_to_paths[task].append(path)
            except Exception as e:
                raise IOError(f"Failed to read task attribute from {path}: {e}")

        train_paths = []
        val_paths = []
        rng = random.Random(42)
        split_ratio = 0.9

        # 2. Perform the 90/10 split within each task group
        for task, paths in task_to_paths.items():
            rng.shuffle(paths)
            split_idx = int(len(paths) * split_ratio)
            
            # Edge case handling: if a task has very few files (e.g., 1 file), put it in train
            if split_idx == 0 and len(paths) > 0:
                train_paths.extend(paths)
            else:
                train_paths.extend(paths[:split_idx])
                val_paths.extend(paths[split_idx:])

        # 3. Shuffle the final lists so batches aren't strictly grouped by task sequence
        rng.shuffle(train_paths)
        rng.shuffle(val_paths)

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

                # --- Process External Camera ---
                if "cam_external" in f["observations"]:
                    raw_cam_external = f["observations"]["cam_external"][:]
 
                    # Call the function (returns the perfectly cropped 224x224 batch)
                    obs_cam_external = process_camera_batch(
                        raw_cam_data=raw_cam_external, 
                        target_h=224, 
                        target_w=224,
                        convert_bgr_to_rgb=False # Change to True if your HDF5 saved BGR images
                    )                   

                # --- Process Wrist Camera ---
                if "cam_wrist" in f["observations"]:
                    raw_cam_wrist = f["observations"]["cam_wrist"][:]
                    
                    obs_cam_wrist = process_camera_batch(
                        raw_cam_data=raw_cam_wrist, 
                        target_h=224, 
                        target_w=224,
                        convert_bgr_to_rgb=False # Change to True if your HDF5 saved BGR images
                    )                   
                    

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



def process_camera_batch(raw_cam_data, target_h=224, target_w=224, convert_bgr_to_rgb=False):
    """
    Resizes and center-crops a batch of images to the target resolution.
    
    Args:
        raw_cam_data (np.ndarray): The raw camera data of shape (T, H, W, C).
        target_h (int): Target height (default 224).
        target_w (int): Target width (default 224).
        convert_bgr_to_rgb (bool): Set to True if images are BGR and need to be RGB.
        
    Returns:
        np.ndarray: Processed image batch of shape (T, target_h, target_w, C) in uint8.
    """
    T = raw_cam_data.shape[0]
    current_h, current_w = raw_cam_data.shape[1], raw_cam_data.shape[2]
    channels = raw_cam_data.shape[3]
    
    # Calculate scale to ensure the resized image is large enough to crop from.
    # We take the max of the height/width ratios so the shortest edge matches the target.
    scale = max(target_h / current_h, target_w / current_w)
    new_h, new_w = int(current_h * scale), int(current_w * scale)
    
    # Calculate center crop coordinates
    y_start = (new_h - target_h) // 2
    x_start = (new_w - target_w) // 2
    
    # Pre-allocate array for speed
    processed_data = np.empty((T, target_h, target_w, channels), dtype=np.uint8)
    
    for i in range(T):
        img = raw_cam_data[i]
        
        # 1. Optional Color Conversion
        if convert_bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # 2. Skip resizing if the image is already the exact right shape
        if (current_h, current_w) == (target_h, target_w):
            processed_data[i] = img
            continue
            
        # 3. Resize keeping aspect ratio
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 4. Apply center crop and store
        processed_data[i] = resized[y_start:y_start+target_h, x_start:x_start+target_w]
        
    return processed_data
