
# filepath: /Users/timohildenbrand/dsa learning/Dl/rlds_dataset_builder/video_read.py
"""
Utility to read/inspect frames from a finished TFDS RLDS dataset.

Default behavior:
  - If DEFAULT_PREPARED_PATH is set to a TFDS-prepared dataset dir (recommended),
    you can run:  python video_read.py
  - Otherwise, pass --dataset/--data_dir for registered TFDS datasets.

Notes:
- RLDS datasets are typically structured as episodes -> steps.
- This script tries common frame keys such as:
  observation.image, observation.rgb, observation.camera_* , image, rgb
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import tensorflow_datasets as tfds
except Exception as e:  # pragma: no cover
    raise ImportError("tensorflow_datasets is required. Install with: pip install tensorflow-datasets") from e

try:
    import tensorflow as tf  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("tensorflow is required. Install with: pip install tensorflow") from e


# =========================
# Defaults (edit these)
# =========================

# Point this at either:
#   - /Users/.../tensorflow_datasets/cobot280_pi_dataset
#   - /Users/.../tensorflow_datasets/cobot280_pi_dataset/1.0.0
DEFAULT_PREPARED_PATH = "/Users/timohildenbrand/tensorflow_datasets/cobot280_pi_dataset/1.0.0"

DEFAULT_SPLIT = "train"
DEFAULT_EPISODE = 0
DEFAULT_FPS = 20.0
DEFAULT_MAX_FRAMES: Optional[int] = None
DEFAULT_FRAME_KEY: Optional[str] = None
DEFAULT_SAVE_DIR: Optional[str] = None  # e.g. "./frames" to save instead of display
DEFAULT_SUMMARY = False


# =========================
# TFDS prepared-dir loader
# =========================

def _to_numpy(x: Any) -> Any:
    """Convert TF/Tensor/np-like objects to numpy when possible."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _is_tfds_dataset_dir(path: str) -> bool:
    """True if directory contains TFDS metadata for a prepared dataset version."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json"))


def _find_version_dir(dataset_root_or_version: str) -> str:
    """
    Accepts either:
      - /.../cobot280_pi_dataset            (root containing versions like 1.0.0/)
      - /.../cobot280_pi_dataset/1.0.0      (a specific version dir)
    Returns the resolved version dir.
    """
    if _is_tfds_dataset_dir(dataset_root_or_version):
        return dataset_root_or_version

    if not os.path.isdir(dataset_root_or_version):
        raise FileNotFoundError(f"Dataset path not found: {dataset_root_or_version}")

    version_dirs: List[str] = []
    for name in os.listdir(dataset_root_or_version):
        p = os.path.join(dataset_root_or_version, name)
        if _is_tfds_dataset_dir(p):
            version_dirs.append(p)

    if not version_dirs:
        raise FileNotFoundError(
            "Could not find a TFDS version directory containing dataset_info.json under: "
            f"{dataset_root_or_version}"
        )

    version_dirs.sort()
    return version_dirs[-1]


def _infer_dataset_name_from_prepared_dir(prepared_path: str) -> str:
    version_dir = _find_version_dir(prepared_path)
    builder = tfds.builder_from_directory(version_dir)
    return builder.info.name


def _load_rlds_from_prepared_dir(prepared_path: str, split: str) -> Any:
    version_dir = _find_version_dir(prepared_path)
    builder = tfds.builder_from_directory(version_dir)
    return builder.as_dataset(split=split)


# =========================
# Frame extraction/display
# =========================

def _find_candidate_frames(step: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    """
    Find candidate image frames in a step dict.
    Returns list of (key_path, frame_array).
    """
    candidates: List[Tuple[str, np.ndarray]] = []

    def visit(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{prefix}.{k}" if prefix else str(k))
            return

        arr = _to_numpy(obj)
        if isinstance(arr, np.ndarray):
            if arr.ndim == 3 and arr.shape[0] > 8 and arr.shape[1] > 8 and arr.shape[2] in (1, 3, 4):
                candidates.append((prefix, arr))
            elif arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
                candidates.append((prefix, arr[0]))

    visit(step)
    return candidates


def _pick_frame_key(
    candidates: List[Tuple[str, np.ndarray]],
    preferred: Optional[str],
) -> Optional[Tuple[str, np.ndarray]]:
    if not candidates:
        return None
    if preferred:
        for k, v in candidates:
            if k == preferred or k.endswith("." + preferred):
                return (k, v)

    preferred_suffixes = [
        "observation.image",
        "observation.rgb",
        "observation.camera",
        "observation.front_camera",
        "image",
        "rgb",
    ]
    for suffix in preferred_suffixes:
        for k, v in candidates:
            if k == suffix or k.endswith("." + suffix):
                return (k, v)

    return candidates[0]


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    f = frame
    if f.dtype == np.uint8:
        return f
    if np.issubdtype(f.dtype, np.floating):
        f = np.clip(f, 0.0, 1.0)
        return (f * 255.0).astype(np.uint8)
    return np.clip(f, 0, 255).astype(np.uint8)


def _display_frames(frames: Iterable[np.ndarray], fps: float, window: str = "RLDS Frames") -> None:
    """
    Display frames using OpenCV if available, else matplotlib.
    """
    delay_ms = max(1, int(1000.0 / max(1e-6, fps)))

    try:
        import cv2  # type: ignore

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        for frame in frames:
            f = _ensure_uint8(frame)
            if f.ndim == 3 and f.shape[2] in (3, 4):
                if f.shape[2] == 4:
                    f = f[:, :, :3]
                f = f[:, :, ::-1]  # RGB->BGR
            cv2.imshow(window, f)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        return
    except Exception:
        pass

    import matplotlib.pyplot as plt  # type: ignore

    plt.ion()
    fig, ax = plt.subplots()
    im = None
    for frame in frames:
        f = _ensure_uint8(frame)
        if f.ndim == 3 and f.shape[2] == 4:
            f = f[:, :, :3]
        if im is None:
            im = ax.imshow(f)
            ax.set_title(window + " (close window to stop)")
            ax.axis("off")
        else:
            im.set_data(f)
        plt.pause(delay_ms / 1000.0)
        if not plt.fignum_exists(fig.number):
            break
    plt.ioff()


def _save_frames(frames: Iterable[np.ndarray], save_dir: str, prefix: str = "frame") -> int:
    os.makedirs(save_dir, exist_ok=True)
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow is required for --save_dir. Install with: pip install pillow") from e

    count = 0
    for i, frame in enumerate(frames):
        f = _ensure_uint8(frame)
        if f.ndim == 3 and f.shape[2] == 4:
            f = f[:, :, :3]
        Image.fromarray(f).save(os.path.join(save_dir, f"{prefix}_{i:06d}.png"))
        count += 1
    return count


def iter_episode_frames(
    ds: Any,
    episode_index: int,
    preferred_key: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Iterable[np.ndarray]:
    """Yield frames for a single episode from a TFDS RLDS dataset."""
    for epi_i, episode in enumerate(tfds.as_numpy(ds)):
        if epi_i != episode_index:
            continue

        steps = episode.get("steps", None)
        if steps is None:
            raise KeyError("Episode does not contain 'steps'. Keys: " + ", ".join(sorted(episode.keys())))

        emitted = 0
        for step in steps:
            candidates = _find_candidate_frames(step)
            picked = _pick_frame_key(candidates, preferred_key)
            if picked is None:
                continue
            _, frame = picked
            yield frame
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                return
        return

    raise IndexError(f"Episode index {episode_index} out of range.")


def _print_episode_summary(ds: Any, episode_index: int) -> None:
    for epi_i, episode in enumerate(tfds.as_numpy(ds)):
        if epi_i != episode_index:
            continue

        steps = episode.get("steps", None)
        if steps is None:
            print("Episode keys:", sorted(list(episode.keys())))
            return

        first_step = None
        for s in steps:
            first_step = s
            break

        print(f"Episode {episode_index} loaded.")
        if first_step is not None and isinstance(first_step, dict):
            print("First step keys:", sorted(list(first_step.keys())))
            candidates = _find_candidate_frames(first_step)
            if candidates:
                print("Detected image candidates (first step):")
                for k, arr in candidates:
                    print(f"  - {k}: shape={arr.shape} dtype={arr.dtype}")
            else:
                print("No image candidates detected in first step.")
        return

    raise IndexError(f"Episode index {episode_index} out of range.")


# =========================
# Entry point
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Read/display frames from a TFDS RLDS dataset.")

    # Everything optional; running with no args uses defaults above.
    parser.add_argument("--dataset", required=False, default=None, help="TFDS dataset name (optional; inferred if using prepared path).")
    parser.add_argument("--data_dir", default=None, help="TFDS data_dir where the dataset is stored.")
    parser.add_argument("--split", default=None, help=f"Split to load (default: {DEFAULT_SPLIT}).")
    parser.add_argument("--episode", type=int, default=None, help=f"Episode index to view (default: {DEFAULT_EPISODE}).")
    parser.add_argument("--fps", type=float, default=None, help=f"Playback FPS (default: {DEFAULT_FPS}).")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to read from the episode (default: no limit).")
    parser.add_argument("--frame_key", type=str, default=None, help="Optional explicit key path to frame.")
    parser.add_argument("--save_dir", type=str, default=None, help="If set, saves frames as PNGs instead of displaying.")
    parser.add_argument("--summary", action="store_true", help="Print episode/key summary and exit.")
    parser.add_argument("--prepared_path", type=str, default=None, help="Path to TFDS-prepared dataset dir (optional).")

    args = parser.parse_args()

    split = args.split or DEFAULT_SPLIT
    episode = args.episode if args.episode is not None else DEFAULT_EPISODE
    fps = args.fps if args.fps is not None else DEFAULT_FPS
    max_frames = args.max_frames if args.max_frames is not None else DEFAULT_MAX_FRAMES
    frame_key = args.frame_key if args.frame_key is not None else DEFAULT_FRAME_KEY
    save_dir = args.save_dir if args.save_dir is not None else DEFAULT_SAVE_DIR
    summary = args.summary or DEFAULT_SUMMARY

    prepared_path = args.prepared_path or (DEFAULT_PREPARED_PATH if DEFAULT_PREPARED_PATH else None)

    dataset_name = args.dataset
    if dataset_name is None:
        if prepared_path:
            dataset_name = _infer_dataset_name_from_prepared_dir(prepared_path)
        else:
            parser.error("No DEFAULT_PREPARED_PATH set; pass --prepared_path or pass --dataset with --data_dir.")

    if prepared_path:
        ds = _load_rlds_from_prepared_dir(prepared_path, split=split)
    else:
        builder = tfds.builder(dataset_name, data_dir=args.data_dir)
        builder.download_and_prepare(download_dir=args.data_dir)  # no-op if already prepared
        ds = builder.as_dataset(split=split)

    if summary:
        _print_episode_summary(ds, episode)
        return

    frames_iter = iter_episode_frames(ds, episode_index=episode, preferred_key=frame_key, max_frames=max_frames)

    if save_dir:
        n = _save_frames(frames_iter, save_dir, prefix=f"ep{episode}")
        print(f"Saved {n} frames to: {save_dir}")
    else:
        _display_frames(frames_iter, fps=fps, window=f"{dataset_name}:{split}:ep{episode}")


if __name__ == "__main__":
    main()