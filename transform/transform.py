from typing import Any, Dict
import numpy as np


def transform_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Map one source step to the target dataset config.

    Input:
        step: nested dict of numpy arrays / numpy scalars

    Output:
        {
            "observation": {"state": np.ndarray(shape=(7,), dtype=np.float32)},
            "action": np.ndarray(shape=(7,), dtype=np.float32),
            "discount": np.float32,
            "reward": np.float32,
            "is_first": np.bool_,
            "is_last": np.bool_,
            "is_terminal": np.bool_,
            "language_instruction": str or bytes,
        }
    """

    arm_angles = np.asarray(step["observation"]["arm_angles"], dtype=np.float32)
    gripper = np.asarray(step["observation"]["gripper"], dtype=np.float32)

    if arm_angles.shape != (6,):
        raise ValueError(
            f"Expected observation['arm_angles'] shape (6,), got {arm_angles.shape}"
        )

    gripper = gripper.reshape(1,)

    state = np.concatenate([arm_angles, gripper], axis=0).astype(np.float32)

    action = np.asarray(step["action"], dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(
            f"Expected action shape (7,), got {action.shape}"
        )

    transformed_step = {
        "observation": {
            "state": state,
        },
        "action": action,
        "discount": np.float32(step["discount"]),
        "reward": np.float32(step["reward"]),
        "is_first": np.bool_(step["is_first"]),
        "is_last": np.bool_(step["is_last"]),
        "is_terminal": np.bool_(step["is_terminal"]),
        "language_instruction": step["language_instruction"],
    }

    return transformed_step