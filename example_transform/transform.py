from typing import Any, Dict
import numpy as np


def transform_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Maps step from source dataset to target dataset config.
       Input is dict of numpy arrays."""

    state = np.concatenate([
        step['observation']['arm_angles'].astype(np.float32),
        np.array([step['observation']['gripper']], dtype=np.float32)
    ], axis=0)

    transformed_step = {
        'observation': {
            'state': state,
        },
        'action': step['action'].astype(np.float32),
    }

    # copy over other fields unchanged
    for copy_key in [
        'discount', 'reward', 'is_first', 'is_last',
        'is_terminal', 'language_instruction'
    ]:
        transformed_step[copy_key] = step[copy_key]

    return transformed_step