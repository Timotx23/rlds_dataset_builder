import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

DATASET_NAME = "Cobot280PiDataset"
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print(f"Testing TFDS load for dataset: {DATASET_NAME}")
print(f"Run ID: {RUN_ID}")


def decode_if_bytes(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


# -----------------------------
# LOAD DATASET + INFO
# -----------------------------
try:
    ds, info = tfds.load(DATASET_NAME, split="train", with_info=True)
    print("✅ tfds.load() SUCCESS")
except Exception as e:
    print("❌ tfds.load() FAILED")
    print(e)
    raise SystemExit(1)

# -----------------------------
# PRINT DATASET METADATA
# -----------------------------
print("\n================ DATASET INFO ================")
print("Name:", info.name)
print("Version:", info.version)
print("Description:", info.description)
print("Homepage:", info.homepage)
print("Splits:", info.splits)

print("\n--- FEATURE STRUCTURE (SCHEMA) ---")
print(info.features)

# -----------------------------
# COLLECT ALL EPISODES
# -----------------------------
episodes_data = []

try:
    for fallback_episode_idx, episode in enumerate(tfds.as_numpy(ds)):
        metadata = episode["episode_metadata"]

        episode_index = metadata.get("episode_index", fallback_episode_idx)
        episode_index = int(episode_index)

        file_path = decode_if_bytes(metadata.get("file_path", "UNKNOWN"))
        file_name = os.path.basename(file_path) if file_path != "UNKNOWN" else "UNKNOWN"

        steps = list(episode["steps"])
        if len(steps) == 0:
            print(f"⚠️ Skipping empty episode {episode_index}")
            continue

        obs_arm = np.array(
            [step["observation"]["arm_angles"] for step in steps],
            dtype=np.float32,
        )
        obs_grip = np.array(
            [step["observation"]["gripper"] for step in steps],
            dtype=np.uint8,
        )
        action = np.array(
            [step["action"] for step in steps],
            dtype=np.float32,
        )
        reward = np.array(
            [step["reward"] for step in steps],
            dtype=np.float32,
        )
        discount = np.array(
            [step["discount"] for step in steps],
            dtype=np.float32,
        )
        is_first = np.array(
            [step["is_first"] for step in steps],
            dtype=np.bool_,
        )
        is_last = np.array(
            [step["is_last"] for step in steps],
            dtype=np.bool_,
        )
        is_terminal = np.array(
            [step["is_terminal"] for step in steps],
            dtype=np.bool_,
        )

        # Basic validation
        if obs_arm.ndim != 2 or obs_arm.shape[1] != 6:
            print(f"❌ Episode {episode_index}: invalid obs_arm shape {obs_arm.shape}")
            continue

        if action.ndim != 2 or action.shape[1] != 7:
            print(f"❌ Episode {episode_index}: invalid action shape {action.shape}")
            continue

        if np.isnan(action).any():
            print(f"❌ Episode {episode_index}: NaNs found in action")
            continue

        episodes_data.append({
            "episode_index": episode_index,
            "file_path": file_path,
            "file_name": file_name,
            "num_steps": len(steps),
            "obs_arm": obs_arm,
            "obs_grip": obs_grip,
            "action": action,
            "reward": reward,
            "discount": discount,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        })

except Exception as e:
    print("❌ Iteration failed")
    print(e)
    raise SystemExit(1)

if not episodes_data:
    print("❌ No valid episodes found")
    raise SystemExit(1)

# Sort by stored episode index
episodes_data.sort(key=lambda x: x["episode_index"])

# -----------------------------
# PRINT SUMMARY
# -----------------------------
print("\n================ EPISODE SUMMARY ================")
for ep in episodes_data:
    print(
        f"Episode {ep['episode_index']:>3} | "
        f"steps={ep['num_steps']:<4} | "
        f"file={ep['file_name']}"
    )
    print(f"    full path: {ep['file_path']}")

print(f"\n✅ Collected {len(episodes_data)} valid episode(s)")

# -----------------------------
# VISUALIZE ALL EPISODES TOGETHER
# -----------------------------

# 1) All rewards together
plt.figure(figsize=(10, 5))
for ep in episodes_data:
    t = np.arange(ep["num_steps"])
    plt.plot(t, ep["reward"], label=f"ep_{ep['episode_index']}")
plt.title("Rewards Across All Episodes")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()

# 2) All discounts together
plt.figure(figsize=(10, 5))
for ep in episodes_data:
    t = np.arange(ep["num_steps"])
    plt.plot(t, ep["discount"], label=f"ep_{ep['episode_index']}")
plt.title("Discounts Across All Episodes")
plt.xlabel("Step")
plt.ylabel("Discount")
plt.legend()
plt.grid(True)
plt.show()

# 3) All gripper states together
plt.figure(figsize=(10, 5))
for ep in episodes_data:
    t = np.arange(ep["num_steps"])
    plt.plot(t, ep["obs_grip"], label=f"ep_{ep['episode_index']}")
plt.title("Gripper State Across All Episodes")
plt.xlabel("Step")
plt.ylabel("Gripper")
plt.legend()
plt.grid(True)
plt.show()

# 4) Arm joint trajectories: one plot per joint, all episodes together
for joint_idx in range(6):
    plt.figure(figsize=(10, 5))
    for ep in episodes_data:
        t = np.arange(ep["num_steps"])
        plt.plot(t, ep["obs_arm"][:, joint_idx], label=f"ep_{ep['episode_index']}")
    plt.title(f"Observed Arm Angle Joint {joint_idx} Across All Episodes")
    plt.xlabel("Step")
    plt.ylabel("Angle")
    plt.legend()
    plt.grid(True)
    plt.show()

# 5) Action trajectories: one plot per action dimension, all episodes together
for action_idx in range(7):
    plt.figure(figsize=(10, 5))
    for ep in episodes_data:
        t = np.arange(ep["num_steps"])
        plt.plot(t, ep["action"][:, action_idx], label=f"ep_{ep['episode_index']}")
    plt.title(f"Action Dimension {action_idx} Across All Episodes")
    plt.xlabel("Step")
    plt.ylabel("Action Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# 6) Terminal flags together
plt.figure(figsize=(10, 5))
for ep in episodes_data:
    t = np.arange(ep["num_steps"])
    plt.plot(t, ep["is_terminal"].astype(np.int32), label=f"ep_{ep['episode_index']}")
plt.title("Terminal Flags Across All Episodes")
plt.xlabel("Step")
plt.ylabel("is_terminal")
plt.legend()
plt.grid(True)
plt.show()

print("\n==================================================")
print("🎉 COMPARATIVE VISUALIZATION COMPLETE")
print(f"RUN ID: {RUN_ID}")
print("==================================================")