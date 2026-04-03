import tensorflow_datasets as tfds
import numpy as np

DATASET_NAME = "test_episode"

print(f"Testing TFDS load for dataset: {DATASET_NAME}")

# Try loading dataset
try:
    ds = tfds.load(DATASET_NAME, split="train")
    print("✅ tfds.load() SUCCESS")
except Exception as e:
    print("❌ tfds.load() FAILED")
    print(e)
    exit(1)

# Try iterating
try:
    episode = next(iter(tfds.as_numpy(ds)))
    print("✅ Successfully iterated dataset")
except Exception as e:
    print("❌ Iteration failed")
    print(e)
    exit(1)

# Check structure
print("\n--- STRUCTURE CHECK ---")
print("Top-level keys:", episode.keys())

if "steps" not in episode:
    print("❌ Missing 'steps' key → NOT RLDS format")
    exit(1)

steps = list(episode["steps"])
print(f"Number of steps: {len(steps)}")

if len(steps) == 0:
    print("❌ No steps found")
    exit(1)

# Inspect first step
step = steps[0]

print("\n--- FIRST STEP CHECK ---")
print("Step keys:", step.keys())

# Check observation
if "observation" not in step:
    print("❌ Missing observation")
    exit(1)

obs = step["observation"]

print("Observation keys:", obs.keys())

# Check required fields
required_obs = ["arm_angles", "gripper"]
for key in required_obs:
    if key not in obs:
        print(f"❌ Missing observation field: {key}")
        exit(1)

# Check shapes
arm = obs["arm_angles"]
action = step["action"]

print("\n--- SHAPE CHECK ---")
print("arm_angles shape:", arm.shape)
print("action shape:", action.shape)

if arm.shape != (6,):
    print("❌ arm_angles should be shape (6,)")
    exit(1)

if action.shape != (7,):
    print("❌ action should be shape (7,)")
    exit(1)

# Check flags
print("\n--- FLAG CHECK ---")
print("is_first:", step["is_first"])
print("is_last:", steps[-1]["is_last"])
print("is_terminal:", steps[-1]["is_terminal"])

if not steps[0]["is_first"]:
    print("❌ First step is not marked correctly")
    exit(1)

if not steps[-1]["is_last"]:
    print("❌ Last step is not marked correctly")
    exit(1)

# Check for NaNs
print("\n--- VALUE CHECK ---")
if np.isnan(action).any():
    print("❌ NaNs found in action")
    exit(1)

print("\n🎉 ALL CHECKS PASSED — DATASET IS VALID TFDS + RLDS FORMAT")