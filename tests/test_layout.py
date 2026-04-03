import tensorflow_datasets as tfds
import numpy as np

DATASET_NAME = "test_episode"

print(f"Testing TFDS load for dataset: {DATASET_NAME}")

# -----------------------------
# LOAD DATASET + INFO
# -----------------------------
try:
    ds, info = tfds.load(DATASET_NAME, split="train", with_info=True)
    print("✅ tfds.load() SUCCESS")
except Exception as e:
    print("❌ tfds.load() FAILED")
    print(e)
    exit(1)

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
# ITERATE ONE EPISODE
# -----------------------------
try:
    episode = next(iter(tfds.as_numpy(ds)))
    print("\n✅ Successfully iterated dataset")
except Exception as e:
    print("❌ Iteration failed")
    print(e)
    exit(1)

# -----------------------------
# TOP-LEVEL STRUCTURE
# -----------------------------
print("\n================ EPISODE STRUCTURE ================")
print("Top-level keys:", list(episode.keys()))

print("\n--- EPISODE METADATA ---")
for k, v in episode["episode_metadata"].items():
    print(f"{k}: {v}")

# -----------------------------
# STEPS STRUCTURE
# -----------------------------
steps = list(episode["steps"])
print("\n--- STEPS ---")
print(f"Number of steps: {len(steps)}")

if len(steps) == 0:
    print("❌ No steps found")
    exit(1)

step = steps[0]

print("\n================ STEP STRUCTURE ================")
print("Step keys (columns):", list(step.keys()))

# -----------------------------
# OBSERVATION DETAILS
# -----------------------------
obs = step["observation"]
print("\n--- OBSERVATION ---")
print("Observation keys:", list(obs.keys()))

for key, val in obs.items():
    print(f"\n{key}:")
    print("  shape:", getattr(val, "shape", "scalar"))
    print("  dtype:", getattr(val, "dtype", type(val)))
    print("  sample value:", val)

# -----------------------------
# ACTION DETAILS
# -----------------------------
print("\n--- ACTION ---")
print("shape:", step["action"].shape)
print("dtype:", step["action"].dtype)
print("sample value:", step["action"])

# -----------------------------
# FLAGS + SCALARS
# -----------------------------
print("\n--- SCALARS / FLAGS ---")
for key in ["reward", "discount", "is_first", "is_last", "is_terminal"]:
    val = step[key]
    print(f"{key}: {val} (type={type(val)})")

# -----------------------------
# LANGUAGE
# -----------------------------
print("\n--- LANGUAGE ---")
instruction = step["language_instruction"]
if isinstance(instruction, bytes):
    instruction = instruction.decode("utf-8")
print("language_instruction:", instruction)

# -----------------------------
# SHAPE VALIDATION
# -----------------------------
print("\n================ SHAPE CHECK ================")
if obs["arm_angles"].shape != (6,):
    print("❌ arm_angles wrong shape")
    exit(1)

if step["action"].shape != (7,):
    print("❌ action wrong shape")
    exit(1)

# -----------------------------
# EPISODE CONSISTENCY
# -----------------------------
print("\n================ EPISODE CHECK ================")
print("First step is_first:", steps[0]["is_first"])
print("Last step is_last:", steps[-1]["is_last"])
print("Last step is_terminal:", steps[-1]["is_terminal"])

# -----------------------------
# NAN CHECK
# -----------------------------
print("\n================ VALUE CHECK ================")
if np.isnan(step["action"]).any():
    print("❌ NaNs found in action")
    exit(1)

# -----------------------------
# PRINT MULTIPLE STEPS (OPTIONAL)
# -----------------------------
print("\n================ SAMPLE STEPS ================")
for i in range(min(3, len(steps))):
    print(f"\nStep {i}:")
    print("  arm_angles:", steps[i]["observation"]["arm_angles"])
    print("  action:", steps[i]["action"])

print("\n🎉 FULL INSPECTION COMPLETE — DATASET LOOKS VALID")