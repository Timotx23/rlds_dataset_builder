import tensorflow_datasets as tfds

DATASET_NAME = "test_episode"

# How many steps per episode to print (avoid spam)
MAX_STEPS_TO_PRINT = 3

print(f"Inspecting dataset: {DATASET_NAME}")

# -----------------------------
# LOAD DATASET
# -----------------------------
try:
    ds = tfds.load(DATASET_NAME, split="train")
    print("✅ Dataset loaded successfully\n")
except Exception as e:
    print("❌ Failed to load dataset")
    print(e)
    exit(1)


def decode_if_bytes(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


# -----------------------------
# ITERATE EPISODES
# -----------------------------
for episode_idx, episode in enumerate(tfds.as_numpy(ds)):

    print("\n==================================================")
    print(f"EPISODE {episode_idx}")
    print("==================================================")

    # -----------------------------
    # METADATA
    # -----------------------------
    metadata = episode.get("episode_metadata", {})

    print("\n--- METADATA ---")
    for k, v in metadata.items():
        v = decode_if_bytes(v)
        print(f"{k}: {v}")

    # -----------------------------
    # STEPS
    # -----------------------------
    steps = list(episode["steps"])
    print(f"\nNumber of steps: {len(steps)}")

    if len(steps) == 0:
        print("❌ No steps in episode")
        continue

    # -----------------------------
    # COLUMN NAMES
    # -----------------------------
    print("\n--- STEP COLUMNS ---")
    print(list(steps[0].keys()))

    print("\n--- OBSERVATION COLUMNS ---")
    print(list(steps[0]["observation"].keys()))

    # -----------------------------
    # PRINT VALUES
    # -----------------------------
    print(f"\n--- VALUES (first {MAX_STEPS_TO_PRINT} steps) ---")

    for i in range(min(MAX_STEPS_TO_PRINT, len(steps))):
        step = steps[i]

        print(f"\nStep {i}:")
        print("--------------------------------------------------")

        # Observation
        print("Observation:")
        for k, v in step["observation"].items():
            print(f"  {k}: {v}")

        # Action
        print("Action:", step["action"])

        # Scalars
        print("Reward:", step["reward"])
        print("Discount:", step["discount"])
        print("is_first:", step["is_first"])
        print("is_last:", step["is_last"])
        print("is_terminal:", step["is_terminal"])

        # Language
        instruction = decode_if_bytes(step["language_instruction"])
        print("Language:", instruction)

print("\n==================================================")
print("🎉 INSPECTION COMPLETE")
print("==================================================")