# Cobot280PiDataset (TFDS / RLDS)

## Description
This dataset contains robotic interaction data collected from a Cobot 280 Pi setup.  
The data is originally stored in **HDF5 format** and then converted into **RLDS-compatible TFDS format**.

**Important Naming Rule**  
The name `Cobot280PiDataset` **must not be changed** unless:
- The `dataset_builder` file name is also changed  
- The class name inside that file is updated accordingly  

---

## Purpose
This dataset is intended for training and evaluating **reinforcement learning (RL)** models on robotic manipulation tasks using the **RLDS (Reinforcement Learning Dataset Standard)** format.

---

## Dataset Structure

### Observation Space
```yaml
observation:
  arm_angles: Tensor
  cam_external: Image
  cam_wrist: Image
  gripper: Scalar
```

### Step Data
```yaml
steps:
  actions: Tensor
  discount: Scalar
  is_first: Scalar
  is_last: Scalar
  is_terminal: Scalar
```

---

## Conversion
Dataset pipeline:
```
HDF5 → RLDS / TFDS
```

---

## Setup & Build

### 1. Install Environment
```bash
conda env create -f environment_macos.yml
# or
conda env create -f environment_ubuntu.yml
```

Activate it:
```bash
conda activate <environment_name>
```

### 2. Build Dataset
```bash
cd Cobot280PiDataset
tfds build
```

---

## Inspecting the Dataset

Navigate to the test folder:
```bash
cd tests
```

### Available Scripts

#### Inspect dataset values
```bash
python inspect_dataset.py
```

#### Test transformations (sanity check before build)
```bash
python test_dataset_transform.py Cobot280PiDataset
```

#### Visualize dataset
```bash
python visualize_dataset.py
```

---

## Accessing the Built Dataset

From the **root directory**:
```bash
cd tensorflow_datasets/cobot280_pi_dataset/
```

Then navigate to a version folder:
- Lower version number → older data  
- Higher version number → newer data  

---


## Dependencies
- tensorflow  
- tensorflow-datasets  
- rlds  
- h5py  
- numpy  

---

  





## Dataset Information 
- will return 3 files
- dataset_info
- dataset_features
- Actual HDF5 / RLDS formatted file to be used in training

---



## Notes
- Must be run from the **root directory**, otherwise paths may not resolve correctly  
- Dataset version folders indicate data age (lower = older)  
