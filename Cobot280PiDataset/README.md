TODO(example_dataset): Markdown description of your dataset.
Description is **formatted** as markdown.
Name of Cobot280piDataset MUST NOT be changed unless the dataset_builder file is also named the same combined with the class inside of said file!
This datset is formatted as HDF5 and has the following columns:
obersevation{
    arm_angles: Tensor
    cam_external: Image
    cam_wrist: Image
    gripper: Scalar
}
steps {

    actions: Tensor
    discount: Scalar
    is_first: Scalar
    is_last: Scalar
    is_terminal: Scalar
}
It is then converted to RLDS TFDS
To run:
install either environment_macos.yml or environment_ubuntu.yml
cd Cobot280PiDataset
tfds build

To check data:
cd tests
python inspect_dataset.py -> prints you the dataset to see what values are in it
python test_dataset_transform.py Cobot280PiDataset -> Lets you test the the tranform.py to check if everything works before running tfds build -> act like a sanity check
python visualize_dataset.py -> to visualize all the data inside of the dataset

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
