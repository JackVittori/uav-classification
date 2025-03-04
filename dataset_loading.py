import os
import torch
from scipy.io import loadmat

class DatasetLoader:
    def __init__(self, root_dir, mapping=False):
        """
        Initialize the DatasetLoader.

        Parameters:
        - root_dir (str): Path to the root folder containing the dataset.
        - mapping (bool): If True, maps labels to integers.
        """
        self.root_dir = root_dir
        self.mapping = mapping

        # Define the mapping from folder names to integers
        self.label_mapping = {
            "1_bird": 0,
            "2_bird_heli": 1,
            "3_long_blades": 2,
            "4_rc_plane": 3,
            "5_short_blades": 4,
            "6_drone": 5
        }

    def load_data(self):
        """
        Load the dataset from the root directory.

        Returns:
        - data (list of torch.Tensor): List of tensors representing the data samples.
        - labels (list of torch.Tensor or str): List of labels corresponding to each sample.
        """
        data = []
        labels = []

        # Iterate through each folder in the root directory
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)

            # Skip if it's not a directory
            if not os.path.isdir(folder_path):
                continue

            # Get the label for the current folder
            if self.mapping:
                label = torch.tensor(self.label_mapping[folder_name])
            else:
                label = folder_name

            # Iterate through all files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".mat"):
                    file_path = os.path.join(folder_path, file_name)

                    # Load the .mat file
                    mat_data = loadmat(file_path)

                    # Assume the .mat file contains a single key-value pair with the data
                    # Exclude metadata keys like "__header__", "__version__", "__globals__"
                    data_key = [key for key in mat_data.keys() if not key.startswith("__")][0]
                    tensor_data = torch.tensor(mat_data[data_key])

                    # Append the data and label to the lists
                    data.append(tensor_data)
                    labels.append(label)

        return data, labels

    def count_samples_per_label(self):
        """
        Count the number of samples for each label.

        Returns:
        - counts (dict): A dictionary where keys are labels and values are the counts of samples.
        """
        counts = {}

        # Iterate through each folder in the root directory
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)

            # Skip if it's not a directory
            if not os.path.isdir(folder_path):
                continue

            # Get the label for the current folder
            label = self.label_mapping[folder_name] if self.mapping else folder_name

            # Count the number of .mat files in the folder
            count = len([file_name for file_name in os.listdir(folder_path) if file_name.endswith(".mat")])
            counts[label] = count

        return counts

    def debug_folder(self, folder_name):
        """
        da correggere
        Debug the specified folder to check for missing or problematic files.

        Parameters:
        - folder_name (str): The name of the folder to debug.

        Returns:
        - issues (dict): A dictionary with details about missing or problematic files.
        """
        folder_path = os.path.join(self.root_dir, folder_name)
        issues = {
            "missing_files": [],
            "failed_to_load": []
        }

        # Check if the folder exists
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder {folder_name} does not exist.")

        # Collect all file indices based on the naming convention
        expected_indices = set()
        actual_indices = set()

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".mat"):
                try:
                    # Extract index from the file name (e.g., 3bbfile1.mat -> 1)
                    index = int(''.join(filter(str.isdigit, file_name)))
                    actual_indices.add(index)

                    # Attempt to load the file to check for loading issues
                    file_path = os.path.join(folder_path, file_name)
                    mat_data = loadmat(file_path)

                except Exception as e:
                    issues["failed_to_load"].append({"file_name": file_name, "error": str(e)})

        # Assume expected indices are continuous from 1 to max(actual_indices)
        if actual_indices:
            max_index = max(actual_indices)
            expected_indices = set(range(1, max_index + 1))

        # Find missing files
        issues["missing_files"] = list(expected_indices - actual_indices)

        return issues

if __name__ == "__main__":

    # Example usage
    root_dir = "data"  # Path to your dataset root directory
    dataset = DatasetLoader(root_dir, mapping=True)
    data, labels = dataset.load_data()
    print(data[4014].shape, labels[4014])
    print(dataset.count_samples_per_label())
    print(len(dataset.debug_folder("3_long_blades")['missing_files']))
