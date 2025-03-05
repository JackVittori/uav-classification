import os
import pickle
import datetime
import logging

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory: str) -> None:
    """
    Ensures the target directory exists. If the directory does not exist, it is created.

    Args:
        directory (str): The path to the directory.

    """
    os.makedirs(directory, exist_ok=True)


def get_safe_filepath(directory: str, filename: str, extension: str = "pickle") -> str:
    """
    Generates a safe file path. If a file with the same name already exists,
    appends a timestamp to avoid overwriting.

    Args:
        directory (str): The target directory.
        filename (str): The base name of the file (without extension).
        extension (str, optional): The file extension. Defaults to "pickle".

    Returns:
        str: The full path to the file, including directory, filename, and extension.

    Example:
        If `training_results.pickle` exists, this might return `training_results_20250304_153000.pickle`.
    """
    ensure_directory_exists(directory)
    filepath = os.path.join(directory, f"{filename}.{extension}")

    if os.path.exists(filepath):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(directory, f"{filename}_{timestamp}.{extension}")

    return filepath


def save_as_pickle(obj: object, directory: str, filename: str) -> str:
    """
    Saves a Python object as a pickle file, avoiding overwrites by adding a timestamp if needed.

    Args:
        obj (object): The Python object to save (e.g., list, dict, etc.).
        directory (str): The directory where the file should be saved.
        filename (str): The base name of the file (without extension).

    Returns:
        str: The actual path where the file was saved.

    Example:
        >>> data = {"accuracy": 0.95, "loss": 0.02}
        >>> save_as_pickle(data, "./output", "training_results")
        This might save: ./output/training_results.pickle or ./output/training_results_20250304_153000.pickle
    """
    filepath = get_safe_filepath(directory, filename)

    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

    logger.info(f"Saved file successfully at: {filepath}")
    return filepath


def load_from_pickle(directory: str, filename: str) -> object:
    """
    Loads a Python object from a pickle file.

    Args:
        directory (str): The directory where the file is located.
        filename (str): The base name of the file (without extension).

    Returns:
        object: The Python object loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> data = load_from_pickle("./output", "training_results")
        This loads: ./output/training_results.pickle
    """
    filepath = os.path.join(directory, f"{filename}.pickle")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as file:
        obj = pickle.load(file)

    logger.info(f"Loaded file successfully from: {filepath}")
    return obj


def load_pickles_from_directory(directory: str) -> dict:
    """
    Recursively loads all pickle files from a given directory and its subdirectories.

    Args:
        directory (str): The directory to search for pickle files.

    Returns:
        dict: A dictionary where the keys are the file names (without extension)
              and the values are the loaded Python objects from the pickle files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")

    pickles = {}

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pickle"):
                # Generate the file path
                file_path = os.path.join(root, file)

                # Load the pickle file
                with open(file_path, 'rb') as f:
                    obj = pickle.load(f)

                # Store the object with the file name (without extension) as key
                file_key = os.path.splitext(file)[0]  # Remove the .pickle extension
                pickles[file_key] = obj
                logger.info(f"Loaded: {file_key} from {file_path}")

    return pickles

import sys
# example usage
if __name__ == '__main__':
    # logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )

    # Prova
    data = {"test": 123}
    save_as_pickle(data, "output", "example")

    all_pickles = load_pickles_from_directory("/Users/jackvittori/Desktop/test_results_giacomo/quantum")

    print(all_pickles.keys())