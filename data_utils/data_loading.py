import os
import logging
import tensorflow as tf
from typing import Tuple, Optional, Any, Union, Dict
import pathlib
import sys

logger = logging.getLogger(__name__)


def load_and_preprocess_dataset(data_dir: str,
                                image_size: Tuple[int, int] = (1400, 1050),
                                batch_size: int = 32,
                                shuffle: bool = True,
                                seed: int = 123,
                                **kwargs: Any) -> tf.data.Dataset:
    """
    Loads and preprocesses an image dataset from the specified directory.

    This function loads images from the specified directory and applies basic preprocessing,
    such as resizing and batching. It also allows for additional preprocessing through
    the use of keyword arguments passed via `kwargs`.

    Args:
        data_dir (str): Path to the directory containing images.
        image_size (Tuple[int, int], optional): Target size for resizing images. Default is (256, 256).
        batch_size (int, optional): Number of images per batch. Default is 32.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
        seed (int, optional): Seed for the random number generator. Default is 123.
        **kwargs: Additional arguments passed to the `image_dataset_from_directory` function,
                  such as `labels`, `label_mode`, `class_names`, `color_mode`, etc.

    Returns:
        tf.data.Dataset: A batched and preprocessed TensorFlow dataset.
    """
    data_path = pathlib.Path(data_dir)
    # Load the dataset from the directory
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        verbose=False,
        **kwargs)
    logger.info(f"{len(list(data_path.glob('*/*.jpg')))} images loaded from {data_dir}")
    return dataset


def dataset_partition(
        dataset: tf.data.Dataset,
        train_ratio: float = 0.8,
        val_ratio: Optional[float] = None,
        test_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int = 123
) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]]:
    """
    Splits the given dataset into training, validation (optional), and testing sets by shuffling the batches.

    This function partitions the input dataset based on the specified ratios for training, validation,
    and testing datasets. If no validation ratio is provided, the function will only return
    the training and testing datasets. The function also supports shuffling the dataset before splitting.

    Args:
        dataset (tf.data.Dataset): The dataset to be split.
        train_ratio (float): The ratio of the dataset to be used for training. Must be a float between 0 and 1. Default is 0.8.
        val_ratio (Optional[float]): The ratio of the dataset to be used for validation. If None, no validation dataset will be returned. Default is None.
        test_ratio (float): The ratio of the dataset to be used for testing. Must be a float between 0 and 1. Default is 0.2.
        shuffle (bool): Whether to shuffle the dataset before splitting. Default is True.
        seed: Set seed for the random number generator. Default is 123.

    Returns:
        Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]]:
            - If no validation ratio is specified, returns a tuple of training and testing datasets.
            - If a validation ratio is specified, returns a tuple of training, validation, and testing datasets.

    Raises:
        ValueError: If the sum of train_ratio, val_ratio, and test_ratio is not equal to 1.0.
    """
    if train_ratio + test_ratio != 1:
        raise ValueError(f"The sum of train_ratio and test_ratio must be equal to 1")

    if val_ratio is None:
        val_ratio = 1 - train_ratio - test_ratio
    if train_ratio + val_ratio + test_ratio != 1:
        if val_ratio is None:
            raise ValueError("The sum of train_ratio, and test_ratio must equal 1")
        else:
            raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must equal 1")

    # Get dataset size
    ds_size = len(dataset)
    # Shuffle the dataset if required
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10, seed=seed)

    # Calculate the sizes of each dataset partition
    train_size = int(train_ratio * ds_size)
    val_size = int(val_ratio * ds_size)

    # Split the dataset
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size)

    sample_image = next(iter(dataset))  # Take one sample image to get the shape
    image_size = sample_image.shape if isinstance(sample_image, tf.Tensor) else sample_image[0].shape
    logger.info(f"Original batch number: {ds_size}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Training dataset contains {train_size} batches.")
    logger.info(f"Validation dataset contains {val_size} batches." if val_ratio else "No validation dataset created.")
    logger.info(f"Test dataset contains {ds_size - train_size - val_size} batches.")

    if val_ratio == 0:
        return train_ds, test_ds
    else:
        return train_ds, val_ds, test_ds


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )

    ds = load_and_preprocess_dataset(data_dir='/Users/jackvittori/Desktop/uav-classification/images')
    train, test = dataset_partition(dataset=ds, train_ratio=0.8, test_ratio=0.2)

    print(type(train))
