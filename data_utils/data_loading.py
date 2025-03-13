import tensorflow as tf
import pathlib
import random
from typing import Tuple, List, Union, Optional, Dict
import logging
from collections import defaultdict, Counter
from PIL import Image, ImageChops

logger = logging.getLogger(__name__)
AUTOTUNE = tf.data.AUTOTUNE


def remove_white_frame(image: tf.Tensor, threshold: float = 0.95) -> tf.Tensor:
    """Removes the white frame from an image tensor.

    Convert the image to grey scale to create a mask of white pixels. On a 0 -> 1 scaled image white pixels are
    represented by near to 1 value. A mask is created marking by 1 the pixels less than the threshold and by 0 the near
    to white pixels. Then the coordinates of the bounding are retrieved and the image is cropped.

    Args:
        image (tf.Tensor): Tensor image of size [height, width, channels].
        threshold (float): Threshold to remove white pixels.

    Returns
        tf.Tensor: Cropped image tensor.
    """
    # Convert image to grayscale
    gray = tf.image.rgb_to_grayscale(image)

    # Create a mask of white areas
    mask = tf.where(gray < threshold, 1.0, 0.0)

    # Find bounding box coordinates
    mask = tf.reduce_max(mask, axis=-1)  # remove the last dimension
    rows = tf.reduce_max(mask, axis=1)  # max along width
    cols = tf.reduce_max(mask, axis=0)  # max along height
    row_start = tf.argmax(rows, output_type=tf.int32)  # first non-white row
    row_end = tf.shape(rows)[0] - tf.argmax(tf.reverse(rows, axis=[0]), output_type=tf.int32)  # last non-white row
    col_start = tf.argmax(cols, output_type=tf.int32)  # first non-white column
    col_end = tf.shape(cols)[0] - tf.argmax(tf.reverse(cols, axis=[0]), output_type=tf.int32)  # last non-white column

    # Crop the image
    cropped_image = image[row_start:row_end, col_start:col_end, :]

    return cropped_image


def retrieve_images(data_root: str,
                    shuffle: bool = True,
                    seed: int = 123,
                    exclude_multiple: bool = True) -> Tuple[List[str], List[int]]:
    """Retrieves all image file paths and labels from a directory and optionally shuffles them.

    Args:
        data_root (str): Directory where images are stored.
        shuffle (bool): Whether to shuffle the list of image paths. Defaults to True.
        seed (int): Random seed for reproducible shuffling. Defaults to 123.
        exclude_multiple (bool): Whether to exclude 'Bird+2_Blade_rotor' classes. Defaults to True.

    Returns:
        Tuple[List[str], List[int]]: List of image file paths and correspondent labels.
    """
    data_root = pathlib.Path(data_root)

    # exclude Bird+2_Blade_rotor if requested
    all_image_paths = [str(path) for path in data_root.glob('*/*') if
                       not (exclude_multiple and pathlib.Path(path).parent.name.startswith('Bird+2_Blade_rotor'))]

    if shuffle:
        random.seed(seed)
        random.shuffle(all_image_paths)

    logger.info(f"Successfully retrieved {len(all_image_paths)} image paths")

    label_names = sorted(set(pathlib.Path(path).parent.name for path in all_image_paths))
    label_to_index = {name: index for index, name in enumerate(label_names)}

    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    return all_image_paths, all_image_labels


def load_and_preprocess_image(image_path: str,
                              resize: Tuple[int, int] = (224, 224),
                              normalization: bool = False) -> tf.Tensor:
    """Loads an image from file, decodes it as JPEG, remove white frames, normalizes pixel values and resize it.

    Args:
        image_path (str): Path to the image file.
        resize (Tuple[int, int], optional): Target size for the image in the format (height, width).
            Defaults to (224,224).
        normalization (bool, optional): Whether to normalize the image. Defaults to False.

    Returns:
        tf.Tensor: Preprocessed image tensor with shape (height, width, 3) and values normalized to [0, 1].
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # normalization to [0,1]
    image = remove_white_frame(image)  # remove white frames
    if not normalization:
        image = image * 255
    image = tf.image.resize(image, resize)  # resize images
    return image


def count_samples_per_class(dataset: tf.data.Dataset,
                            true_labels_path: str = None) -> Dict:
    """Counts the number of samples per class in a batched dataset.

    Args:
        dataset (tf.data.Dataset): A batched dataset of (image, label) pairs.
        true_labels_path (str): The path of the directory containing data. If given, it returns true labels. Defaults to None.

    Returns:
        dict: A dictionary with class indices as keys and counts as values.
    """
    class_counts = defaultdict(int)

    for images, labels in dataset:
        labels_np = labels.numpy()
        batch_count = Counter(labels_np)

        for label, count in batch_count.items():
            class_counts[label] += count

    num_classes = max(class_counts.keys()) + 1
    index_to_label = {i: f"Class_{i}" for i in range(num_classes)}

    if true_labels_path is not None:
        data_root = pathlib.Path(true_labels_path)
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

        if len(label_names) != num_classes:
            label_names.remove("Bird+2_Blade_rotor")

        index_to_label = {index: name for index, name in enumerate(label_names)}

    return {index_to_label[i]: class_counts.get(i, 0) for i in range(num_classes)}


def create_dataset(data_root: str,
                   train_ratio: float = 0.8,
                   test_ratio: float = 0.2,
                   val_ratio: Optional[float] = None,
                   resize: Tuple[int, int] = (1050, 1400),
                   batch_size: Optional[int] = None,
                   shuffle: bool = True,
                   exclude_multiclass: bool = False,
                   normalization: bool = False,
                   seed: int = 123) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset],
Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]]:
    """Creates tensorflow datasets for training, validation (optional) and testing.

    Args:
        data_root(str): Directory where images are stored.
        train_ratio(float): The proportion of the dataset to include in the train split.
        test_ratio(float): The proportion of the dataset to include in the test split.
        val_ratio(Optional[float]): The proportion of the dataset to include in the validation split. Defaults to None.
        resize(Tuple[int, int], optional): Target size for the images in the format (height, width).
        batch_size(int): Batch size. Defaults to None.
        shuffle(bool): Whether to shuffle the list of image paths. Defaults to True.
        exclude_multiclass(bool): Whether to exclude 'Bird+2_Blade_rotor' classes.
        normalization (bool): Whether to normalize the image from [0,255] to [0,1]. Defaults to False.
        seed(int): Seed for reproducible shuffling. Defaults to 123.

    Returns:
        Union[Tuple[tf.data.Dataset, tf.data.Dataset],Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]]:
        Training and test dataset (if val_ratio = None) else training, validation and test dataset.

    Raises:
        ValueError: If the sum of train, validation and test ratio is not equal to 1.
    """
    if val_ratio is None:
        if train_ratio + test_ratio != 1:
            raise ValueError("The sum of train_ratio and test_ratio must be equal to 1.")

    else:
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("The sum of train_ratio, val_ratio and test_ratio must be equal to 1.")

    images_paths, labels_paths = retrieve_images(data_root,
                                                 shuffle=shuffle,
                                                 exclude_multiple=exclude_multiclass,
                                                 seed=seed)
    total_images = len(images_paths)

    if val_ratio is not None:
        # compute train/vali/test splits
        train_size = int(train_ratio * total_images)
        val_size = int(val_ratio * total_images)
        # split the paths
        train_paths = images_paths[:train_size]
        val_paths = images_paths[train_size:train_size + val_size]
        test_paths = images_paths[train_size + val_size:]
        # split the labels
        train_labels = labels_paths[:train_size]
        val_labels = labels_paths[train_size:train_size + val_size]
        test_labels = labels_paths[train_size + val_size:]
        # create path datasets
        train_path_ds = tf.data.Dataset.from_tensor_slices(train_paths)
        val_path_ds = tf.data.Dataset.from_tensor_slices(val_paths)
        test_path_ds = tf.data.Dataset.from_tensor_slices(test_paths)
        # load and preprocess images
        train_images_ds = train_path_ds.map(
            lambda path: load_and_preprocess_image(path, resize=resize, normalization=normalization))
        val_images_ds = val_path_ds.map(
            lambda path: load_and_preprocess_image(path, resize=resize, normalization=normalization))
        test_images_ds = test_path_ds.map(
            lambda path: load_and_preprocess_image(path, resize=resize, normalization=normalization))
        # create label datasets
        train_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
        val_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))
        test_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))
        # create (image, label) pairs dataset
        train_ds = tf.data.Dataset.zip((train_images_ds, train_labels_ds))
        val_ds = tf.data.Dataset.zip((val_images_ds, val_labels_ds))
        test_ds = tf.data.Dataset.zip((test_images_ds, test_labels_ds))
        # logging info
        logger.info(
            f"Splits: {len(train_ds)}/{total_images} for training, {len(val_ds)}/{total_images} for validation, {len(test_ds)}/{total_images} for testing.")

        if batch_size is not None:
            return train_ds.batch(batch_size), val_ds.batch(batch_size), test_ds.batch(batch_size)
        return train_ds, val_ds, test_ds

    else:
        # compute train/test splits
        train_size = int(train_ratio * total_images)
        # split the paths
        train_paths = images_paths[:train_size]
        test_paths = images_paths[train_size:]
        # split the labels
        train_labels = labels_paths[:train_size]
        test_labels = labels_paths[train_size:]
        # create path datasets
        train_path_ds = tf.data.Dataset.from_tensor_slices(train_paths)
        test_path_ds = tf.data.Dataset.from_tensor_slices(test_paths)
        # load and preprocess images
        train_images_ds = train_path_ds.map(
            lambda path: load_and_preprocess_image(path, resize=resize, normalization=normalization))
        test_images_ds = test_path_ds.map(
            lambda path: load_and_preprocess_image(path, resize=resize, normalization=normalization))
        # create label datasets
        train_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
        test_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))
        # create (image, label) pairs dataset
        train_ds = tf.data.Dataset.zip((train_images_ds, train_labels_ds))
        test_ds = tf.data.Dataset.zip((test_images_ds, test_labels_ds))
        # logging info
        logger.info(f"Splits: {len(train_ds)}/{total_images} for training, {len(test_ds)}/{total_images} for testing.")
        if batch_size is not None:
            return train_ds.batch(batch_size), test_ds.batch(batch_size)
        return train_ds, test_ds
