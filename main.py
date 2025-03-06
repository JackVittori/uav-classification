import logging
import sys
from data_utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)

base_path = '/Users/jackvittori/Desktop/uav-classification/images'

if __name__ == '__main__':

    train_ds, test_ds = create_dataset(data_root=base_path)
    train_ds, validation_ds, test_ds = create_dataset(data_root=base_path,
                                                      train_ratio=0.8,
                                                      val_ratio=0.1,
                                                      test_ratio=0.1,
                                                      resize=(224, 224),
                                                      batch_size=64)

    logging.info(f" Train balance: {count_samples_per_class(train_ds, num_classes = 10)}")
    logging.info(f" Train balance: {count_samples_per_class(train_ds, num_classes=10, true_labels_path=base_path)}")
    logging.info(f" Test balance: {count_samples_per_class(test_ds, num_classes=10)}")
    logging.info(f" Test balance: {count_samples_per_class(test_ds, num_classes=10, true_labels_path=base_path)}")
    logging.info(f" Validation balance: {count_samples_per_class(validation_ds, num_classes=10)}")
    logging.info(f" Validation balance: {count_samples_per_class(validation_ds, num_classes=10, true_labels_path=base_path)}")
