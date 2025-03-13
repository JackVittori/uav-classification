import logging
import sys
from data_utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from models import *
from tensorflow.keras.callbacks import EarlyStopping
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)

base_path = '/Users/jackvittori/Desktop/uav-classification/images'

#  Hyperparameters
#LR = 0.0001  # Learning rate efbo
LR = 0.001
N_b  = 32    # Batch size
N_e1 = 60    # Number of epochs for last layer
N_e2 = 50    # Number of epochs for first finetuning

if __name__ == '__main__':
    #with tf.device('/CPU:0'):
        train_ds, validation_ds, test_ds = create_dataset(data_root=base_path,
                                                          train_ratio=0.8,
                                                          val_ratio=0.1,
                                                          test_ratio=0.1,
                                                          resize=(224, 224),
                                                          batch_size=64,
                                                          exclude_multiclass=True)

        net = MNV2_custom(num_classes=5, learning_rate=LR)
        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=8,
                                   restore_best_weights=True)
        # First training
        history1 = net.fit(train_ds,
                           epochs=N_e1,
                           validation_data=validation_ds,
                           shuffle=True,
                           callbacks=[early_stop])

        L_e = len(history1.history['loss'])
        ep = range(1, L_e + 1)

        # Plot loss curve
        plt.figure()
        plt.plot(ep, history1.history['loss'], linewidth=2, label='Training loss')
        plt.plot(ep, history1.history['val_loss'], linewidth=2, label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig("/Users/jackvittori/Desktop/first_training_loss.pdf", format='pdf')

        plt.figure()
        plt.plot(ep, history1.history['accuracy'], linewidth=2, label='Training accuracy')
        plt.plot(ep, history1.history['val_accuracy'], linewidth=2, label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        fig_name = "/Users/jackvittori/Desktop/first_training_accuracy.pdf"
        plt.savefig(fig_name, format='pdf')

        # Unfreeze some
        unfreeze_model(net, N_unfreeze=20)  # EN: 20 - 100 - 242  # MN: 158

        history2 = net.fit(train_ds,
                           epochs=N_e2,
                           validation_data=validation_ds,
                           shuffle=True,
                           callbacks=[early_stop])

        L_e = len(history2.history['loss'])
        ep = range(1, L_e + 1)

        # Plot loss curve
        plt.figure()
        plt.plot(ep, history2.history['loss'], linewidth=2, label='Training loss')
        plt.plot(ep, history2.history['val_loss'], linewidth=2, label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        fig_name = "/Users/jackvittori/Desktop/second_training_loss.pdf"
        plt.savefig(fig_name, format='pdf')

        # Plot accuracy curve
        plt.figure()
        plt.plot(ep, history2.history['accuracy'], linewidth=2, label='Training accuracy')
        plt.plot(ep, history2.history['val_accuracy'], linewidth=2, label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        fig_name = "/Users/jackvittori/Desktop/second_training_accuracy.pdf"
        plt.savefig(fig_name, format='pdf')