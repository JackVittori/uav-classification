# -*- coding: utf-8 -*-
"""
Script for training the EfficientNet-B0 on the new dataset, derived from the
Damage Level Classification Challenge (PHI-Net), organized by the Pacific
Earthquake Engineering Research (PEER) Center, and complemented by the INGV DFM
database. The whole datset was manually labelled and pre-prcessed.
Sata augmentation has been used.

Created on Wed Dec 11 17:23:49 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import models
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.utils import shuffle
# import tensorflow_datasets as tfds



# Set main hyper-parameters
LR = 0.0001  # Learning rate
N_b  = 32    # Batch size
N_e1 = 60    # Number of epochs for last layer
N_e2 = 50    # Number of epochs for first finetuning


# Set data folder
data_folder = 'E:/Data/NPY/Data/'
save_folder = 'E:/Data/NPY/Models/'
result_folder = 'E:/Data/NPY/Results/'



# %% Load training set
# training_set = data_folder + 'X_train_augmented.npy'
# training_lab = data_folder + 'y_train_augmented.npy'
training_set = data_folder + 'X_train_wavelet.npy'
training_lab = data_folder + 'y_train_augmented.npy'


with tf.device('CPU/:0'):
    X = np.load(training_set)
    y = np.load(training_lab)

    np.random.seed(seed=42)
    idx = np.random.permutation(len(y))
    X = X[idx,:,:,:]
    y = y[idx]

    # X = X / 255.
    # X = np.array(X, dtype = 'float32')
    # y_cat = to_categorical(y, 4)

    # dataset = tf.data.Dataset.from_tensor_slices((X,y))
    # dataset = dataset.batch(N_b)

    dataset = tf.data.Dataset.from_tensor_slices((X[:-1931],y[:-1931]))
    dataset = dataset.batch(N_b)

    validation = tf.data.Dataset.from_tensor_slices((X[-1931:],y[-1931:]))
    validation = validation.batch(N_b)




# %% Select the model

# Early fusion (S3)
net = models.build_model(num_classes=4, LR=LR)
model_name = 'EfficientNetB0_val_wavlet_original'
# net.summary()


# net = models.build_MN_model(num_classes=4, LR=LR)
# model_name = 'MobileNetV2_'
# net.summary()


# Early stopping
# early_stop = EarlyStopping(monitor='accuracy', patience=2, restore_best_weights=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_folder, verbose=1, save_freq=150*N_b)


# %% Train the selected model

# history = net.fit(X, y_cat, batch_size=N_b, epochs=N_e1, validation_split=0.1,
#                   shuffle=True, callbacks=[early_stop])

# history1 = net.fit(dataset, epochs=N_e1, shuffle=True, callbacks=[early_stop])
history1 = net.fit(dataset, epochs=N_e1, validation_data=validation, shuffle=True, callbacks=[early_stop])
# history1 = net.fit(dataset, epochs=N_e1, validation_data=validation, shuffle=True, callbacks=[early_stop, cp_callback])


# Save the trained model
save_file = save_folder + model_name
# net.save(save_file)
# save_file = save_folder + model_name + '.h5'
# net.save(save_file, overwrite=True, include_optimizer=True, save_format='h5')

# np.save(save_folder + model_name + '_history.npy', history1.history)
    # history = np.load(save_folder + model_name + '_history.npy', allow_pickle='TRUE').item()


# %% Plot curves
L_e = len(history1.history['loss'])
ep = range(1, L_e+1)

# Plot loss curve
plt.figure()
plt.plot(ep, history1.history['loss'], linewidth=2, label='Training loss')
# plt.plot(ep, history1.history['val_loss'], linewidth=2, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
#plt.show()
# fig_name = save_folder + model_name + '_Loss.pdf'
# plt.savefig(fig_name, format='pdf')


# Plot accuracy curve
plt.figure()
plt.plot(ep, history1.history['accuracy'], linewidth=2, label='Training accuracy')
# plt.plot(ep, history1.history['val_accuracy'], linewidth=2, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
#plt.show()
# fig_name = save_folder + model_name + '_Accuracy.pdf'
# plt.savefig(fig_name, format='pdf')


# %% Finetune the selected model

models.unfreeze_model(net, N_unfreeze = 20)  # EN: 20 - 100 - 242  # MN: 158

# history2 = net.fit(dataset, epochs=N_e2, shuffle=True, callbacks=[early_stop])
history2 = net.fit(dataset, epochs=N_e2, validation_data=validation, shuffle=True, callbacks=[early_stop])

# net.save(save_file)
# net.save(save_file, overwrite=True, include_optimizer=True, save_format='h5')

np.save(save_folder + model_name + '_history.npy', history2.history)



# %% Plot curves
L_e = len(history2.history['loss'])
ep = range(1, L_e+1)

# Plot loss curve
plt.figure()
plt.plot(ep, history2.history['loss'], linewidth=2, label='Training loss')
# plt.plot(ep, history2.history['val_loss'], linewidth=2, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
#plt.show()
# fig_name = save_folder + model_name + '_Loss.pdf'
# plt.savefig(fig_name, format='pdf')


# Plot accuracy curve
plt.figure()
plt.plot(ep, history2.history['accuracy'], linewidth=2, label='Training accuracy')
# plt.plot(ep, history2.history['val_accuracy'], linewidth=2, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
#plt.show()
# fig_name = save_folder + model_name + '_Accuracy.pdf'
# plt.savefig(fig_name, format='pdf')



# %% Testing

# del X, y


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, matthews_corrcoef


# Load test set
# test_set = data_folder + 'X_val_augmented.npy'
# test_lab = data_folder + 'y_val_augmented.npy'
# test_set = data_folder + 'X_val_wavelet.npy'
# test_lab = data_folder + 'y_val_augmented.npy'
# test_set = data_folder + 'X_val_original.npy'
test_set = data_folder + 'X_val_wavelet_original.npy'
test_lab = data_folder + 'y_val_original.npy'

Xt = np.load(test_set)
yt = np.load(test_lab)


# Convert in floating point and One-Hot encoding
# Xt = Xt / 255.
# Xt = np.array(Xt, dtype = 'float32')

# yt_cat = to_categorical(yt, 5)
yt_t = np.argmax(yt, axis=1)


# Load the trained model
# net = tf.keras.models.load_model(save_file)
# net = tf.saved_model.load(save_file)


# %% Evaluate the model
# results = net.evaluate(Xt, yt)
# print('Final Loss:', results[0])
# print('Overall Accuracy:', results[1])


# %% Evaluate the model output for test set
y_prob = net.predict(Xt)
y_pred = np.argmax(y_prob, axis=1)


# Evaluating the trained model
acc = accuracy_score(yt_t, y_pred)
pre = precision_score(yt_t, y_pred, average='weighted')
rec = recall_score(yt_t, y_pred, average='weighted')
f1  = f1_score(yt_t, y_pred, average='weighted')
auc = roc_auc_score(yt, y_prob, multi_class='ovo')
mcc = matthews_corrcoef(yt_t, y_pred)


# Printing metrics
print("Overall accuracy: {}%".format(round(100*acc,2)))
print("Precision: {}".format(round(pre,3)))
print("Recall: {}".format(round(rec,3)))
print("F1-score: {}".format(round(f1,3)))
print("AUC: {}".format(round(auc,4)))
print("MCC: {}".format(round(auc,4)))
# print("AUC: {}".format(round(AUC,3)))
print(" ", end='\n')
print("Complete report: ", end='\n')
print(classification_report(yt_t, y_pred))
print(" ", end='\n')



# Showing CM results
labels = ['None', 'Slight', 'Moderate', 'Severe']

cm = confusion_matrix(yt_t, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
disp.plot(cmap='Blues')


# Save results on a text file
res_file = result_folder + 'Results_' + model_name + '.txt'
with open(res_file, 'a') as results:  # save the results in a .txt file
      results.write('-------------------------------------------------------\n')
      results.write('Acc: %s\n' % round(100*acc,2))
      results.write('Pre: %s\n' % round(pre,3))
      results.write('Rec: %s\n' % round(rec,3))
      results.write('F1: %s\n' % round(f1,3))
      results.write('AUC: %s\n' % round(auc,4))
      results.write('MCC: %s\n\n' % round(mcc,4))
      results.write(classification_report(yt_t, y_pred, digits=3))
      results.write('\n\n')
