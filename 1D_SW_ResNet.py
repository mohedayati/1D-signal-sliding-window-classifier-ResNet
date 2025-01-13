# %%
import time
start_time = time.time()

# %%
import numpy as np
import os
import scipy.io
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv1DTranspose, BatchNormalization, Dropout, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Reshape, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal


# %%
def create_sliding_windows(signal, descriptor, window_size):
    half_window = window_size // 2
    signal_windows = []
    descriptor_windows = []
    indices = []
    for j in range(half_window, len(signal) - half_window):
        signal_window = signal[j - half_window: j + half_window + 1]
        descriptor_window = descriptor[j - half_window: j + half_window + 1]
        signal_windows.append(signal_window)
        descriptor_windows.append(descriptor_window[half_window])  # Use the central descriptor
        indices.append(j)
    return np.array(signal_windows), np.array(descriptor_windows), np.array(indices)

# %%
def reconstruct_signal_and_descriptor(signal_windows, descriptor_windows, indices, signal_length):
    half_window = signal_windows.shape[1] // 2
    reconstructed_signal = np.zeros(signal_length)
    reconstructed_descriptor = np.zeros(signal_length)
    signal_count = np.zeros(signal_length)
    descriptor_count = np.zeros(signal_length)
    
    for signal_window, descriptor, idx in zip(signal_windows, descriptor_windows, indices):
        start_idx = int(idx) - half_window
        end_idx = int(idx) + half_window + 1
        reconstructed_signal[start_idx:end_idx] += signal_window
        reconstructed_descriptor[int(idx)] += descriptor
        signal_count[start_idx:end_idx] += 1
        descriptor_count[int(idx)] += 1
    
    # Avoid division by zero
    signal_count[signal_count == 0] = 1
    descriptor_count[descriptor_count == 0] = 1
    
    # Average overlapping regions
    reconstructed_signal /= signal_count
    reconstructed_descriptor /= descriptor_count
    
    return reconstructed_signal, reconstructed_descriptor

# %%
mat = scipy.io.loadmat('***.mat')  # Datasets have been made publicly available.

dataset = mat['dataset']

# Determine the number of samples and the shape of the individual samples
num_samples = dataset.shape[0]
signal_shape = dataset[0, 0][1, :].shape[0]
descriptor_shape = dataset[0, 0][0, :].shape[0]

# Desired number of samples for the rudimentary model
desired_samples = 5000  # Adjust this number as needed

# Randomly sample the indices
sample_indices = np.random.choice(num_samples, size=desired_samples, replace=False)

for i in range(len(dataset)): # so the nominal condition's descriptor changes from 1 to 0
    dataset[i, 0][0, :] -= 1

# %%
# Parameters for sliding window
window_size = 201  # Choose an odd number to have a center timestep
half_window = window_size // 2

# %%
org_signals = None
org_descriptors = None

for i in sample_indices:
    if org_signals is None and org_descriptors is None:
        org_signals = dataset[i, 0][1, :].reshape(1, -1)
        org_descriptors = dataset[i, 0][0, :].reshape(1, -1)
    else:
        org_signals = np.vstack((org_signals, dataset[i, 0][1, :].reshape(1, -1)))
        org_descriptors = np.vstack((org_descriptors, dataset[i, 0][0, :].reshape(1, -1)))

# %%
signal_windows = np.empty([len(org_signals),
                           create_sliding_windows(org_signals[0], org_descriptors[0], window_size)[0].shape[0],
                           create_sliding_windows(org_signals[0], org_descriptors[0], window_size)[0].shape[1]])
descriptor_windows = np.empty([len(org_descriptors), 
                               create_sliding_windows(org_signals[0], org_descriptors[0], window_size)[1].shape[0]])
indices = np.empty([len(org_descriptors), 
                               create_sliding_windows(org_signals[0], org_descriptors[0], window_size)[1].shape[0]])

for i in range(len(org_signals)):     
    signal_windows[i,:,:], descriptor_windows[i,:], indices[i,:] = create_sliding_windows(org_signals[i], 
                                                                                          org_descriptors[i], 
                                                                                          window_size)

# %%
# Convert descriptors to one-hot encoded format
num_classes = 9  # including the 0 class
descriptor_windows = to_categorical(descriptor_windows, num_classes=9)

# %%
# Split data into training and test sets
signals_train, signals_test, descriptors_train, descriptors_test = train_test_split(signal_windows, 
                                                                                    descriptor_windows, test_size=0.2)

# %%
# Scale signals to be within -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))

# Reshape signals for scaling
signals_train_shape = signals_train.shape
signals_test_shape = signals_test.shape

signals_train = scaler.fit_transform(signals_train.reshape(-1, signals_train.shape[-1])).reshape(signals_train_shape)
signals_test = scaler.transform(signals_test.reshape(-1, signals_test.shape[-1])).reshape(signals_test_shape)

# %%
# Flatten signal windows
signals_train_flat = np.concatenate(signals_train)
signals_test_flat = np.concatenate(signals_test)

# %%
# Flatten descriptor windows
descriptors_train_flat = np.concatenate(descriptors_train)
descriptors_test_flat = np.concatenate(descriptors_test)

# %%
# Add a new axis to make signals shape (num_windows, window_size, 1)
signals_train_flat = signals_train_flat[..., np.newaxis]
signals_test_flat = signals_test_flat[..., np.newaxis]

# ResNet's architecture: 

# %%
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# %%
input_tensor = Input(shape=(window_size, 1))
x = Conv1D(80, kernel_size=7, padding='same', activation='relu')(input_tensor)
x = MaxPooling1D(pool_size=2, padding='same')(x)
x = Conv1D(80, kernel_size=3, padding='same', activation='relu')(x)

# Add residual blocks
for _ in range(3):
    x = residual_block(x, 80)

x = MaxPooling1D(pool_size=2, padding='same')(x)
x = Flatten()(x)
x = Dense(200, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=x)

model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(signals_train_flat, descriptors_train_flat, epochs=15, batch_size=64, validation_data=(signals_test_flat, descriptors_test_flat))
loss, accuracy = model.evaluate(signals_test_flat, descriptors_test_flat)
print(f'Test Accuracy: {accuracy:.2f}')

# %%
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time = elapsed_time/60
print(f"Elapsed Time: {elapsed_time:.2f} minutes")

# %%
# Predict the descriptors using the trained model
predicted_descriptors = model.predict(signals_test_flat)

# %%
descriptor_test_shape = descriptors_test.shape[:-1]

# Convert the predicted descriptors from one-hot encoded format to original class labels
predicted_descriptors_labels = np.argmax(predicted_descriptors, axis=1)

# %%
# Reshape the predictions to match the original shape (if needed)
predicted_descriptors_labels = predicted_descriptors_labels.reshape(descriptor_test_shape)

# %%
# same for the actual test results

descriptors_test = np.argmax(descriptors_test, axis=2)

# %%
# Reconstruct the original signal and descriptor

indices = indices[0:len(signals_test)]

reconstructed_signals_test_true = np.empty([len(signals_test),
                           org_signals.shape[1]])
                                  
reconstructed_descriptors_true = np.empty([len(signals_test),
                           org_signals.shape[1]])

for i in range(len(signals_test)):     
        reconstructed_signals_test_true[i,:], reconstructed_descriptors_true[i,:] = reconstruct_signal_and_descriptor(signals_test[i,: ,:], 
                                                                                     descriptors_test[i,:],
                                                                                     indices[i,:], 
                                                                                     len(org_signals[0]))

# %%
# Reconstruct the original signal and descriptor

indices = indices[0:len(signals_test)]

reconstructed_signals_test_predicted = np.empty([len(signals_test),
                           org_signals.shape[1]])
                                  
reconstructed_descriptors_predicted = np.empty([len(signals_test),
                           org_signals.shape[1]])

for i in range(len(signals_test)):     
        reconstructed_signals_test_predicted[i,:], reconstructed_descriptors_predicted[i,:] = reconstruct_signal_and_descriptor(signals_test[i,: ,:], 
                                                                                     predicted_descriptors_labels[i,:],
                                                                                     indices[i, :], 
                                                                                     len(org_signals[0]))

# %%
# Compare original and reconstructed signals and descriptors
folder_name = 'saved_plots'

# Create the directory, if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

for sample_index in range(30):

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(reconstructed_signals_test_true[sample_index,:], label='Test Signals')
    plt.plot(reconstructed_signals_test_predicted[sample_index,:], label='Reconstructed Predicted Signal', linestyle='--')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_descriptors_true[sample_index,:], label='Test Descriptors')
    plt.plot(reconstructed_descriptors_predicted[sample_index,:], label='Reconstructed Predicted Descriptor', linestyle='--')
    plt.legend()
    plt.savefig(f'{folder_name}/classification_instance_{sample_index}.svg')
    plt.show()

# %%

# Define the folder and file name
folder_name = 'data'
file_name = 'data.mat'
file_path = os.path.join(folder_name, file_name)

# Create the directory if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

# Save the variables in a .mat file
scipy.io.savemat(file_path, {
    'reconstructed_signals_test_true': reconstructed_signals_test_true,
    'reconstructed_descriptors_true': reconstructed_descriptors_true,
    'reconstructed_descriptors_predicted': reconstructed_descriptors_predicted
})

print(f'File saved to {file_path}')

# %%
reconstructed_descriptors_predicted_flat = reconstructed_descriptors_predicted.flatten()
reconstructed_descriptors_true_flat = reconstructed_descriptors_true.flatten()

# %%
# Calculate accuracy
accuracy = np.mean(reconstructed_descriptors_predicted_flat == reconstructed_descriptors_true_flat)
print(f'Accuracy based on winner classes: {accuracy:.2f}')

# %%
# Calculate precision, recall, and F1 score for each class
precision, recall, f1, support = precision_recall_fscore_support(reconstructed_descriptors_true_flat, reconstructed_descriptors_predicted_flat)

# Print the metrics for each class
for i in range(len(precision)):
    print(f'Class {i}:')
    print(f'  Precision: {precision[i]:.2f}')
    print(f'  Recall: {recall[i]:.2f}')
    print(f'  F1 Score: {f1[i]:.2f}')
    print(f'  Support: {support[i]}')

# %%
confusion_mat = confusion_matrix(reconstructed_descriptors_true_flat, reconstructed_descriptors_predicted_flat)

print(f'Confusion Matrix:\n\n {confusion_mat}')


