import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import resample
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.svm import SVC

def list_unf_files(directory):
    unf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_Unf-export.mul") or file.endswith("_unf-export.mul"):
                unf_files.append(os.path.join(root, file))
    return unf_files

def list_lie_files(directory):

    lie_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_Lie-export.mul") or file.endswith("_lie-export.mul"):
                lie_files.append(os.path.join(root, file))
    return lie_files

def list_true_files(directory):
    """
    列出指定目录下的所有true文件路径。
    """
    true_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_True-export.mul") or file.endswith("_true-export.mul"):
                true_files.append(os.path.join(root, file))
    return true_files

def read_mul_file(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # head information
    header_info = lines[0].strip()
    # channel information
    channels = lines[1].strip().split()
    # data information
    data = []
    for line in lines[2:]:
        data.append([float(x) if x != '...' else None for x in line.strip().split()])

    # change the type into DataFrame
    df = pd.DataFrame(data, columns=channels)
    return df, header_info, channels

exp3_directory = "D:\\新建文件夹\\Exp3"

unf_files = list_unf_files(exp3_directory)
lie_files = list_lie_files(exp3_directory)
true_files = list_true_files(exp3_directory)

def all_channel(file, transpose=False):
    all_data = []
    for file_path in file:
        df, _, _ = read_mul_file(file_path)
        num_experiments = len(df) // 614
        num_channels = len(df.columns)

        for i in range(num_experiments):
            experiment_data = []
            for channel in range(num_channels):
                start_index = i * 614
                end_index = start_index + 614
                values_form = df.iloc[start_index:end_index, channel].tolist()
                values = [value for value in values_form]
                experiment_data.append(values[102:512])

            if transpose:
                experiment_data = np.array(experiment_data).T.tolist()

            all_data.append(experiment_data)
    return all_data

unf_data = all_channel(unf_files, transpose=True)
true_data = all_channel(true_files, transpose=True)
lie_data = all_channel(lie_files, transpose=True)

np_unf = np.array(unf_data)
print(np_unf.shape)
np_true = np.array(true_data)
print(np_true.shape)
np_lie = np.array(lie_data)
print(np_lie.shape)

target_samples = 1000

np_unf_resampled = np_unf
np_true_resampled = np_true
np_lie_resampled = np_lie

def downsample_experiments(data, target_samples):

    current_samples = data.shape[0]
    if current_samples < target_samples:
        indices = np.random.choice(current_samples, target_samples, replace=True)
        data = data[indices]
    return data

np_unf_resampled = downsample_experiments(np_unf_resampled, target_samples)
# np_f_resampled = downsample_experiments(f_data, target_samples)
np_true_resampled = downsample_experiments(np_true_resampled, target_samples)
np_lie_resampled = downsample_experiments(np_lie_resampled, target_samples)

print(np_unf_resampled.shape)
# print(np_f_resampled.shape)
print(np_true_resampled.shape)
print(np_lie_resampled.shape)
electrode_indices = [35, 11, 63, 12]

lie_electrodes = np_lie_resampled[:, :, electrode_indices]
unf_electrodes = np_unf_resampled[:, :, electrode_indices]
# f_electrodes = np_f_resampled[:, :, electrode_indices]
true_electrodes = np_true_resampled[:, :, electrode_indices]

print(lie_electrodes.shape)
print(unf_electrodes.shape)
print(true_electrodes.shape)

true_labels = np.ones(len(true_electrodes), dtype=int)
unf_labels = np.zeros(len(unf_electrodes), dtype=int)
lie_labels = np.full(len(true_electrodes), 2, dtype=int)

eeg_data = np.concatenate((true_electrodes, unf_electrodes, lie_electrodes), axis=0)
from scipy.signal import resample
# num_samples = 64
# downsampled_data = resample(eeg_data, num=num_samples, axis=1)
label = np.concatenate((true_labels, unf_labels, lie_labels), axis=0)

def add_noise(data, noise_factor=0.01):
    noise = noise_factor * np.random.normal(size=data.shape)
    return data + noise

augmented_data = add_noise(eeg_data)

print(augmented_data.shape)
print(label.shape)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def standardize_data(data):
    standardized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        sample = data[i]
        sample = scaler.fit_transform(sample)
        standardized_data[i] = sample
    return standardized_data

# 应用数据标准化
data = standardize_data(augmented_data)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools

window_size = 200
sampling_rate = 512
num_samples_per_window = int(window_size / 1000 * sampling_rate)

train_data0, test_data0, train_labels0, test_labels0 = train_test_split(data, label, test_size=0.95, random_state=42)
train_data0 = train_data0.reshape(train_data0.shape[0], 410, 4, 1).astype('float32')

train_length = train_data0.shape[1]
test_length = test_data0.shape[1]

time_windows = range(0, train_length - num_samples_per_window + 1, num_samples_per_window)
average_accuracies = []
std_accuracies = []
classes = ['familiar', 'unfamiliar', 'lie']

for start_idx in time_windows:
    end_idx = start_idx + num_samples_per_window

    window_accuracies = []
    predicted_labels_list = []
    true_labels_list = []

    for _ in range(10):

        train_data, test_data, train_labels, test_labels = train_test_split(test_data0, test_labels0, test_size=0.2, random_state=42)
        train_data = train_data.reshape(train_data.shape[0], 410, 4, 1).astype('float32')
        test_data = test_data.reshape(test_data.shape[0], 410, 4, 1).astype('float32')

        train_labels = to_categorical(train_labels, 3)
        test_labels = to_categorical(test_labels, 3)
        train_window = train_data[:, start_idx:end_idx, :, :]
        test_window = test_data[:, start_idx:end_idx, :, :]

        num_samples = train_window.shape[1]
        model = Sequential()
        model.add(Input(shape=(num_samples, 4, 1)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))  # 使用 padding='same' 保持输入大小
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train_window, train_labels, validation_data=(test_window, test_labels), epochs=40, batch_size=64, verbose=2)

        _, accuracy = model.evaluate(test_window, test_labels, verbose=0)
        window_accuracies.append(accuracy)

        predicted_labels = model.predict(test_window)
        predicted_labels = np.argmax(predicted_labels, axis=1)
        true_labels = np.argmax(test_labels, axis=1)
        predicted_labels_list.append(predicted_labels)
        true_labels_list.append(true_labels)

    average_accuracy = sum(window_accuracies) / len(window_accuracies)
    average_accuracies.append(average_accuracy)
    std_accuracy = np.std(window_accuracies)
    std_accuracies.append(std_accuracy)

    avg_confusion_matrix = sum([confusion_matrix(true_labels_list[i], predicted_labels_list[i]) for i in range(5)]) / 5
    #confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for window {start_idx}-{end_idx}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    thresh = avg_confusion_matrix.max() / 2.
    for i, j in itertools.product(range(avg_confusion_matrix.shape[0]), range(avg_confusion_matrix.shape[1])):
        plt.text(j, i, format(avg_confusion_matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if avg_confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

# avg acc
for i, avg_acc in enumerate(average_accuracies):
    print(f"Average accuracy for window {i + 1}: {avg_acc * 100:.2f}%")

num_windows = len(average_accuracies)

time_labels = ['0-200ms', '200-400ms', '400-600ms', '600-800ms']

plt.figure(figsize=(12, 6))
bars = plt.bar(range(num_windows), [acc * 100 for acc in average_accuracies], color='navy', alpha=0.7, label='Accuracy')

# plt.plot(range(num_windows), [acc * 100 for acc in average_accuracies], color='red', marker='o', label='Trend Line')
plt.errorbar(range(num_windows), [acc * 100 for acc in average_accuracies], yerr=[std * 100 for std in std_accuracies],
             fmt='o', color='black', capsize=5, label='Standard Deviation')
plt.ylim(76, 82)
plt.title('Average Accuracy per Time Window')
plt.xlabel('Time windows (After stimulus)')
plt.ylabel('Accuracy (%)')
plt.xticks(range(num_windows), time_labels)
plt.legend()
plt.show()
