import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import matplotlib as mpl
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

#
exp3_directory = "D:\\新建文件夹\\Exp3"

unf_files = list_unf_files(exp3_directory)
lie_files = list_lie_files(exp3_directory)
true_files = list_true_files(exp3_directory)

def unf_file_avg_trials(file_name):
    i=1
    sum=0
    for file in file_name:
        df, header_info, channels = read_mul_file(file)
        num_rows = df.shape[0]
        num_trials = num_rows/614
        i += 1
        sum += num_trials
    avg_unf=sum/i
    return avg_unf

def lie_file_avg_trials(file_name):
    i=1
    sum=0
    for file in file_name:
        df, header_info, channels = read_mul_file(file)
        num_rows = df.shape[0]
        num_trials = num_rows/614
        i += 1
        sum += num_trials
    avg_lie = sum/i
    return avg_lie

def true_file_avg_trials(file_name):
    i=1
    sum=0
    for file in file_name:
        df, header_info, channels = read_mul_file(file)
        num_rows = df.shape[0]
        num_trials = num_rows/614
        i += 1
        sum += num_trials
    avg_true = sum/i
    return avg_true

# avg_trials_unf = unf_file_avg_trials(unf_files)
# avg_trials_lie = lie_file_avg_trials(lie_files)
# avg_trials_true = true_file_avg_trials(true_files)
#
# print("Average number of trials of unfamiliar:", avg_trials_unf)
# print("Average number of trials of lie:", avg_trials_lie)
# print("Average number of trials of true:", avg_trials_true)

def column_data(column_name, files_name):
    student_matrices = []
    all_data = []

    for file_path in files_name:
        df, _, _ = read_mul_file(file_path)
        num_experiments = len(df) // 614
        person_matrix = []
        # 循环处理每次实验
        for i in range(num_experiments):
            start_index = i * 614
            end_index = start_index + 614
            avg_form = df[column_name].iloc[0:102].mean()
            values_form = df[column_name].iloc[start_index:end_index].tolist()
            values = [value - avg_form for value in values_form]
            person_matrix.append(values_form)
            all_data.append(values)
        averages = [sum(values) / len(values) for values in zip(*person_matrix)]
        student_matrices.append(averages)
    column_data = [sum(values) / len(values) for values in zip(*student_matrices)]
    return column_data, student_matrices, all_data

TP9_lie, TP9_lie_all, Part_TP9_lie = column_data('TP9', lie_files)
TP10_lie, TP10_lie_all, Part_TP10_lie = column_data('TP10', lie_files)
P9_lie, P9_lie_all, Part_P9_lie = column_data('P9', lie_files)
P10_lie, P10_lie_all, Part_P10_lie = column_data('P10', lie_files)

TP9_true, TP9_true_all, Part_TP9_true = column_data('TP9', true_files)
TP10_true, TP10_true_all, Part_TP10_true = column_data('TP10', true_files)
P9_true, P9_true_all, Part_P9_true = column_data('P9', true_files)
P10_true, P10_true_all, Part_P10_true = column_data('P10', true_files)

TP9_unf, TP9_unf_all, Part_TP9_unf = column_data('TP9', unf_files)
TP10_unf, TP10_unf_all, Part_TP10_unf = column_data('TP10', unf_files)
P9_unf, P9_unf_all, Part_P9_unf = column_data('P9', unf_files)
P10_unf, P10_unf_all, Part_P10_unf = column_data('P10', unf_files)

FT10_unf,_,_=column_data('FT10',unf_files)
FT10_true,_,_=column_data('FT10',true_files)
FT10_lie,_,_=column_data('FT10',lie_files)

FT9_unf,_,_=column_data('FT9',unf_files)
FT9_true,_,_=column_data('FT9',true_files)
FT9_lie,_,_=column_data('FT9',lie_files)

P8_unf,_,_=column_data('P8',unf_files)
P8_true,_,_=column_data('P8',true_files)
P8_lie,_,_=column_data('P8',lie_files)

O10_unf,_,_=column_data('O10',unf_files)
O10_true,_,_=column_data('O10',true_files)
O10_lie,_,_=column_data('O10',lie_files)
######204-512Hz
sub_part_tp9_lie = [sub_list[203:512] for sub_list in Part_TP9_lie]
sub_part_tp10_lie = [sub_list[203:512] for sub_list in Part_TP10_lie]
sub_part_p9_lie = [sub_list[203:512] for sub_list in Part_P9_lie]
sub_part_p10_lie = [sub_list[203:512] for sub_list in Part_P10_lie]

sub_part_tp9_true = [sub_list[203:512] for sub_list in Part_TP9_true]
sub_part_tp10_true = [sub_list[203:512] for sub_list in Part_TP10_true]
sub_part_p9_true = [sub_list[203:512] for sub_list in Part_P9_true]
sub_part_p10_true = [sub_list[203:512] for sub_list in Part_P10_true]

sub_part_tp9_unf = [sub_list[203:512] for sub_list in Part_TP9_unf]
sub_part_tp10_unf = [sub_list[203:512] for sub_list in Part_TP10_unf]
sub_part_p9_unf = [sub_list[203:512] for sub_list in Part_P9_unf]
sub_part_p10_unf = [sub_list[203:512] for sub_list in Part_P10_unf]
sampling_interval_ms = 1000 / 512

time_points = [(i * sampling_interval_ms - 200) for i in range(len(FT9_lie))]

PO10_1 = [PO10_unf[i] - PO10_true[i] for i in range(len(PO10_unf))]
PO10_2 = [PO10_unf[i] - PO10_lie[i] for i in range(len(PO10_unf))]
PO8_1 = [PO8_unf[i] - PO8_true[i] for i in range(len(PO8_unf))]
PO8_2 = [PO8_unf[i] - PO8_lie[i] for i in range(len(PO8_unf))]
TP10_1 = [TP10_unf[i] - TP10_true[i] for i in range(len(TP10_unf))]
TP10_2 = [TP10_unf[i] - TP10_lie[i] for i in range(len(TP10_unf))]
P10_1 = [P10_unf[i] - P10_true[i] for i in range(len(P10_unf))]
P10_2 = [P10_unf[i] - P10_lie[i] for i in range(len(P10_unf))]

list1 = [(PO10_1[i]+PO8_1[i]+TP10_1[i]+P10_1[i])/4 for i in range(len(PO10_1))]
list2 = [(PO10_2[i]+PO8_2[i]+TP10_2[i]+P10_2[i])/4 for i in range(len(PO10_2))]

plt.figure(figsize=(10, 6))
# plt.plot(time_points, list1, label='Familiarity Effect')
# plt.plot(time_points, list2, label='Concealed Familiarity Effect', color='red')
plt.plot(time_points, P10_lie, label='P10_lie')
plt.plot(time_points, P10_true, color='red', label='P10_true')
plt.plot(time_points, P10_unf, color='black', linestyle='--', label='P10_unf')
plt.xlabel('Time (ms)')
plt.ylabel('μV')
# plt.title('PO10_lie, PO10_true, and PO10_unf Data over Time')
plt.legend()
plt.axhline(0, color='black')
for y_value in [200, 400, 600, 800]:
    plt.axvline(y_value, color='gray', linestyle='--')
plt.show()

##############
def all_channel(file, transpose=False):
    all_data = []
    num_ex = []
    num_file = len(file)
    for file_path in file:
        df, _, _ = read_mul_file(file_path)
        num_experiments = len(df) // 614
        num_ex.append(num_experiments)
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
    mean_ex = sum(num_ex) / num_file
    return all_data, mean_ex, num_ex


# Example usage:
unf_data, unfnum, unf_ex_list = all_channel(unf_files, transpose=True)
true_data, truenum, true_ex_list = all_channel(true_files, transpose=True)
lie_data, lienum, lie_ex_list = all_channel(lie_files, transpose=True)

# 计算标准差、最大值和最小值
unf_std = np.std(unf_ex_list)
true_std = np.std(true_ex_list)
lie_std = np.std(lie_ex_list)

unf_max = np.max(unf_ex_list)
true_max = np.max(true_ex_list)
lie_max = np.max(lie_ex_list)

unf_min = np.min(unf_ex_list)
true_min = np.min(true_ex_list)
lie_min = np.min(lie_ex_list)

# 打印结果
print(f"unfnum: {unfnum}, Standard Deviation: {unf_std}, Max: {unf_max}, Min: {unf_min}")
print(f"truenum: {truenum}, Standard Deviation: {true_std}, Max: {true_max}, Min: {true_min}")
print(f"lienum: {lienum}, Standard Deviation: {lie_std}, Max: {lie_max}, Min: {lie_min}")

np_unf = np.array(unf_data)
print(np_unf.shape)
np_true = np.array(true_data)
print(np_true.shape)
np_lie = np.array(lie_data)
print(np_lie.shape)

# 对每个类别的数据集进行平均处理
avg_unf_data = np.mean(unf_data, axis=0)
avg_true_data = np.mean(true_data, axis=0)
avg_lie_data = np.mean(lie_data, axis=0)

# 输出平均数据集的形状
print(avg_unf_data.shape)  # 应为 (410, 64)
print(avg_true_data.shape)  # 应为 (410, 64)
print(avg_lie_data.shape)  # 应为 (410, 64)

unf_minus_true = avg_unf_data - avg_true_data
unf_minus_lie = avg_unf_data - avg_lie_data
lie_minus_true = avg_lie_data - avg_true_data

print("数据的最小值：", unf_minus_lie.min())
print("数据的最大值：", unf_minus_lie.max())

print("数据的最小值：", lie_minus_true.min())
print("数据的最大值：", lie_minus_true.max())

print("数据的最小值：", unf_minus_true.min())
print("数据的最大值：", unf_minus_true.max())
print(unf_minus_true.shape)


ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'P9', 'FC1', 'FC2', 'P10', 'FT9', 'T7', 'C3', 'Cz', 'C4', 'T8', 'FT10', 'PO9', 'CP1', 'CP2', 'PO10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2', 'TP9', 'AF3', 'AF4', 'TP10', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'O9', 'PO3', 'PO4', 'O10', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8']

sfreq = 512
info_custom = mne.create_info(ch_names=ch_names, ch_types='eeg', sfreq=sfreq)


montage = mne.channels.make_standard_montage('standard_1020')
info_custom.set_montage(montage)
evoked = mne.EvokedArray(unf_minus_true.T, info_custom, tmin=-0.1)
fig, ax = plt.subplots(1, 4, figsize=(10, 5),gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
kwargs1 = dict(times=0.16, show=False, show_names=True, average= 0.04)
kwargs2 = dict(times=0.3, show=False, show_names=True, average= 0.2)
kwargs3 = dict(times=0.5, show=False, show_names=True, average= 0.2)
evoked.plot_topomap(axes=ax[0], colorbar=False, **kwargs1, size=0.2, extrapolate='head')
evoked.plot_topomap(axes=ax[1], colorbar=False, **kwargs2, size=0.2, extrapolate='head')
evoked.plot_topomap(axes=ax[2], colorbar=False, **kwargs3, size=0.2, extrapolate='head')
vmin = -3
vmax = 4.4
cmap = mpl.cm.get_cmap('RdBu_r')
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig.colorbar(sm, cax=ax[-1])
ax[-1].set_ylabel('Custom Colorbar Label')
for ax, title in zip(ax[:3], ['N170', 'N250', 'SFE']):
    ax.set_title(title)
fig.suptitle('Familiarity Effect', fontsize=20)
plt.tight_layout()
plt.show()

evoked1 = mne.EvokedArray(unf_minus_lie.T, info_custom, tmin=-0.1)
fig, ax = plt.subplots(1, 4, figsize=(10, 5),gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
kwargs1 = dict(times=0.16, show=False, show_names=True, average= 0.04)
kwargs2 = dict(times=0.3, show=False, show_names=True, average= 0.2)
kwargs3 = dict(times=0.5, show=False, show_names=True, average= 0.2)
evoked1.plot_topomap(axes=ax[0], colorbar=False, **kwargs1, size=0.2, extrapolate='head')
evoked1.plot_topomap(axes=ax[1], colorbar=False, **kwargs2, size=0.2, extrapolate='head')
evoked1.plot_topomap(axes=ax[2], colorbar=False, **kwargs3, size=0.2, extrapolate='head')
vmin = -2.2
vmax = 3.4
cmap = mpl.cm.get_cmap('RdBu_r')
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig.colorbar(sm, cax=ax[-1])
ax[-1].set_ylabel('Custom Colorbar Label')
for ax, title in zip(ax[:3], ['N170', 'N250', 'SFE']):
    ax.set_title(title)
fig.suptitle('Concealed Familiarity Effect', fontsize=20)
plt.tight_layout()
plt.show()

evoked2 = mne.EvokedArray(lie_minus_true.T, info_custom, tmin=-0.1)
fig, ax = plt.subplots(1, 4, figsize=(10, 5),gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
kwargs1 = dict(times=0.16, show=False, show_names=True, average= 0.04)
kwargs2 = dict(times=0.3, show=False, show_names=True, average= 0.2)
kwargs3 = dict(times=0.5, show=False, show_names=True, average= 0.2)
evoked2.plot_topomap(axes=ax[0], colorbar=False, **kwargs1, size=0.2, extrapolate='head')
evoked2.plot_topomap(axes=ax[1], colorbar=False, **kwargs2, size=0.2, extrapolate='head')
evoked2.plot_topomap(axes=ax[2], colorbar=False, **kwargs3, size=0.2, extrapolate='head')
vmin = -1.3
vmax = 1.9
cmap = mpl.cm.get_cmap('RdBu_r')
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig.colorbar(sm, cax=ax[-1])
ax[-1].set_ylabel('Custom Colorbar Label')
for ax, title in zip(ax[:3], ['N170', 'N250', 'SFE']):
    ax.set_title(title)
fig.suptitle('The difference between familiarity & lie', fontsize=20)
plt.tight_layout()
plt.show()

#################

true_labels = np.ones(len(np_true), dtype=int)
unf_labels = np.zeros(len(np_unf), dtype=int)
lie_labels = np.full(len(np_lie), 2, dtype=int)

eeg_data = np.concatenate((np_true, np_unf, np_lie), axis=0)
labels = np.concatenate((true_labels, unf_labels, lie_labels), axis=0)
b,c,a=read_mul_file("D:\\新建文件夹\\Exp3\\Part02_Lie-export.mul")

print(a)

channel_to_index = {channel: idx for idx, channel in enumerate(a)}

regions = {
    "Region 1": ['Fp1',"AF3'",'F7','F5','F3','F1','FT9','FT7','FC3','FC1','T7','C5','C3','C1'],
    "Region 2": ['Fp2',"AF4'",'F2','F4','F6','F8','FC2','FC4','FT8','FT10','C2','C4','C6','T8'],
    "Region 3": ['TP9','TP7','CP3','CP1','P9','P7','P5','P3','P1','PO7',"PO3'",'PO9','O1','O9'],
    "Region 4": ['CP2','CP4','TP8','TP10','P2','P4','P6','P8','P10',"PO4'",'PO8','PO10','O2','O10']
}

region_indices = {region: [channel_to_index[channel] for channel in channels] for region, channels in regions.items()}
region_variances = {}
for region, indices in region_indices.items():
    if len(indices) > 0:
        all_region_variances = []
        for channel_idx in indices:
            channel_data = eeg_data[:, channel_idx, :]
            variances = []
            for label in np.unique(labels):
                variances.append(np.var(channel_data[labels == label], axis=0))
            all_region_variances.append(np.mean(variances))
        region_variances[region] = np.mean(all_region_variances)
    else:
        region_variances[region] = 0
sorted_regions = sorted(region_variances.items(), key=lambda x: x[1], reverse=True)
print("Sorted regions by variance (descending order):")
for region, variance in sorted_regions:
    print(f"{region}: Variance = {variance}")

best_region = sorted_regions[0][0]
print(f"\nBest region for classification: {best_region}")


regions_names, variances = zip(*sorted_regions)
plt.bar(regions_names, variances, color='skyblue')
plt.xlabel('Regions')
plt.ylabel('Variance')
plt.title('Variance of each region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


avg_unf_data = np.mean(unf_data, axis=0)
avg_true_data = np.mean(true_data, axis=0)
avg_lie_data = np.mean(lie_data, axis=0)

unf_minus_true = avg_unf_data - avg_true_data
unf_minus_lie = avg_unf_data - avg_lie_data
lie_minus_true = avg_lie_data - avg_true_data

l1 = np.ones(len(unf_minus_true), dtype=int)
l2 = np.zeros(len(unf_minus_lie), dtype=int)
l3 = np.full(len(lie_minus_true), 2, dtype=int)

eeg_data0 = np.concatenate((unf_minus_true, unf_minus_lie, lie_minus_true), axis=0)
labels0 = np.concatenate((l1, l2, l3), axis=0)

region4_channels = [channel_to_index[channel] for channel in regions['Region 4']]

region4_data = eeg_data0[:, region4_channels]

channel_variances = []
for channel in range(region4_data.shape[1]):
    channel_data = region4_data[:, channel]
    variances = []
    for label in np.unique(labels0):
        variances.append(np.var(channel_data[labels == label], axis=0))
    channel_variances.append(np.mean(variances))

for i, variance in enumerate(channel_variances):
    print(f"Channel {regions['Region 4'][i]}: Variance = {variance}")

sorted_indices = np.argsort(channel_variances)[::-1]
sorted_variances = [channel_variances[i] for i in sorted_indices]
sorted_channel_names = [regions['Region 4'][i] for i in sorted_indices]

N = 4
selected_channels = sorted_indices[:N]

print("Selected channels:", selected_channels)

for i in selected_channels:
    print(regions['Region 4'][i])

plt.figure(figsize=(12, 8))
plt.bar(range(len(sorted_variances)), sorted_variances, color='blue')
plt.xticks(range(len(sorted_variances)), sorted_channel_names, rotation=90)
plt.xlabel('Channels')
plt.ylabel('Variance')
plt.title('Variance of Each EEG Channel in Region 4 (Sorted)')
plt.tight_layout()
plt.show()

#############################
unf_data1 = np.array(unf_data)
true_data1 = np.array(true_data)
lie_data1 = np.array(lie_data)

print(unf_data1.shape)
print(true_data1.shape)
print(lie_data1.shape)

electrode_indices = [35, 11, 57, 27]

lie_electrodes = lie_data1[:, :, electrode_indices]
unf_electrodes = unf_data1[:, :, electrode_indices]
true_electrodes = true_data1[:, :, electrode_indices]

print(lie_electrodes.shape)
print(unf_electrodes.shape)
# print(f_electrodes.shape)
print(true_electrodes.shape)

N170_lie = lie_electrodes[:,72:93,:]
N250_lie = lie_electrodes[:,103:205,:]
SFE_lie = lie_electrodes[:,205:308,:]

N170_unf = unf_electrodes[:,72:93,:]
N250_unf = unf_electrodes[:,103:205,:]
SFE_unf = unf_electrodes[:,205:308,:]

N170_true = true_electrodes[:,72:93,:]
N250_true = true_electrodes[:,103:205,:]
SFE_true = true_electrodes[:,205:308,:]

mean_values_N170_unf = []

for person_data in N170_unf:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_N170_unf.append(person_mean)

mean_unf_N170 = np.array(mean_values_N170_unf)

mean_values_N250_unf = []
for person_data in N250_unf:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_N250_unf.append(person_mean)

mean_unf_N250 = np.array(mean_values_N250_unf)

mean_values_SFE_unf = []

for person_data in SFE_unf:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_SFE_unf.append(person_mean)

mean_unf_SFE = np.array(mean_values_SFE_unf)

mean_values_N170_lie = []
for person_data in N170_lie:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_N170_lie.append(person_mean)

mean_lie_N170 = np.array(mean_values_N170_lie)
mean_values_N250_lie = []

for person_data in N250_lie:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_N250_lie.append(person_mean)

mean_lie_N250 = np.array(mean_values_N250_lie)

mean_values_SFE_lie = []
for person_data in SFE_lie:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_SFE_lie.append(person_mean)
mean_lie_SFE = np.array(mean_values_SFE_lie)

mean_values_N170_true = []

for person_data in N170_true:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_N170_true.append(person_mean)
mean_true_N170 = np.array(mean_values_N170_true)
mean_values_N250_true = []

for person_data in N250_true:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_N250_true.append(person_mean)

mean_true_N250 = np.array(mean_values_N250_true)

mean_values_SFE_true = []

for person_data in SFE_true:
    person_mean = np.mean(person_data, axis=0)
    person_mean = np.mean(person_mean)
    mean_values_SFE_true.append(person_mean)

mean_true_SFE = np.array(mean_values_SFE_true)

print(mean_unf_N170)
print(mean_lie_N170)
print(mean_true_N170)

print(mean_unf_N250)
print(mean_lie_N250)
print(mean_true_N250)

print(mean_unf_SFE)
print(mean_lie_SFE)
print(mean_true_SFE)

import matplotlib.pyplot as plt
N170_unf_data = mean_unf_N170.flatten()
N170_lie_data = mean_lie_N170.flatten()
N170_true_data = mean_true_N170.flatten()

N250_unf_data = mean_unf_N250.flatten()
N250_lie_data = mean_lie_N250.flatten()
N250_true_data = mean_true_N250.flatten()

SFE_unf_data = mean_unf_SFE.flatten()
SFE_lie_data = mean_lie_SFE.flatten()
SFE_true_data = mean_true_SFE.flatten()

labels = ['unf', 'lie', 'true']

plt.figure(figsize=(12, 18))

plt.subplot(3, 2, 1)
plt.gca().cla()
bplot1 = plt.boxplot([N170_unf_data, N170_lie_data, N170_true_data], labels=labels, showfliers=True, showmeans=False)
plt.title('N170 (140-180ms)')
plt.ylabel('μV')
plt.xlabel('Groups')

plt.scatter([1] * len(N170_unf_data), N170_unf_data, marker='o', facecolors='none', edgecolors='blue', label='unf data')
plt.scatter([2] * len(N170_lie_data), N170_lie_data, marker='^', facecolors='none', edgecolors='green', label='lie data')
plt.scatter([3] * len(N170_true_data), N170_true_data, marker='s', facecolors='none', edgecolors='red', label='true data')

plt.subplot(3, 2, 3)
plt.gca().cla()
bplot2 = plt.boxplot([N250_unf_data, N250_lie_data, N250_true_data], labels=labels, showfliers=True, showmeans=False)
plt.title('N250 (200-400ms)')
plt.ylabel('μV')
plt.xlabel('Groups')

plt.scatter([1] * len(N250_unf_data), N250_unf_data, marker='o', facecolors='none', edgecolors='blue', label='unf data')
plt.scatter([2] * len(N250_lie_data), N250_lie_data, marker='^', facecolors='none', edgecolors='green', label='lie data')
plt.scatter([3] * len(N250_true_data), N250_true_data, marker='s', facecolors='none', edgecolors='red', label='true data')

plt.subplot(3, 2, 5)
bplot3 = plt.boxplot([SFE_unf_data, SFE_lie_data, SFE_true_data], labels=labels, showfliers=True, showmeans=False)
plt.title('SFE (400-600ms)')
plt.ylabel('μV')
plt.xlabel('Groups')

plt.scatter([1] * len(SFE_unf_data), SFE_unf_data, marker='o', facecolors='none', edgecolors='blue', label='unf data')
plt.scatter([2] * len(SFE_lie_data), SFE_lie_data, marker='^', facecolors='none', edgecolors='green', label='lie data')
plt.scatter([3] * len(SFE_true_data), SFE_true_data, marker='s', facecolors='none', edgecolors='red', label='true data')
familiarity_effect_N170 = []
familiarity_effect_N250 = []
familiarity_effect_SFE = []

for i in range(len(mean_unf_N170)):
    unf_minus_true_N170 = mean_unf_N170[i] - mean_true_N170[i]
    lie_minus_true_N170 = mean_lie_N170[i] - mean_true_N170[i]
    lie_minus_unf_N170 = mean_unf_N170[i] - mean_lie_N170[i]
    familiarity_effect_N170.append((unf_minus_true_N170, lie_minus_true_N170, lie_minus_unf_N170))
familiarity_effect_N170 = np.array(familiarity_effect_N170)

for i in range(len(mean_unf_N250)):
    unf_minus_true_N250 = mean_unf_N250[i] - mean_true_N250[i]
    lie_minus_true_N250 = mean_lie_N250[i] - mean_true_N250[i]
    lie_minus_unf_N250 = mean_unf_N250[i] - mean_lie_N250[i]
    familiarity_effect_N250.append((unf_minus_true_N250, lie_minus_true_N250, lie_minus_unf_N250))
familiarity_effect_N250 = np.array(familiarity_effect_N250)

for i in range(len(mean_unf_SFE)):
    unf_minus_true_SFE = mean_unf_SFE[i] - mean_true_SFE[i]
    lie_minus_true_SFE = mean_lie_SFE[i] - mean_true_SFE[i]
    lie_minus_unf_SFE = mean_unf_SFE[i] - mean_lie_SFE[i]
    familiarity_effect_SFE.append((unf_minus_true_SFE, lie_minus_true_SFE, lie_minus_unf_SFE))
familiarity_effect_SFE = np.array(familiarity_effect_SFE)

print(familiarity_effect_N170)
print(familiarity_effect_N170[:,0])
print(familiarity_effect_N250.shape)
print(familiarity_effect_SFE.shape)

unf_true_N170 = familiarity_effect_N170[:,0]
lie_true_N170 = familiarity_effect_N170[:,1]
lie_unf_N170 = familiarity_effect_N170[:,2]

unf_true_N250 = familiarity_effect_N250[:,0]
lie_true_N250 = familiarity_effect_N250[:,1]
lie_unf_N250 = familiarity_effect_N250[:,2]

unf_true_SFE = familiarity_effect_SFE[:,0]
lie_true_SFE = familiarity_effect_SFE[:,1]
lie_unf_SFE = familiarity_effect_SFE[:,2]

print(unf_true_N170.shape)
print(lie_true_N170.shape)
print(lie_unf_N170.shape)

print(unf_true_N250.shape)
print(lie_true_N250.shape)
print(lie_unf_N250.shape)

print(unf_true_SFE.shape)
print(lie_true_SFE.shape)
print(lie_unf_SFE.shape)

N170_unf_data = unf_true_N170.flatten()
N170_lie_data = lie_true_N170.flatten()
N170_true_data = lie_unf_N170.flatten()

N250_unf_data = unf_true_N250.flatten()
N250_lie_data = lie_true_N250.flatten()
N250_true_data = lie_unf_N250.flatten()

SFE_unf_data = unf_true_SFE.flatten()
SFE_lie_data = lie_true_SFE.flatten()
SFE_true_data = lie_unf_SFE.flatten()

labels = ['unf-true', 'lie-true', 'unf-lie']

plt.subplot(3, 2, 2)
plt.gca().cla()
bplot1 = plt.boxplot([N170_unf_data, N170_lie_data, N170_true_data], labels=labels, showfliers=True, showmeans=False)
plt.title('N170 (140-180ms)')
plt.ylabel('μV')
plt.xlabel('Groups')

plt.scatter([1] * len(N170_unf_data), N170_unf_data, marker='o', facecolors='none', edgecolors='blue', label='unf data')
plt.scatter([2] * len(N170_lie_data), N170_lie_data, marker='^', facecolors='none', edgecolors='green', label='lie data')
plt.scatter([3] * len(N170_true_data), N170_true_data, marker='s', facecolors='none', edgecolors='red', label='true data')

plt.subplot(3, 2, 4)
plt.gca().cla()
bplot2 = plt.boxplot([N250_unf_data, N250_lie_data, N250_true_data], labels=labels, showfliers=True, showmeans=False)
plt.title('N250 (200-400ms)')
plt.ylabel('μV')
plt.xlabel('Groups')

plt.scatter([1] * len(N250_unf_data), N250_unf_data, marker='o', facecolors='none', edgecolors='blue', label='unf data')
plt.scatter([2] * len(N250_lie_data), N250_lie_data, marker='^', facecolors='none', edgecolors='green', label='lie data')
plt.scatter([3] * len(N250_true_data), N250_true_data, marker='s', facecolors='none', edgecolors='red', label='true data')

plt.subplot(3, 2, 6)
bplot3 = plt.boxplot([SFE_unf_data, SFE_lie_data, SFE_true_data], labels=labels, showfliers=True, showmeans=False)
plt.title('SFE (400-600ms)')
plt.ylabel('μV')
plt.xlabel('Groups')

plt.scatter([1] * len(SFE_unf_data), SFE_unf_data, marker='o', facecolors='none', edgecolors='blue', label='unf data')
plt.scatter([2] * len(SFE_lie_data), SFE_lie_data, marker='^', facecolors='none', edgecolors='green', label='lie data')
plt.scatter([3] * len(SFE_true_data), SFE_true_data, marker='s', facecolors='none', edgecolors='red', label='true data')

plt.tight_layout()
plt.show()
#################
true_labels = np.ones(len(true_electrodes), dtype=int)
unf_labels = np.zeros(len(unf_electrodes), dtype=int)
lie_labels = np.full(len(lie_electrodes), 2, dtype=int)

eeg_data = np.concatenate((true_electrodes, unf_electrodes, lie_electrodes), axis=0)
from scipy.signal import resample
num_samples = 64
downsampled_data = resample(eeg_data, num=num_samples, axis=1)
label = np.concatenate((true_labels, unf_labels, lie_labels), axis=0)


def add_noise(data, noise_factor=0.01):
    noise = noise_factor * np.random.normal(size=data.shape)
    return data + noise

augmented_data = add_noise(downsampled_data)

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

data = standardize_data(augmented_data)

train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

train_data = train_data.reshape(train_data.shape[0], 64, 4, 1).astype('float32')
test_data = test_data.reshape(test_data.shape[0], 64, 4, 1).astype('float32')

train_labels = to_categorical(train_labels, 3)
test_labels = to_categorical(test_labels, 3)

print(train_data.shape)
print(train_labels.shape)

from sklearn.model_selection import KFold

def create_model():
    model = Sequential()
    model.add(Input(shape=(64, 4, 1)))
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model)
    return model

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cvscores = []

for train_index, val_index in kf.split(train_data):

    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    model = create_model()

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=64, verbose=2)

    scores = model.evaluate(X_val, y_val, verbose=0)
    # print(f"Fold accuracy: {scores[1] * 100:.2f}%")
    cvscores.append(scores[1] * 100)

print(f"Average accuracy: {np.mean(cvscores):.2f}% (+/- {np.std(cvscores):.2f}%)")

final_model = create_model()
final_model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=50, batch_size=500, verbose=2)
final_scores = final_model.evaluate(test_data, test_labels, verbose=0)
print(f"Final model accuracy on test data: {final_scores[1] * 100:.2f}%")

predictions = final_model.predict(test_data)
predictions = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['familiar', 'unfamiliar', 'lie/conceal'])

disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of CNN')
plt.show()

print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=['familiar', 'unfamiliar', 'lie/conceal']))


########################RF

train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=42)


def cross_validate_rf(data, labels, num_folds=10):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cvscores = []

    for train_index, val_index in kf.split(data):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)

        rf_pred = rf_clf.predict(X_val.reshape(X_val.shape[0], -1))

        acc = accuracy_score(y_val, rf_pred)
        cvscores.append(acc * 100)

    print(f"Average Random Forest accuracy: {np.mean(cvscores):.2f}% (+/- {np.std(cvscores):.2f}%)")


cross_validate_rf(train_data, train_labels)

def evaluate_final_rf_model(data, labels, test_data, test_labels):
    final_rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    final_rf_clf.fit(data.reshape(data.shape[0], -1), labels)
    rf_pred = final_rf_clf.predict(test_data.reshape(test_data.shape[0], -1))

    test_labels_flat = np.ravel(test_labels)
    rf_pred_flat = np.ravel(rf_pred)

    final_acc = accuracy_score(test_labels_flat, rf_pred_flat)
    final_cm = confusion_matrix(test_labels_flat, rf_pred_flat)

    print(f"Final Random Forest model accuracy on test data: {final_acc * 100:.2f}%")
    print("Final Random Forest model confusion matrix:")
    print(final_cm)

    print("Classification Report:")
    print(classification_report(test_labels_flat, rf_pred_flat, target_names=['familiar', 'unfamiliar', 'lie/conceal']))

    return final_cm

b = evaluate_final_rf_model(train_data, train_labels, test_data, test_labels)

class_names = ['familiar', 'unfamiliar', 'lie/conceal']
disp = ConfusionMatrixDisplay(confusion_matrix=b, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Random Forest')
plt.show()

def preprocess_data(eeg_data, labels, num_samples=64):
    downsampled_data = resample(eeg_data, num=num_samples, axis=1)
    augmented_data = add_noise(downsampled_data)
    standardized_data = standardize_data(augmented_data)
    return standardized_data, labels

def cross_validate_svm(data, labels, num_folds=10):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cvscores = []

    for train_index, val_index in kf.split(data):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        svm_clf = SVC(kernel='rbf', random_state=42)
        svm_clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)

        svm_pred = svm_clf.predict(X_val.reshape(X_val.shape[0], -1))

        acc = accuracy_score(y_val, svm_pred)
        cvscores.append(acc * 100)

    print(f"Average accuracy: {np.mean(cvscores):.2f}% (+/- {np.std(cvscores):.2f}%)")

eeg_data_1 = np.concatenate((unf_electrodes, true_electrodes), axis=0)
labels_1 = np.concatenate((unf_labels, true_labels), axis=0)
data_1, labels_1 = preprocess_data(eeg_data_1, labels_1)
train_data_1, test_data_1, train_labels_1, test_labels_1 = train_test_split(data_1, labels_1, test_size=0.2, random_state=42)
cross_validate_svm(train_data_1, train_labels_1)

eeg_data_2 = np.concatenate((unf_electrodes, lie_electrodes), axis=0)
labels_2 = np.concatenate((unf_labels, lie_labels), axis=0)
data_2, labels_2 = preprocess_data(eeg_data_2, labels_2)
train_data_2, test_data_2, train_labels_2, test_labels_2 = train_test_split(data_2, labels_2, test_size=0.2, random_state=42)
cross_validate_svm(train_data_2, train_labels_2)

eeg_data_3 = np.concatenate((true_electrodes, lie_electrodes), axis=0)
labels_3 = np.concatenate((true_labels, lie_labels), axis=0)
data_3, labels_3 = preprocess_data(eeg_data_3, labels_3)
train_data_3, test_data_3, train_labels_3, test_labels_3 = train_test_split(data_3, labels_3, test_size=0.2, random_state=42)
cross_validate_svm(train_data_3, train_labels_3)

def evaluate_final_model(data, labels, test_data, test_labels, title, ax):
    final_svm_clf = SVC(kernel='rbf', random_state=42)
    final_svm_clf.fit(data.reshape(data.shape[0], -1), labels)
    svm_pred = final_svm_clf.predict(test_data.reshape(test_data.shape[0], -1))
    final_acc = accuracy_score(test_labels, svm_pred)
    final_cm = confusion_matrix(test_labels, svm_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=final_cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    ax.set_title(f"{title}\nAccuracy: {final_acc * 100:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

evaluate_final_model(train_data_1, train_labels_1, test_data_1, test_labels_1, "Task 1: Unfamiliar vs True", axes[0])
evaluate_final_model(train_data_2, train_labels_2, test_data_2, test_labels_2, "Task 2: Unfamiliar vs Lie", axes[1])
evaluate_final_model(train_data_3, train_labels_3, test_data_3, test_labels_3, "Task 3: True vs Lie", axes[2])

plt.tight_layout()
plt.show()
######################
