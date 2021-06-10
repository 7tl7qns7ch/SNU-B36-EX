import numpy as np
import os
import csv
import librosa
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

attribute_file = 'attribute/train_zsl_linear.csv'
data = pd.read_csv(attribute_file, delimiter=',')
data = np.array(data)
print(data.shape)

seen_mels = []
unseen_mels = []
seen_target = []
unseen_target = []
seen_att = []
unseen_att = []

test_seen_loc = []
test_unseen_loc = []
train_loc = []
trainval_loc = []
val_loc = []

seen_loc = []
unseen_loc = []
label = []
att = []
mels = []
count = 0
image_files = []

test_seen_label = []
trainval_label = []

for line in range(data.shape[0]):
    count += 1
    file_name = data[line, 0]

    x, sr = librosa.load(file_name, sr=None, mono=True)

    # parameters
    unit_frequency = 20
    slicing_num = 1
    n_fft = int(sr / slicing_num / unit_frequency)
    hop_length = int(n_fft / 5)

    # zero mean centering, slicing
    x = x - sum(x) / len(x)

    # pre emphasising with digital filter
    x_filter = lfilter([1, -0.95], 1, x)

    mel = librosa.feature.melspectrogram(y=x_filter, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                         n_mels=120, norm=np.inf, fmax=10000, fmin=0)

    mel = mel / np.max(np.max(mel))

    print(line)
    mels.append(mel)

    if data[line, 2] == 1:
        seen_mels.append(mel)
        seen_target.append(data[line, 1])
        seen_loc.append(line)
        #seen_att.append(data[line, 4:].astype(float))

    elif data[line, 2] == 0:
        unseen_mels.append(mel)
        unseen_target.append(data[line, 1])
        unseen_loc.append(line)

    #print(len(seen_features), len(unseen_features))
    label.append(data[line, 1])
    if line % 50 == 1:
        if data[line, 2] == 1:
            seen_att.append(data[line, 4:].astype(float))
            #seen_target.append(data[line, 1])
        elif data[line, 2] == 0:
            unseen_att.append(data[line, 4:].astype(float))
            #unseen_target.append(data[line, 1])

        att.append(data[line, 4:].astype(float))
    if data[line, 3] == 0:
        test_unseen_loc.append(count)
    elif data[line, 3] == 1:
        if count % 5 == 1:
            test_seen_loc.append(count)
            test_seen_label.append(data[line, 1])
        else:
            train_loc.append(count)
            trainval_loc.append(count)
            trainval_label.append(data[line, 1])
    elif data[line, 3] == 2:
        if count % 5 == 1:
            test_seen_loc.append(count)
            test_seen_label.append(data[line, 1])
        else:
            val_loc.append(count)
            trainval_loc.append(count)
            trainval_label.append(data[line, 1])

    image_files.append(data[line, 0])

###### Feature #######
seen_features = np.array(seen_mels)
unseen_features = np.array(unseen_mels)
print(seen_features.shape, unseen_features.shape)
print(seen_features.shape)

np.save('mel_seen.npy', seen_features)
np.save('mel_unseen.npy', unseen_features)


###### location ######
test_seen_loc = np.array(test_seen_loc).astype(float)
test_unseen_loc = np.array(test_unseen_loc).astype(float)
train_loc = np.array(train_loc).astype(float)
trainval_loc = np.array(trainval_loc).astype(float)
val_loc = np.array(val_loc).astype(float)
image_files = np.array(image_files)

np.save('data_for_fg/test_seen_loc.npy', test_seen_loc)
np.save('data_for_fg/test_unseen_loc.npy', test_unseen_loc)
np.save('data_for_fg/train_loc.npy', train_loc)
np.save('data_for_fg/trainval_loc.npy', trainval_loc)
np.save('data_for_fg/val_loc.npy', val_loc)
np.save('data_for_fg/image_files.npy', image_files)

###### features #######
mels = np.array(mels).astype(float)
mels = mels.T
print(mels.shape)
np.save('data_for_fg/features.npy', mels)
random_seed = 462
val_split = 0.2
shuffle_dataset = True

##### train, test locations #####
dataset_size = len(seen_loc)

split = int(np.floor(val_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(seen_loc)
train_indices, val_indices = seen_loc[split:], seen_loc[:split]

train_indices = np.array(train_indices).astype(float) + 1
val_indices = np.array(val_indices).astype(float) + 1
unseen_loc = np.array(unseen_loc).astype(float) + 1

np.save('data_for_fg/train_loc.npy', train_indices)
np.save('data_for_fg/test_seen_loc.npy', val_indices)
np.save('data_for_fg/test_unseen_loc.npy', unseen_loc)

#f.close()

##### For pre-training #####
trainval_label = np.array(trainval_label)
test_seen_label = np.array(test_seen_label)

trainval_uni = np.unique(trainval_label)
test_seen_uni = np.unique(test_seen_label)

print(trainval_label.shape[0], test_seen_label.shape[0])

trainval_encoding = np.zeros((trainval_label.shape[0]))
test_seen_encoding = np.zeros((test_seen_label.shape[0]))

for j in range(trainval_uni.shape[0]):
    for i in range(trainval_label.shape[0]):
        if trainval_label[i] == trainval_uni[j]:
            trainval_encoding[i] = j

    for ii in range(test_seen_label.shape[0]):
        if test_seen_label[ii] == trainval_uni[j]:
            test_seen_encoding[ii] = j

print(trainval_encoding.shape, test_seen_encoding.shape)

np.save('data_for_fg/trainval_class.npy', trainval_encoding)
np.save('data_for_fg/test_seen_class.npy', test_seen_encoding)


##### category number, target and One-hot #####
seen_target = np.array(seen_target).astype(float)
unseen_target = np.array(unseen_target).astype(float)

num = np.unique(np.hstack((seen_target, unseen_target)), axis=0)
seen_num = np.unique(seen_target, axis=0)
unseen_num = np.unique(unseen_target, axis=0)
print(num, seen_num, unseen_num)

seen_encoding = np.zeros((seen_target.shape[0]))
unseen_encoding = np.zeros((unseen_target.shape[0]))

seen_one_hot = np.zeros((seen_target.shape[0], 59))
unseen_one_hot = np.zeros((unseen_target.shape[0], 59))

for i in range(seen_target.shape[0]):
    for j in range(59):
        if seen_target[i] == j + 1:
            seen_one_hot[i, j] = 1
            break

for i in range(unseen_target.shape[0]):
    for j in range(59):
        if unseen_target[i] == j + 1:
            unseen_one_hot[i, j] = 1
            break

print(seen_one_hot.shape, unseen_one_hot.shape)
np.save('data_for_fg/seen_att_one_hot_unique.npy', seen_one_hot)
np.save('data_for_fg/unseen_att_one_hot_unique.npy', unseen_one_hot)

for i in range(seen_target.shape[0]):
    for j in range(seen_num.shape[0]):
        if seen_target[i] == seen_num[j]:
            seen_encoding[i] = j
            break
for i in range(unseen_target.shape[0]):
    for j in range(num.shape[0]):
        if unseen_target[i] == num[j]:
            unseen_encoding[i] = j
            break

np.save('data_for_fg/seen_target.npy', seen_target)
np.save('data_for_fg/unseen_target.npy', unseen_target)

np.save('data_for_fg/seen_class.npy', seen_encoding)
np.save('data_for_fg/unseen_class.npy', unseen_encoding)

seen_att = np.array(seen_att)
unseen_att = np.array(unseen_att)

label = np.array(label).astype(float)
print(label.shape)
np.save('data_for_fg/labels', label)


att = np.array(att)
att = att.T
print(att.shape)

np.save('data_for_fg/att.npy', att)
