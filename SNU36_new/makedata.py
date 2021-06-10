import numpy as np
import os
import csv
import librosa
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import pandas as pd

attribute_file = 'attribute/train_zsl_linear.csv'
dataroot = 'train_zsl_linear'


if os.path.isdir(dataroot) == False:
    os.makedirs(dataroot)

data = pd.read_csv(attribute_file, delimiter=',')
data = np.array(data)
print(data.shape)

seen_features = []
unseen_features = []
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
features = []
count = 0
image_files = []

test_seen_label = []
trainval_label = []

for line in range(data.shape[0]):
    if data[line, 2] == -1:
        pass
    else:
        count += 1
        file_name = data[line, 0]

        '''x, sr = librosa.load(file_name, sr=None, mono=True)
    
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
    
        mel = mel / np.max(np.max(mel))'''

        print(line)

        #features.append(mel)

        if data[line, 2] == 1:
            #seen_features.append(mel)
            seen_target.append(data[line, 1])
            seen_loc.append(line)
            #seen_att.append(data[line, 4:].astype(float))

        elif data[line, 2] == 0:
            #unseen_features.append(mel)
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
seen_features = np.array(seen_features)
unseen_features = np.array(unseen_features)
print(seen_features.shape, unseen_features.shape)
print(seen_features.shape)

SNU36_mel_seen_dir = dataroot + '/SNU36_mel_10K_120_0.95_seen.npy'
SNU36_mel_unseen_dir = dataroot + '/SNU36_mel_10K_120_0.95_unseen.npy'

np.save(SNU36_mel_seen_dir, seen_features)
np.save(SNU36_mel_unseen_dir, unseen_features)

##### for pre-training #####
seen_target = np.array(seen_target).astype(float)
unseen_target = np.array(unseen_target).astype(float)

num = np.unique(np.hstack((seen_target, unseen_target)), axis=0)
seen_num = np.unique(seen_target, axis=0)
unseen_num = np.unique(unseen_target, axis=0)

seen_encoding = np.zeros((seen_target.shape[0]))
unseen_encoding = np.zeros((unseen_target.shape[0]))

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

seen_target_dir = dataroot + '/seen_target.npy'
unseen_target_dir = dataroot + '/unseen_target.npy'

seen_class_dir = dataroot + '/seen_class.npy'
unseen_class_dir = dataroot + '/unseen_class.npy'

np.save(seen_target_dir, seen_target)
np.save(unseen_target_dir, unseen_target)

np.save(seen_class_dir, seen_encoding)
np.save(unseen_class_dir, unseen_encoding)

###### location ######
test_seen_loc = np.array(test_seen_loc).astype(float)
test_unseen_loc = np.array(test_unseen_loc).astype(float)
train_loc = np.array(train_loc).astype(float)
trainval_loc = np.array(trainval_loc).astype(float)
val_loc = np.array(val_loc).astype(float)
image_files = np.array(image_files)

test_seen_loc_dir = dataroot + '/test_seen_loc.npy'
test_unseen_loc_dir = dataroot + '/test_unseen_loc.npy'
train_loc_dir = dataroot + '/train_loc.npy'
trainval_loc_dir = dataroot + '/trainval_loc.npy'
val_loc_dir = dataroot + '/val_loc.npy'
image_files_dir = dataroot + '/image_files.npy'

np.save(test_seen_loc_dir, test_seen_loc)
np.save(test_unseen_loc_dir, test_unseen_loc)
np.save(train_loc_dir, train_loc)
np.save(trainval_loc_dir, trainval_loc)
np.save(val_loc_dir, val_loc)
np.save(image_files_dir, image_files)

###### features #######
'''features = np.array(features).astype(float)
features = features.T
print(features.shape)
np.save('data_for_fg/features.npy', features)'''
'''random_seed = 462
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

np.save('train_loc.npy', train_indices)
np.save('test_seen_loc.npy', val_indices)
np.save('test_unseen_loc.npy', unseen_loc)'''

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
np.save('seen_att_one_hot_unique.npy', seen_one_hot)
np.save('unseen_att_one_hot_unique.npy', unseen_one_hot)

seen_att = np.array(seen_att)
unseen_att = np.array(unseen_att)

print(seen_att.shape, unseen_att.shape)

np.save('seen_att_spherical_unique.npy', seen_att)
np.save('unseen_att_spherical_unique.npy', unseen_att)

label = np.array(label).astype(float)
print(label.shape)
np.save('labels', label)

##### attribute #####
att_dir = dataroot + '/att.npy'
data_name = dataroot.split('_')
if data_name[-1] == 'oh':
    vec_size = 21
else:
    vec_size = 7
in_att1 = np.zeros([vec_size, 13])
in_att2 = np.zeros([vec_size, 13])

for i in range(13):
    in_att1[:, i] = att[i]
    in_att1[0, i], in_att1[4, i] = 0, 1
    att.insert(i + 156, in_att1[:, i])

    in_att2[:, i] = att[i + 26]
    in_att2[0, i], in_att2[4, i] = 0, 1
    att.append(in_att2[:, i])

att = np.array(att).astype(float)
att = att.T
print(att.shape)


np.save(att_dir, att)


##### all feature ######
mel_dir = dataroot + '/mel.npy'
features = np.array(features).astype(float)
np.save(mel_dir, features)

##### label #####
label_dir = dataroot + '/labels.npy'
label = np.array(label).astype(float)
'''label_uni = np.unique(label)
labels = []
for i in range(label.shape[0]):
    for j in range(label_uni.shape[0]):
        if label[i] == label_uni[j]:
            labels.append(j + 1)
labels = np.array(labels).astype(float)'''
labels = np.array(label).astype(float)
np.save(label_dir, labels)
print(labels)
