import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import pandas as pd

import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
from model import ConvNet
import os
from torchvggish_test import vggish
import vggish_input_test

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# attribute_file = 'attribute/train_8_oh.csv'
# data = pd.read_csv(attribute_file, delimiter=',')
# data = np.array(data)
# print(data.shape)


mel_dir = 'mel.npy'
seen_x = np.load(mel_dir)
print(seen_x.shape)
seen_tensor_x = torch.from_numpy(seen_x).float()
# print(seen_tensor_x)
train_dataset = torch.utils.data.TensorDataset(seen_tensor_x, seen_tensor_x)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False)


features = []

count = 0

PATH = 'model.pth'

model = ConvNet(num_classes=29, feature_size=128, is_urban=False)
model.load_state_dict(torch.load(PATH))

model.cuda()
model.eval()

with torch.no_grad():
    for seen_images, _ in train_loader:
        seen_images = seen_images.cuda()
        # seen_labels = seen_labels.cuda()

        seen_features, _ = model.forward(seen_images)
        seen_features = seen_features.cpu().detach().numpy()
        # print(seen_features.shape)
        features.append(seen_features)

features = np.array(np.squeeze(features, axis=1)).astype(float)
print(features.shape)
np.save('features_urban.npy', features)


# model.eval()
# features = []
#
# for line in range(data.shape[0]):
#     count += 1
#     file_name = data[line, 0]
#
#     example = vggish_input_test.wavfile_to_examples(file_name)
#
#     embeddings = model(example)
#     if (line + 1) % 100 == 0:
#         print(embeddings.shape)
#
#
#     embeddings_mean = torch.mean(embeddings, dim=0).detach().numpy()
#     embeddings_mean = np.expand_dims(embeddings_mean, axis=0)
#
#
#     features.append(embeddings_mean)
#     print(count)
#
#
# features = np.array(features).astype(float)
# print(features.shape)


# seen images and labels
# dataroot = 'data_ran8'
# num_classes = 136
# feature_size = 128
# mel_dir = dataroot + '/mel.npy'
# label_dir = dataroot + '/labels.npy'
# seen_x = np.load(mel_dir)
# seen_y = np.load(label_dir)
#
# #seen_x = np.transpose(seen_x, [2, 1, 0])
# print(seen_x.shape, seen_y.shape)
#
# seen_tensor_x = torch.from_numpy(seen_x).float()
# seen_tensor_y = torch.from_numpy(seen_y).long()
#
# train_dataset = torch.utils.data.TensorDataset(seen_tensor_x, seen_tensor_y)
# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False)
#
# seen_f = []
# unseen_f = []
#
# model = vggish()
# model.eval()
#
# example = vggish_input_test.wavfile_to_examples('000002.wav')
#
# embeddings = model(example)
#
# embeddings_mean = torch.mean(embeddings, dim=0)
