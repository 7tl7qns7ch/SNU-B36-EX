import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
from model import ConvNet
import matplotlib.pyplot as plt
import os

import time

# Hyper parameters
num_epochs = 100
num_classes = int(4680/40)
print(num_classes)
batch_size = 100
learning_rate = 0.001
val_split = 0.2
shuffle_dataset = True
random_seed = 42
feature_size = 128
task = 'flo'
pos_mod = 'la'
dataroot = 'train_' + task + '_' + pos_mod

mel_dir = 'mel.npy'
labels_dir = dataroot + '/labels.npy'
trainval_loc_dir = dataroot + '/trainval_loc.npy'

mel = np.load(mel_dir)
labels = np.load(labels_dir).astype(int)
trainval_loc = np.load(trainval_loc_dir).astype(int) - 1

seen_x = mel[trainval_loc, :, :]
seen_y_load = labels[trainval_loc]
seen_y = np.zeros((seen_y_load.shape[0]))

seen_num = np.unique(seen_y_load, axis=0)

for i in range(seen_y_load.shape[0]):
    for j in range(seen_num.shape[0]):
        if seen_y_load[i] == seen_num[j]:
            seen_y[i] = j
            break

print(seen_x.shape, seen_y.shape)

seen_tensor_x = torch.from_numpy(seen_x).float()
seen_tensor_y = torch.from_numpy(seen_y).long()

train_dataset = torch.utils.data.TensorDataset(seen_tensor_x, seen_tensor_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = ConvNet(num_classes=num_classes, feature_size=feature_size)

model.cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

seen_features = []
acc_tr = []

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    end = time.time()
    model.train()
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        _, logits = model.forward(images)
        loss = criterion(logits, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        '''if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            print('Train Accuracy of the model on images: {} %'.format(100 * correct / total))'''''

    acc_tr.append(100 * correct / total)

    print('Epoch[{}/{}], Loss: {:.5f}'.format(epoch + 1, num_epochs, loss.item()))
    print('Train Accuracy: {:.5f} %, Time: {:.4f} s'.format(acc_tr[epoch], time.time() - end))

# Save the model checkpoint
model_path = 'models' + '/model_' + task + '.pth'
print("Model saved")
torch.save(model.state_dict(), model_path)
