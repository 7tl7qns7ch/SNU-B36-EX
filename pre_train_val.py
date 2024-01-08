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


# Hyper parameters
num_epochs = 100
num_classes = 136
batch_size = 100
learning_rate = 0.001
val_split = 0.2
shuffle_dataset = True
random_seed = 42
feature_size = 128
dataroot = 'data_ran8'

'''seen_x_dir = dataroot + '/SNU36_mel_10K_120_0.95_seen.npy'
seen_y_dir = dataroot + '/seen_class.npy'

seen_x = np.load(seen_x_dir)
seen_y = np.load(seen_y_dir)'''

mel_dir = dataroot + '/mel.npy'
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

dataset_size = seen_x.shape[0]
indices = list(range(dataset_size))
split = int(np.floor(val_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

seen_tensor_x = torch.from_numpy(seen_x).float()
seen_tensor_y = torch.from_numpy(seen_y).long()

train_dataset = torch.utils.data.TensorDataset(seen_tensor_x, seen_tensor_y)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, sampler=val_sampler)

'''print(len(train_indices), len(val_indices))
train_indices = np.array(train_indices).astype(float)
val_indices = np.array(val_indices).astype(float)
train_indices = train_indices + 1
val_indices = val_indices + 1

np.save('train_loc.npy', train_indices)
np.save('test_seen_loc.npy', val_indices)'''

model = ConvNet(num_classes=num_classes, feature_size=feature_size).cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

seen_features = []
acc_tr = []
acc_te = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    total = 0
    correct = 0
    total_val = 0
    correct_val = 0
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

    model.eval()
    for _, (images_val, labels_val) in enumerate(val_loader):
        images_val = images_val.cuda()
        labels_val = labels_val.cuda()

        _, logits_val = model.forward(images_val)
        _, predicted_val = torch.max(logits_val, 1)
        total_val += labels_val.size(0)

        correct_val += (predicted_val == labels_val).sum().item()

    #print(100 * correct_val / total_val)
    acc_tr.append(100 * correct / total)
    acc_te.append(100 * correct_val / total_val)

    print('Epoch[{}/{}], Loss: {:.5f}'.format(epoch + 1, num_epochs, loss.item()))
    print('Train Accuracy: {:.5f} %'.format(acc_tr[epoch]))
    print('Validation Accuracy: {:.5f} %'.format(acc_te[epoch]))


acc_tr = np.array(acc_tr)
acc_te = np.array(acc_te)
x = np.arange(num_epochs)

plt.figure()
plt.plot(x, acc_tr)
plt.plot(x, acc_te)
plt.show()
# Save the model checkpoint
#print("Model saved")
#torch.save(model.state_dict(), 'model.pth')
