import torch.nn as nn
import torch.nn.functional as F
import math
import torch

import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
from model import ConvNet
import os

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# seen images and labels
num_classes = int(5200/40)
feature_size = 128
# ratio = '8'
task = 'randi'
pos_mod = 'la'
dataroot = 'train_' + task + '_' + pos_mod
# dataroot = 'pkg_' + ratio + '_' + pos_mod

mel_dir = 'mel.npy'
labels_dir = dataroot + '/labels' + '.npy'
seen_x = np.load(mel_dir)
seen_y = np.load(labels_dir)

#seen_x = np.transpose(seen_x, [2, 1, 0])
print(seen_x.shape, seen_y.shape)

seen_tensor_x = torch.from_numpy(seen_x).float()
seen_tensor_y = torch.from_numpy(seen_y).long()

train_dataset = torch.utils.data.TensorDataset(seen_tensor_x, seen_tensor_y)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False)

# unseen images and labels
'''unseen_x = np.load('SNU36_mel_10K_120_0.95_unseen.npy')
unseen_y = np.load('unseen_class.npy')

unseen_tensor_x = torch.from_numpy(unseen_x).float()
unseen_tensor_y = torch.from_numpy(unseen_y).long()

test_dataset = torch.utils.data.TensorDataset(unseen_tensor_x, unseen_tensor_y)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)'''

seen_f = []
unseen_f = []

PATH = 'models/' + 'model_' + task + '.pth'

the_model = ConvNet(num_classes=num_classes, feature_size=feature_size)
the_model.load_state_dict(torch.load(PATH))

#print(the_model)
the_model.cuda()
the_model.eval()

with torch.no_grad():
    for seen_images, _ in train_loader:
        seen_images = seen_images.cuda()
        # seen_labels = seen_labels.cuda()

        seen_features, _ = the_model.forward(seen_images)
        seen_features = seen_features.cpu().detach().numpy()
        print(seen_features.shape)
        seen_f.append(seen_features)

    '''for unseen_images, unseen_labels in test_loader:
        unseen_images = unseen_images.cuda()
        unseen_labels = unseen_labels.cuda()

        unseen_features, _ = the_model.forward(unseen_images)
        unseen_features = unseen_features.cpu().detach().numpy()
        print(unseen_features.shape)
        unseen_f.append(unseen_features)'''

seen_f = np.array(np.squeeze(seen_f, axis=1)).astype(float)
print(seen_f.shape)
#np.save('seen_features.npy', seen_f)
if not os.path.exists(dataroot):
    os.makedirs(dataroot)

feature_root = 'features/' + 'features_' + task + '.npy'
np.save(feature_root, seen_f)

'''unseen_f = np.array(np.squeeze(unseen_f, axis=1)).astype(float)
print(unseen_f.shape)'''
#np.save('unseen_features.npy', unseen_f)

