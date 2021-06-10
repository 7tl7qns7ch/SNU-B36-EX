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
# from model import ConvNet
import os
from torchvggish_test import vggish
import vggish_input_test

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

attribute_file = 'attribute/train_8_linear.csv'
data = pd.read_csv(attribute_file, delimiter=',')
data = np.array(data)
print(data.shape)

count = 0

model = vggish()

model.eval()
features = []

for line in range(data.shape[0]):
    count += 1
    file_name = data[line, 0]

    example = vggish_input_test.wavfile_to_examples(file_name)

    embeddings = model(example)
    if (line + 1) % 100 == 0:
        print(embeddings.shape)


    embeddings_mean = torch.mean(embeddings, dim=0).detach().numpy()
    embeddings_mean = np.expand_dims(embeddings_mean, axis=0)


    features.append(embeddings_mean)
    print(count)


features = np.array(features).astype(float)
print(features.shape)
np.save('features_vggish.npy', features)
