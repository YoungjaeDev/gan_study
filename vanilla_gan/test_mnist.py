#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   test_mnist.py
@Time    :   2023/02/18 20:35:05
@Author  :   youngjae you 
@Version :   1.0
@Contact :   youngjae.you@avikus.ai
@License :   (C)Copyright 2023-2024, youngjae you
@Desc    :   None
'''

#%%

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

random_seed = 123
generator_learning_rate = 0.001
discriminator_learning_rate = 0.001
num_epochs = 100
batch_size = 128
LATENT_DIM = 100
IMG_SHAPE = (1, 28, 28)
IMG_SIZE = 1
for x in IMG_SHAPE:
    IMG_SIZE *= x
# %%
# ToTensor() -> to 0-1 range

train_dataset = datasets.MNIST(root='data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

for images, labels in train_loader:
    print('image batch dimension', images.shape)
    print('label batch dimension', labels.shape)
    break
    
# %%

### Model 

class GAN(torch.nn.Module):
    
    def __init__(self):
        super(GAN, self).__init__()
        
        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, IMG_SIZE),
            nn.Tanh()
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(IMG_SIZE, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        pred = self.discriminator(img)
        return pred.view(-1)
# %%

torch.manual_seed(random_seed)
model = GAN()
model = model.to(device)

optim_gener = torch.optim.Adam(model.generator.parameters(), lr=generator_learning_rate)
optim_discr = torch.optim.Adam(model.discriminator.parameters(), lr=discriminator_learning_rate)

model
# %%

start_time = time.time()

discr_costs = []
gener_costs = []

# k=1
for epoch in range(num_epochs):
    
    model = model.train()
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = (features - 0.5) * 2. # 0-1 -> -1 to 1
        features = features.view(-1, IMG_SIZE).to(device)
        targets = targets.to(device)
        
        valid = torch.ones(targets.size(0)).float().to(device) # real
        fake = torch.zeros(targets.size(0)).float().to(device) # fake
        
        ## train generator
        z = torch.zeros((targets.size(0), LATENT_DIM)).uniform_(-1.0, 1.0).to(device)
        generated_features = model.generator_forward(z)
        
        discr_pred = model.discriminator_forward(generated_features)
        
        gener_loss = F.binary_cross_entropy(discr_pred, valid)
        
        optim_gener.zero_grad()
        gener_loss.backward()
        optim_gener.step()
        
        ## train discriminator
        
        discr_pred_real = model.discriminator_forward(features)
        real_loss = F.binary_cross_entropy(discr_pred_real, valid)
        
        discr_pred_fake = model.discriminator_forward(generated_features.detach())
        fake_loss = F.binary_cross_entropy(discr_pred_fake, fake)
        
        discr_loss = 0.5 * (real_loss + fake_loss)
        
        optim_discr.zero_grad()
        discr_loss.backward()
        optim_discr.step()
        
        discr_costs.append(discr_loss.item())
        gener_costs.append(gener_loss.item())
        
        if not batch_idx % 100:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), gener_loss, discr_loss))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
# %%

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(len(gener_costs)), gener_costs, label="generator_loss")
plt.plot(range(len(discr_costs)), discr_costs, label="disriminator_loss")
plt.legend()
plt.show()

#%%

model.eval()

z = torch.zeros((5, LATENT_DIM)).uniform_(-1.0, 1.0).to(device)
generated_features = model.generator_forward(z)
imgs = generated_features.view(-1, 28, 28)

fig, axes = plt.subplots(1, 5, figsize=(20, 2.5))

for i, ax in enumerate(axes):
    axes[i].imshow(imgs[i].to(torch.device('cpu')).detach(), cmap='binary')
# %%
