# -*- coding: utf-8 -*-
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import pandas as pd
import os

from PIL import Image

import matplotlib.pyplot as plt

cwd = os.getcwd()
datadir = cwd + "/alzheimers_binary/"
traindir = datadir + 'train/'
validdir = datadir + 'valid/'
testdir = datadir + 'test/'

save_file_name = 'resnet_trained'

batch_size = 128

#Image transformations
image_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])

data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transform),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transform),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transform)
}

#Load into iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
features.shape, labels.shape
n_classes = 4

def get_pretrained_model():

    model = models.resnet50(pretrained=True)

    #Freeze trained layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features

    #Add to end of classifier
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1)
        )

    #Check GPU availability
    if cuda.is_available():
        model = model.to('cuda')

    return model

#Mapping stages to classes for reference

model = get_pretrained_model()

model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

list(model.idx_to_class.items())

#Using Adam because I don't know any better
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2):

    #Early stopping
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    for epoch in range(n_epochs):

        #Logging loss after each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        #Set to train
        model.train()

        for ii, (data, target) in enumerate(train_loader):

            if cuda.is_available():
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)

            #Find loss and backprop, update params
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

            #Find max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            train_acc += accuracy.item() * data.size(0)

            #Print format taken from pytorch website
            print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete.', end='\r')

        #Val!
        else:
            with torch.no_grad():
                model.eval()

                for data, target in valid_loader:

                    if cuda.is_available():
                        data, target = data.cuda(), target.cuda()

                    output = model(data)

                    loss = criterion(output, target)
                    valid_loss += loss.item() * data.size(0)

                    #Max log probability
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean( correct_tensor.type(torch.FloatTensor))

                    valid_acc += accuracy.item() * data.size(0)

                #Avg loss
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                #Avg acc
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                #Save local maxima
                if valid_loss < valid_loss_min:

                    torch.save(model.state_dict(), save_file_name)

                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                else:
                    epochs_no_improve += 1

                    #Early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nTotal epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )

                        # Load maxima
                        model.load_state_dict(torch.load(save_file_name))
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer

    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )

    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history

# model = get_pretrained_model()

model, history = train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    max_epochs_stop=5,
    n_epochs=30
    )

# #History plots
# plt.figure(figsize=(8, 6))
# for c in ['train_loss', 'valid_loss']:
#     plt.plot(
#         history[c], label=c)
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Negative Log')
# plt.title('Loss')
#
# plt.figure(figsize=(8, 6))
# for c in ['train_acc', 'valid_acc']:
#     plt.plot(
#         100 * history[c], label=c)
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Average Acc')
# plt.title('Accuracy')
#
# model, history = train(
#     model,
#     criterion,
#     optimizer,
#     dataloaders['train'],
#     dataloaders['val'],
#     save_file_name=save_file_name,
#     max_epochs_stop=5,
#     n_epochs=30,
#     print_every=2)
