# -*- coding: utf-8 -*-
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn

import numpy as np
import pandas as pd
import os

import torch.nn.functional as F
import torch.optim as optim

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
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
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

    model = models.vgg16_bn(pretrained=True)
    print(model)

    #Freeze trained layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[0].in_features

    #Add to end of classifier
    model.classifier = nn.Sequential(
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

model, _ = train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    max_epochs_stop=5,
    n_epochs=4
    )

dataset = datasets.ImageFolder(validdir, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]))

image_means = np.array([0.485, 0.456, 0.406])
image_stds = np.array([0.229, 0.224, 0.225])

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=image_means, std=image_stds)
])

def get_sample(index):
    """Returns the raw image, the transformed image, and the class."""
    raw_image, class_index = dataset[index]
    raw_image = np.array(raw_image)  # convert from PIL to numpy
    image_tensor = input_transform(raw_image)
    return raw_image, image_tensor, class_index


def sensitivity_analysis(model, image_tensor, target_class=None, postprocess='abs'):
    # image_tensor can be a pytorch tensor or anything that can be converted to a pytorch tensor (e.g. numpy, list)

    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    X = Variable(image_tensor[None], requires_grad=True)  # add dimension to simulate batch

    model.eval()
    output = model(X)
    output_class = output.max(1)[1].data.numpy()[0]
    print('Image was classified as:', output_class)

    model.zero_grad()
    one_hot_output = torch.zeros(output.size())
    if target_class is None:
        one_hot_output[0, output_class] = 1
    else:
        one_hot_output[0, target_class] = 1
    output.backward(gradient=one_hot_output)

    relevance_map = X.grad.data[0].numpy()

    if postprocess == 'abs':  # as in Simonyan et al. (2013)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map**2
    elif postprocess is None:
        return relevance_map
    else:
        raise ValueError()

def guided_backprop(model, image_tensor, target_class=None, postprocess='abs'):

    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        if isinstance(module, nn.ReLU):
            return (torch.clamp(grad_in[0], min=0.0),)

    hook_handles = []

    try:
        # Loop through layers, hook up ReLUs with relu_hook_function, store handles to hooks.
        for pos, module in model.features._modules.items():
            if isinstance(module, nn.ReLU):
                hook_handle = module.register_backward_hook(relu_hook_function)
                hook_handles.append(hook_handle)

        # Calculate backprop with modified ReLUs.
        relevance_map = sensitivity_analysis(model, image_tensor, target_class=target_class, postprocess=postprocess)

    finally:
        # Remove hooks from model.
        # The finally clause re-raises any possible exceptions.
        for hook_handle in hook_handles:
            hook_handle.remove()
            del hook_handle

    return relevance_map

image, image_tensor, class_index = get_sample(0)
map = guided_backprop(model, image_tensor, postprocess='abs')
map = Image.fromarray(map)
map.save(cwd + "backprop.jpeg")


fig, axes = plt.subplots(2, 4, figsize=(15, 18))

for i, vertical_axes in enumerate(axes.T):
    image, image_tensor, class_index = get_sample(i)

    plt.sca(vertical_axes[0])
    plt.axis('off')
    plt.imshow(image)

    plt.sca(vertical_axes[1])
    plt.axis('off')
    plt.imshow(sensitivity_analysis(model, image_tensor, postprocess='abs').max(0), cmap='gray')

    plt.sca(vertical_axes[2])
    plt.axis('off')
    plt.imshow(guided_backprop(model, image_tensor, postprocess='abs').max(0), cmap='gray')

plt.savefig('comparison.png')
