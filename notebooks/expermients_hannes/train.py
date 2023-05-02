#!pip install datasets transformers

import datasets
import numpy as np
from tqdm.notebook import tqdm_notebook
import torch
import math
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Pad, ConvertImageDtype, Lambda, RandomCrop
import matplotlib.pyplot as plt
from PIL import Image

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.min_model = None

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.min_model = model
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

################################# define dataset streaminig #################################
train_dataset = datasets.load_dataset("Hanneseh/custom_image_classifier_data", split="train", streaming=True)
val_dataset = datasets.load_dataset("Hanneseh/custom_image_classifier_data", split="validation", streaming=True)

def count_samples(dataset):
    count = 0
    for _ in tqdm_notebook(dataset):
        count += 1
    return count

# train_dataset_size = count_samples(train_dataset)
train_dataset_size = 13
# val_dataset_size = count_samples(val_dataset)
val_dataset_size = 3
print("train_dataset size: {}".format(train_dataset_size))
print("val_dataset size: {}".format(val_dataset_size))

################################# define the model and parameters #################################

# Set hyperparameters for training
num_epochs = 1
batch_size = 3
learning_rate = 0.001
patience = 5

def collate_fn(examples):
    images, labels = [], []

    image_transform = Compose([
        # apply random square crop with max possible size of the current image, then resize to 375x375
        RandomCrop(375, pad_if_needed=True),
        Resize((375,375)),
        ToTensor(),
        ])

    # Iterate through the examples, apply the image transformation, and append the results
    for example in examples:
        image = image_transform(example['image'])
        label = example['label']
        images.append(image)
        labels.append(label)

        pixel_values = torch.stack(images)
    labels = torch.tensor(labels)

    return {"pixel_values": pixel_values, "label": labels}


train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)

# load and modify the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
stopper = EarlyStopper(patience=patience, min_delta=0)

################################# train the model #################################
tr = model.train()

train_accu = []
train_losses = []

eval_losses=[]
eval_accu=[]

def train(epoch, data):
    print('\nEpoch : %d' % epoch)
    model.train()
    correct = 0
    running_loss = 0
    total = 0
    for batch in tqdm_notebook(data):
        # Move input and label tensors to the device
        inputs = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / train_dataset_size
    accu = 100. * correct / total

    train_accu.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))

def val(data):
    model.eval()
    correct = 0
    running_loss = 0
    total = 0
    with torch.no_grad():
        for batch in data:
            # Move input and label tensors to the device
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / val_dataset_size
    accu = 100. * correct / total

    eval_accu.append(accu)
    eval_losses.append(val_loss)
    print('Val Loss: %.3f | Accuracy: %.3f' % (val_loss, accu))
    return val_loss


for epoch in range(1, num_epochs + 1):
    train(epoch, train_dataloader)
    val_loss = val(val_dataloader)
    PATH = './resnet50_hannes_v2_{}.pth'.format(epoch)
    torch.save(model.state_dict(), PATH)
    if stopper.early_stop(val_loss, model):
        model = stopper.min_model
        break
