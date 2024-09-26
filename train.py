# Imports here
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models import vgg16,resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim, cuda
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


# load data
def load_data(data_dir):
     # Define the directories for data
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir,'test')

    #transform data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    image_datasets = {
        'train': datasets.ImageFolder(train_dir,transform = data_transforms['train']),
        'val': datasets.ImageFolder(valid_dir,transform = data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    }


    class_to_idx = image_datasets['train'].class_to_idx
    class_json= json.dumps(class_to_idx)
    # saving class to idx json for prediction use
    with open("class_to_idx.json", "w") as outfile:
        outfile.write(class_json)

    #dataloader
    dataloaders = {'train':DataLoader(image_datasets['train'],batch_size =32,shuffle=True),
                'val':DataLoader(image_datasets['val'],batch_size=32,shuffle=True),
                'test':DataLoader(image_datasets['test'],batch_size=32,shuffle=True)}
            
    n_classes = len(image_datasets['train'].classes)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders,n_classes,dataset_sizes,class_to_idx

def get_model(model_type,n_of_classes,device,hidden_units):
    if model_type == 'vgg16':
        model = vgg16(pretrained=True)
    elif model_type =='resnet18':
        model = resnet18(pretrained=True)
    else:
        print ("No model available")

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    input_features = model.classifier[0].in_features
    # Define a custom classifier
    model.classifier = nn.Sequential(
        nn.Linear(input_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, n_of_classes),  # 102 flower classes
        nn.LogSoftmax(dim=1)
    )
   
    model.to(device)
    return model


def train_model(model, dataloaders, device, num_epochs,dataset_sizes,criterion,optimizer):
    since = time.time()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print("Training")
            else:
                model.eval()   # Set model to evaluate mode
                print("Evaluating")

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track gradients only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')



def main(args):
    dataloaders,n_classes,dataset_sizes,class_to_idx = load_data(args.data_dir)
    device = torch.device("cuda" if args.gpu else "cpu")
    model = get_model(args.model_type,n_classes,device,args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    train_model(model, dataloaders, device, args.epochs,dataset_sizes,criterion,optimizer)

    # Save checkpoint
    checkpoint = {
        'architecture': args.model_type,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epoch': args.epochs
    }
    torch.save(checkpoint, 'checkpoint.pth')
    
    

if __name__ == "__main__":
    # Argument parsing for user inputs
    parser = argparse.ArgumentParser(description='Train a new network on a dataset of images')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--model_type', type=str, default='vgg16', choices=['vgg16', 'resnet18'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    args = parser.parse_args()
    main(args)


