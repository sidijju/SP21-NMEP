import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchsummary import summary
from torch.utils.data.dataset import Dataset  # For custom datasets
from data import Data
from rotnet import RotNet
import time
import shutil
import yaml
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        print("loss: ", loss)
    return loss

def validate(val_loader, model, criterion):
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        outputs = model(input)
        predicted = torch.argmax(outputs)
        loss = criterion(outputs, target)
        if i % 10 == 0:
            plt.imshow(input[0])
            plt.show()
            print("loss: ", loss)
            print("label", torch.argmax(target)[0])
            print("predicted: ", predicted[0])
    return loss

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
    torch.save(state, filename)
    #best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
    n_epochs = config["num_epochs"]
    model = RotNet()
    summary(model, (3, 32, 32))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    train_dataset = Data(args.data_dir + "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataset = Data(args.data_dir + "test")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)

    best_loss = 1
    for epoch in range(n_epochs):
     	 #TODO: make your loop which trains and validates. Use the train() func
         train_loss = train(train_loader, model, criterion, optimizer, config["num_epochs"])
         val_loss = validate(val_loader, model, criterion)
     	 #TODO: Save your checkpoint
         best_one = val_loss < best_loss
         save_checkpoint(model.state_dict(), best_one)

if __name__ == "__main__":
    main()
