# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.lenet5 import LeNet
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # print log
        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                100. * batch_index / len(train_loader), loss.item()))


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    print("start")
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Test')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.5,
        metavar='M',
        help='SGD momentum (default: 0.5)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # device setting
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # data load
    transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
    ])
    ## train data
    train_set = datasets.MNIST(
        root='dataset/MNIST/', train=True, transform=transform, download=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    ## test data
    test_set = datasets.MNIST(
        root='dataset/MNIST/', train=False, transform=transform, download=True)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=True)

    # model call
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)
    ## train and test
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, test_loader, criterion)

    print("end")
