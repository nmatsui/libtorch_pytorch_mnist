#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_train_data
from model import MnistCNNModel
from func import train, test


def parse_args():
    parser = argparse.ArgumentParser(description='train mnist by using CNN')
    parser.add_argument('model_file')
    parser.add_argument('--data_root_path', type=str,
                        default='./data', metavar='root_path_of_data',
                        help='root path to save MNIST data (default: ./data)')
    parser.add_argument('--batch-size', type=int,
                        default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int,
                        default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int,
                        default=12, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float,
                        default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float,
                        default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int,
                        default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int,
                        default=10, metavar='N',
                        help='how many batches to wait before logging')
    return parser.parse_args()


def train_and_test(model,
                   train_loader, test_loader,
                   epochs, lr, momentum, log_interval):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, log_interval)
        test(model, test_loader, criterion)


def main(args):
    torch.manual_seed(args.seed)
    train_loader, test_loader = get_train_data(args.data_root_path,
                                               args.batch_size,
                                               args.test_batch_size)

    model = MnistCNNModel()
    train_and_test(model,
                   train_loader, test_loader,
                   args.epochs, args.lr, args.momentum, args.log_interval)
    torch.save(model.state_dict(), args.model_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
