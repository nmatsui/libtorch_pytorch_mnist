#!/usr/bin/env python

import argparse
import torch
from model import MnistCNNModel


def parse_args():
    parser = argparse.ArgumentParser(description='predict a handwritten digit')
    parser.add_argument('model_file')
    parser.add_argument('trace_file')
    return parser.parse_args()


def main(args):
    model = MnistCNNModel()
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    trace = torch.jit.trace(model, torch.rand(1, 1, 28, 28))
    trace.save(args.trace_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
