#!/usr/bin/env python

import argparse
import cv2
import torch
import numpy as np
from model import MnistCNNModel


def parse_args():
    parser = argparse.ArgumentParser(description='predict a handwritten digit')
    parser.add_argument('model_file', help='trained model file')
    parser.add_argument('image_file', help='test image file')
    return parser.parse_args()


def load_image(image_file):
    raw = cv2.imread(image_file)
    src = cv2.resize(raw, (28, 28))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray)
    data = invert / 255
    data = data.reshape(1, 1, 28, 28).astype(np.float32)
    return data


def main(args):
    smax = torch.nn.Softmax(dim=1)

    data = load_image(args.image_file)

    model = MnistCNNModel()
    model.load_state_dict(torch.load(args.model_file))

    output = smax(model(torch.from_numpy(data)))
    pred = output.argmax(dim=1, keepdim=True)
    label = pred.item()
    prob = output[0][label].item()

    print('label: {} (prob: {:.6f})'.format(label, prob))


if __name__ == '__main__':
    args = parse_args()
    main(args)
