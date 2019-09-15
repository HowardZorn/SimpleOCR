from model import *
import torch
import torch.nn
import torch.utils.data
import random
import numpy as np
import os
from torch.autograd import Variable
from utils import *
from torchvision.transforms import transforms
import PIL
import PIL.Image

MODEL_PATH = "./model.pkl"


def img_open(path):
    data = PIL.Image.open(path)
    height = 32
    width = int(data.size[0] / (data.size[1] / height))
    data = data.resize((width, height))
    Transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.unsqueeze(x, 0))
    ])
    data = Transform(data)
    return data


def predict(img):
    img = Variable(img).to(device)
    outputs = model(img)
    _, pred = torch.max(outputs, 2)
    pred = pred.transpose(1, 0)
    str = batch_decode(pred)
    return str[0]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CRNN().to(device)

    model.load_state_dict(torch.load(MODEL_PATH))

    data = img_open("./test.png")

    str = predict(data)

    print(str)
