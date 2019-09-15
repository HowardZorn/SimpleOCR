from model import *
import torch
import torch.nn
import torch.utils.data
import torchvision
import random
import numpy as np
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import *

MODEL_PATH = "./model.pkl"


def example():
    print(len(train_loader))
    X_train, y_train = train_loader.__iter__().__next__()
    img = torchvision.utils.make_grid(X_train, nrow=4)
    img = img.permute(1, 2, 0)
    X_train = Variable(X_train).to(device)
    outputs = model(X_train)
    _, pred = torch.max(outputs, 2)
    pred = pred.transpose(1, 0)
    print(pred)
    print(pred.shape)
    str = batch_decode(pred)
    for i in range(len(str)):
        print(y_train[i], str[i])

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CRNN().to(device)

    model.load_state_dict(torch.load(MODEL_PATH))

    from dataset import *

    train_set = DataSet('./number_area_img', myTransform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=0)
    test_set = DataSet('./number_area_img')

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=0)

    # optimizer = torch.optim.Adam(
    #    model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer = torch.optim.Adam(model.parameters())

    example()
