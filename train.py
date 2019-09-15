from model import *
import torch
import torch.nn
import torch.utils.data
import random
import numpy as np
import os
from torch.autograd import Variable
from utils import *

MODEL_PATH = "./model.pkl"


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.2)
    elif classname.find('BatchNormal') != -1:
        m.weight.data.normal_(1.0, 0.2)
        m.bias.data.fill_(0)


def create_dict():
    weights_init(model)
    torch.save(model.state_dict(), MODEL_PATH)


def train(epochs=5):
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    for i in range(1, epochs + 1):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(i, epochs))
        print("-"*10)
        j = 0
        for data in train_loader:

            X_train, y_train = data
            # X_train, y_train = Variable(X_train).to(
            #     device), Variable(y_train).to(device)
            X_train = Variable(X_train).to(device)

            optimizer.zero_grad()
            outputs = model(X_train)

            target, target_lengths = batch_encode(y_train)

            target = torch.tensor(target)
            target_lengths = torch.tensor(target_lengths)

            input_lengths = torch.full(
                size=(target_lengths.shape[0],), fill_value=21, dtype=torch.long)
            target, target_lengths, input_lengths = target.to(
                device), target_lengths.to(device), input_lengths.to(device)

            j += 1
            print(j, end=' ')
            loss = ctc_loss(outputs, target, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs, 2)
            pred = pred.transpose(1, 0)
            pred = batch_decode(pred)

            running_loss += abs(loss)
            for k in range(len(y_train)):
                if y_train[k] == pred[k]:
                    running_correct += 1
        print()
        print('[%d/%d] Loss: %f, Train Accu: %f%%' %
              (i, epochs, running_loss / len(train_loader), running_correct * 100 / len(train_set)))
        if i % 10 == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            print('%s saved' % MODEL_PATH)


def example():
    print(len(train_loader))
    X_train, y_train = train_loader.__iter__().__next__()
    X_train = Variable(X_train).to(device)
    outputs = model(X_train)
    _, pred = torch.max(outputs, 2)
    pred = pred.transpose(1, 0)
    print(pred)
    print(pred.shape)
    str = batch_decode(pred)
    for i in range(len(str)):
        print(y_train[i], str[i])


if __name__ == '__main__':
    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CRNN().to(device)

    # 检测是否有模型存档
    if not os.path.exists(MODEL_PATH):
        create_dict()

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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    #optimizer = torch.optim.Adam(model.parameters())

    # train(50)
    example()

    # train(500)
    # example()
