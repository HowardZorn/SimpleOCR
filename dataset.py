import torch.utils.data
import PIL.Image
import os
import re
import random
import numpy as np
import cv2
from torchvision.transforms import transforms


def add_noise(img):
    for i in range(20):
        temp_color = np.random.randint(0, 255) / 255
        temp_x = np.random.randint(0, img.shape[1])
        temp_y = np.random.randint(0, img.shape[2])
        img[0][temp_x][temp_y] = temp_color
    return img


def add_erode(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.erode(img, kernel)
    return img


def add_dilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, kernel)
    return img


def do(img):
    # if random.random() < 0.5:
    img = add_noise(img)

    # if random.random() < 0.5:
    #     img = add_dilate(img)
    # else:
    #     img = add_erode(img)
    return img


class DataSet(torch.utils.data.Dataset):
    def __init__(self, img_path, transform=None, train=True):
        super().__init__()
        self.path = img_path
        self.transform = transform
        self.isTrain = train
        self.fileList = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.fileList[index]
        data = PIL.Image.open('%s/%s' % (self.path, img_name))
        height = 32
        width = int(data.size[0] / (data.size[1] / height))
        data = data.resize((width, height))
        if self.transform is not None:
            data = self.transform(data)
        label = re.search('([0-9_]+)', img_name).group(1)
        label = re.sub('\D', '', label)
        return data, label

    def __len__(self):
        return self.fileList.__len__()


myTransform = transforms.Compose([
    transforms.RandomAffine(5.0, translate=None, scale=(1.05, 0.95),
                            shear=0.2, resample=False, fillcolor=127),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: do(x)),
    #transforms.Normalize((0.1307,), (0.3081,)),
])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision

    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    train = DataSet('./number_area_img', transform=myTransform)
    img, label = iter(train).__next__()
    img = torchvision.utils.make_grid(img)
    img = img.numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()
    print(label)
