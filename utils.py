#!/usr/bin/python
# encoding: utf-8
import os
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable

alphabet = "0123456789_"


def decode(pred):
    last = 11
    str = ""
    for i in pred:
        if i != last and i != 10:
            str += alphabet[i]
        last = i
    return str


def batch_decode(preds):
    sstr = []
    for i in preds:
        str = decode(i)
        sstr.append(str)
    return sstr


def encode(str):
    t = []
    l = []
    for i in str:
        tmp = alphabet.find(i)
        if(tmp != -1):
            t.append(tmp)
        else:
            raise Exception("Out of alphabet!")
    l.append(len(t))
    return t, l


def batch_encode(sstr):
    t, l = [], []
    for str in sstr:
        tmp_t, tmp_l = encode(str)
        t.append(tmp_t)
        l += tmp_l
    length = max(l)
    for j in t:
        if len(j) != length:
            for i in range(length - len(j)):
                j.append(10)

    return t, l


if __name__ == "__main__":
    f = encode("0113_")
    print(f)
    str = decode([1, 10, 10, 1, 2, 10, 2, 2, 2, 3, 4])
    print(str)

    f = batch_encode(["0113", "2233", "334", "3333"])
    print(f)
