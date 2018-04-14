# %load advancednn.py
# %load blncq.py
#import pygame
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import matplotlib.pyplot as plt
import os
from collections import deque

def fun(x,y):
    return (y-torch.sin(x))**2 - (x-np.pi)**2

x = Variable(torch.FloatTensor([5]), requires_grad = True)
y = Variable(torch.FloatTensor([-0.8]), requires_grad = True)

Opt1 = 


