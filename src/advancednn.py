# %load blncq.py
import pygame
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import matplotlib.pyplot as plt
import os
from collections import deque

import IPython.display

print("OK")

pygame.init()

# Pygame screen size:
sx = 900
sy = 500

# Table hgitht, left brim, right brim...
tableH = sy-20
tableL = int(sx*0.1)
tableR = int(sx*0.9)

# Stick picture size:
stickL = 400
stickW = 50

cticker = pygame.time.Clock()
testsc = pygame.display.set_mode((sx, sy))

im_bg = pygame.image.load("assets/bg.png").convert_alpha()
im_st = pygame.image.load("assets/pp.png").convert_alpha()

im_bg = pygame.transform.scale(im_bg, (sx,sy))
im_st = pygame.transform.scale(im_st, (stickW, stickL))

# 
def get_bottom(img, th, bwidth = stickW):
    x = -img.get_rect()[2]
    y = -img.get_rect()[3]
    if np.sin(th)<0:
        x=0
    if np.cos(th)<0:
        y=0
    bx = np.sign(np.cos(th)) * np.abs(np.sin(th)) * bwidth/2
    by = np.sign(np.sin(th)) * np.abs(np.cos(th)) * bwidth/2
    return (x+by, y+bx)

#for i in range(1300):
#     testsc.blit(im_bg, (0,0))
#     tst = pygame.transform.rotate(im_st, i)
#     x,y = get_bottom(tst, i*np.pi/180)
#     testsc.blit(tst, (500+x,300+y))
#     # testsc.blit(im_bd, (300,500))
#     # print(tst.get_rect(),(x,y))
#     pygame.display.update()
#     cticker.tick(30)
    


class gameState():
    def __init__(self, th,w, x, v, g=1, zhuanguan=10):
        self.th = th
        self.w = w
        self.x = x
        self.v = v
        self.g = g
        self.guanliang = zhuanguan

    def forward(self, dv, disp = 0):
        self.v += dv
        self.x += self.v
        self.w +=  ( dv * np.cos(self.th*np.pi/180) + self.g * np.sin(self.th*np.pi/180))/self.guanliang
        self.th += self.w
        
        if disp == 1:
            one_frame = self.drawgame()
        else:
            one_frame = []

        ##
        stateLabel = 1
        if self.x < tableL or self.x > tableR or self.th < -90 or self.th > 90:
            stateLabel = 0
        ##
        return np.stack((self.th, self.w, self.x, self.v), axis=0), stateLabel, one_frame
    
    def drawgame(self):
        testsc.blit(im_bg, (0, 0))
        tst = pygame.transform.rotate(im_st, self.th)
        x,y = get_bottom(tst, self.th*np.pi/180)
        testsc.blit(tst, (self.x + x, tableH + y))
        pygame.display.update()
        cticker.tick(30)
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def printgame(self):
        print("Theta:%s, w: %s, position_x: %s, v: %s" % (self.th, self.w, self.x, self.v))
    ##

    def plotgame(self, fig, img):
        IPython.display.clear_output(wait=True)
        img.set_data(image_data)
        IPython.display.display(fig)
        cticker.tick(30)

    def getState(self):
        return np.stack((self.th, self.w, self.x, self.v) , axis=0)

    def gameinit(self, dis = 5):
        self.th = (np.random.rand()-0.5)*2*dis
        self.w = 0.0
        self.x = np.random.choice(range(tableL+200, tableR-200))
        self.v = np.random.rand()-0.5
        

game = gameState(1.0,0.0,400.0,0.0)

# for i in range(200):
#     x,s = game.forward(np.random.rand()-0.5,1)
#     # game.printgame()
#     if s==0:
#         game.printgame()
#         break

    


# In[ ]:




class ActionNN(nn.Module):
    def __init__(self, ni, no, nh=10, scale = 20):
        super(ActionNN, self).__init__()
        self.inputN = ni
        self.hiddenN = nh
        self.outN = no
        self.fc1 = nn.Linear(ni, nh)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nh, no)
    ##
    def forward(self, input):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    ##
##

class QValue(nn.Module):
    def __init__(self, ni, nh=10):
        super(QValue, self).__init__()
        self.inputN = ni
        self.fc1 = nn.Linear(ni, nh)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nh,1)
    ##
    def forward(self, stateAction):
        sa = stateAction
        out = self.fc1(sa)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    ##
##

netA = ActionNN(4,1)
netQ = QValue(5)
critS = nn.MSELoss()
optimizerA = torch.optim.Adam(netA.parameters(), lr=1e-3)
optimizerQ = torch.optim.Adam(netQ.parameters(), lr = 1e-3)

records = deque()
recoMax = 50000

sss=np.array([])
aaa=np.array([])

def catsa(s,a):
    sss = torch.Tensor(s)
    aaa = a.data
    result =  Variable(torch.cat((sss,aaa)))
    return result

###    return torch.cat((torch.Tensor(np.array(s)), torch.Tensor([a])))



    
game.gameinit()
def repeatGame(g, ann, epochN=100, disp_game=0):
    for epoch in range(epochN):
        s1 = g.getState()
        a = ann(Variable(torch.FloatTensor(s1)))
        
        s2, label, oneframe = g.forward(a.data[0], disp=disp_game)
        if label == 0:
            r = -100
            g.gameinit()
        else:
            r = 1
        epochRecord = (s1, a, r, s2, label)
        records.append(epochRecord)
        if len(records) > recoMax:
            records.popleft()
##

epochN = 1000
batchSize = 40
stepQ = 30
stepA = 30


def getLoss(sigma = 0.99):
    minibatch = random.sample(records, batchSize)
    qt  = [netQ( catsa( minibatch[i][0], minibatch[i][1])) for i in range(batchSize)]
    stt = [netA( Variable(torch.FloatTensor(np.array(minibatch[i][3])))) for i in range(batchSize)]
    qtt = [minibatch[i][2] * netQ( catsa(minibatch[i][3], stt[i]) ) for i in range(batchSize)]
    betterQ = []
    for i in range(batchSize):
        qq = sigma * qtt[i] + minibatch[i][2]
        ##qq.requires_grad = False
        qq = qq.detach()
        betterQ.append(qq)
    ##
    lossQ = sum( [critS(qt[i], betterQ[i]) for i in range(batchSize)] )
    lossA = -sum(qtt)

    return lossQ, lossA
##
repeatGame(game, netA, 100)
for epoch in range(epochN):
    
    repeatGame(game, netA, 10)
    
    
    lossQ, lossA = getLoss()

    optimizerA.zero_grad()
    lossA.backward()
    optimizerA.step()
    
    optimizerQ.zero_grad()
    lossQ.backward()
    optimizerQ.step()

    if epoch%50 == 0:
        print(epoch)
        print("Loss_of_Action: %f; loss_of_Q: %f" %(lossA.data[0], lossQ.data[0]))
    
    
    

    
        
