from trace import Trace

import pygame
import numpy as np
import math

class SimpleEnv():



    def __init__(self, increment = 0.05, rewardDependentOnHowClose = True):
        self.rewardDependentOnHowClose = rewardDependentOnHowClose
        self.reward = 0
        self.pos = 0
        self.winPos = 1
        self.increment = increment
        self.done = False


    def step(self, action):
        if action == 1 & self.pos < self.winPos:
            self.pos += self.increment
        else:
            self.pos -= self.increment

        if self.pos >= self.winPos: # if win
            self.reward = 1
            self.done = True
        else:
            if self.rewardDependentOnHowClose: # if incrementary reward
                self.reward = self.pos
            else:
                self.reward = 0

        return self.pos, self.reward, self.done, 0

    def reset(self):
        self.pos = 0
        self.done = False


