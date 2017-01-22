import pygame
import numpy as np
import random
import NeuralNetwork
from GridworldTestEnv import Gridworld

# env setup
tileSize = 30
envDim = 3, 3
startTile = 0, int((envDim[1]-1)/2)          # auto
endTile = envDim[0]-1, int((envDim[1]-1)/2)  # auto
trapTiles = [(3,1), (4,5), (9,0)]

gworld = Gridworld(tileSize, envDim, startTile, endTile, trapTiles)


active = True
updateEnv = True
dirToGo = 0
curpos = startTile

#NN setup
inputs = 3;
numOfActions = 4;
qvals  = np.zeros((numOfActions))
newQvals  = np.zeros((numOfActions))
nn = NeuralNetwork.NeuralNetwork(1,inputs,1,5,1)
state = np.zeros((1,inputs))
a = -1
gamma = 0.9
learningRate = 0.1
updateEnvTimes = 100
updateEnvTimer = 0

while active:

    # if QUIT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            active = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                dirToGo = 0
                updateEnv = True
            if event.key == pygame.K_UP:
                dirToGo = 1
                updateEnv = True
            if event.key == pygame.K_RIGHT:
                dirToGo = 2
                updateEnv = True
            if event.key == pygame.K_DOWN:
                dirToGo = 3
                updateEnv = True
            if(event.key == pygame.K_SPACE):
                updateEnvTimer = 1
            if (event.key == pygame.K_x):
                updateEnvTimer = updateEnvTimes

    if(updateEnvTimer > 0):
        updateEnvTimer -= 1
        updateEnv= True

    # update Env
    if updateEnv:

        # Feedforward current state and predict qval for all actions
        state[0, 0:2] = curpos
        #print("CUR STATE", state)
        for i in range(numOfActions):
            state[0, 2] = i
            qvals[i] = nn.feedForward(state)

        print(qvals)

        # Determine action
        if (random.random() < .5):
            dirToGo = round(float(np.argmax(qvals)))
        else:
            dirToGo = round(np.random.random_integers(0, 3))

        print("Action: ", dirToGo)

        curpos, reward, repositioned = gworld.performAction(curpos, dirToGo)  # L

        #print("Reward: ", reward)
        gworld.startUpdating()                       # start updating

        for x in range(envDim[0]):                   # for each state
            for y in range(envDim[1]):
                # some random q-values
                val0 = random.random() * 2 - 1       # calculate q-values
                val1 = random.random() * 2 - 1
                val2 = random.random() * 2 - 1
                val3 = random.random() * 2 - 1
                qvalues = (val0, val1, val2, val3)

                gworld.update(x, y, qvalues)        # update

        gworld.doneUpdating()                       # stop updating

        # test
        # print(gworld.performAction((envDim[0]-2, int((envDim[1]-1)/2)), 0)) # L
        # print(gworld.performAction((envDim[0]-2, int((envDim[1]-1)/2)), 1)) # U
        # print(gworld.performAction((envDim[0]-2, int((envDim[1]-1)/2)), 2)) # R
        # print(gworld.performAction((envDim[0]-2, int((envDim[1]-1)/2)), 3)) # D




        updateEnv = False

    # Feedforward new state and calculate max q.
    for i in range(numOfActions):
        state[0, 2] = i
        newQvals[i] = nn.feedForward(state)

    qMax = max(newQvals)
    qTarget = reward + gamma * qMax
    errorVector = abs(qTarget - qvals[a])
    nn.setData(state, qTarget)
    error = nn.train(1, learningRate)

pygame.quit() # clears data pieces
quit()