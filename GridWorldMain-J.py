import pygame
import numpy as np
import random
import NeuralNetwork
from GridworldTestEnv import Gridworld


# env setup
tileSize = 30
envDim = 8, 7
startTile = 0, int((envDim[1]-1)/2)          # auto
# endTile = envDim[0]-1, int((envDim[1]-1)/2)  # auto
endTile = 3, 6  # auto
trapTiles = [(3,0), (4,7), (9,0)]
gworld = Gridworld(tileSize, envDim, startTile, endTile, trapTiles, trapReward= 0, standardReward=0.5)
active = True
counter = 0


# Agent
updateEnv = True
dirToGo = 0
curpos = startTile
oldpos = startTile


#NN setup
inputs = 3;
numOfActions = 4;
qvals  = np.zeros((numOfActions))
newQvals  = np.zeros((numOfActions))
nn = NeuralNetwork.NeuralNetwork(1,inputs,5,20,1)
state = np.zeros((1, inputs))
a = -1
gamma = 0.1
learningRate = 0.01
updateEnvTimes = 100000
updateEnvTimer = 0


def calcQvalues(position):
    qvalsf = np.zeros((numOfActions))

    # current state
    curState = [position[0] / envDim[0], position[1] / envDim[1]]
    # evaluate your position - find q-values
    for i in range(numOfActions):
        state[0, 0] = curState[0]
        state[0, 1] = curState[1]
        state[0, 2] = i

        qvalsf[i] = nn.feedForward(state)
    return qvalsf,

def findHighest(values):

    highestN = -1
    highestVal = -1000

    for i in range(len(values)):
        if values[i] > highestVal:
            highestVal = values[i]
            highestN = i

    return highestN, highestVal

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


        # evaluate your position - find q-values
        # qvals = calcQvalues(curpos) # normalizes pos and sends to network
        # print(qvals)
        # curState = [curpos[0] / envDim[0], curpos[1] / envDim[1]]

        curState = [curpos[0] / envDim[0], curpos[1] / envDim[1]]
        # evaluate your position - find q-values
        for i in range(numOfActions):
            state[0, 0] = curState[0]
            state[0, 1] = curState[1]
            state[0, 2] = i
            # print("1", state)
            qvals[i] = nn.feedForward(state)

        q = (qvals[0], qvals[1], qvals[2], qvals[3])

        # print(q)

        # dirToGo, value = findHighest(q)
        dirToGo = random.randrange(0,4,1)


        if counter % 50 == 0:
            # update env
            gworld.startUpdating()  # start updating

            for x in range(envDim[0]):  # for each state
                for y in range(envDim[1]):
                    # some random q-values

                    for i in range(numOfActions):
                        state[0, 0] = curState[0]
                        state[0, 1] = curState[1]
                        state[0, 2] = i
                        # print("1", state)
                        qvals[i] = nn.feedForward(state)

                    q = (qvals[0]*2-1, qvals[1]*2-1, qvals[2]*2-1, qvals[3]*2-1)

                    gworld.update(x, y, q)  # update

            gworld.doneUpdating()  # stop updating


        # do action
        # new state + reward
        curpos, reward, repositioned = gworld.performAction(curpos, dirToGo)

        # print("reward ",reward)
        # update brain: prev state + action + reward + maxQ*gamma

        # prev state and action
        state[0, 0] = curState[0]
        state[0, 1] = curState[1]
        state[0, 2] = dirToGo

        # print("2", state)

        if reward != 0.5:
            print(state)
            print(reward)

            #qMax = max(newQvals)
            qTarget = reward #+ gamma * qMax
            nn.setData(state, qTarget)
            error = nn.train(1, learningRate, showError= True)
            print("error",error)
        counter += 1
        updateEnv = False

pygame.quit() # clears data pieces
quit()


#
# def policyFindBestAction(self,):s
#     print("policy")
#
# def normalize(self,):
#     print("normalize")

# # Feedforward current state and predict qval for all actions
#         state[0, 0:2] = curpos
#         #print("CUR STATE", state)
#         for i in range(numOfActions):
#             state[0, 2] = i
#             qvals[i] = nn.feedForward(state)
#
#         print(qvals)
#
#         # Determine action
#         if (random.random() < .5):
#             dirToGo = round(float(np.argmax(qvals)))
#         else:
#             dirToGo = round(np.random.random_integers(0, 3))
#
#         print("Action: ", dirToGo)
#
#         curpos, reward, repositioned = gworld.performAction(curpos, dirToGo)  # L
#
#         #print("Reward: ", reward)
#         gworld.startUpdating()                       # start updating
#
#         for x in range(envDim[0]):                   # for each state
#             for y in range(envDim[1]):
#                 # some random q-values
#                 # val0 = random.random() * 2 - 1       # calculate q-values
#                 # val1 = random.random() * 2 - 1
#                 # val2 = random.random() * 2 - 1
#                 # val3 = random.random() * 2 - 1
#                 state[0, 0] = x
#                 state[0, 1] = y
#                 state[0, 2] = 0
#                 val0 = nn.feedForward(state)      # calculate q-values
#                 state[0, 2] = 1
#                 val1 = nn.feedForward(state)
#                 state[0, 2] = 2
#                 val2 = nn.feedForward(state)
#                 state[0, 2] = 3
#                 val3 = nn.feedForward(state)
#                 qvalues = (val0, val1, val2, val3)
#
#                 gworld.update(x, y, qvalues)        # update
#
#         gworld.doneUpdating()                       # stop updating
#
#
#
#         updateEnv = False
#
#     # Feedforward new state and calculate max q.
#     for i in range(numOfActions):
#         state[0, 2] = i
#         newQvals[i] = nn.feedForward(state)
#
#     qMax = max(newQvals)
#     qTarget = reward + gamma * qMax
#     errorVector = abs(qTarget - qvals[a])
#     nn.setData(state, qTarget)
#     error = nn.train(1, learningRate)