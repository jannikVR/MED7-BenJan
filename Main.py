import NeuralNetwork
import numpy as np
import gym
import time
from colorama import Fore
import PerformanceView
import pygame
import random

# pygame.init()
# pygame.display.set_caption("Stats")
# bgCol = (255,255,255)
# frontCol = (0,0,0)
# screen = pygame.display.set_mode((600,200))
# frameRate = 60
# clock = pygame.time.Clock()

env = gym.make('CartPole-v0')
env.reset()
print(env.action_space)
print(env.observation_space)

action = [0]
qvals  = np.zeros((2))
newQvals  = np.zeros((2))

action[0] = env.action_space.sample()

inputs = 5

state = np.zeros((1,inputs))

env.reset()

a = 0

totalReward = 0
bestReward = 0
rewardStreak = 0
bestRewardStreak = 0
averageReward = 0

learningRate = 2
render = False

avgRewardLast100Episodes = []

hiddenLayers = [3,4]
neurons = [12,14,16]
alpha = [4,5]
alphaDecrease = [0.85,0.9,0.95,.99,.999,0.999]

#hiddenLayers = [2,3]
#neurons = [4,6,8,10,12,14,16]
#alpha = [2.5,3,4,5,6]
#alphaDecrease = [0.5,0.7,0.9,0.99,0.999,0.99999]

hiddenLayers = [1,2,3]
neurons = [4,6,8,10,22]
#discount set to 0.1
alpha = [0.001,0.01,0.1,0.5,1,1.5,2] #For alpha decrease every epoch
alphaDecrease = [.00000001,.000001,.0001,.001,.01] #for alpha decrease every epoch image 1000 and up
discount = [0]
#alpha = [0.0001,.1,.25,.5,.75,1,1.5,2] #For alpha decrease every time it wins image 2000 and up also used for image 3000 and up, with qtarget as input
#alphaDecrease = [0.99999] #For alpha decrease every time it wins

#image 4000, no discount factor, using qtarget. Also decrease every time it wins.
#image 5000, discount factor of 0.05, using qtarget, decrease every time it wins
#image 6000, no discount, qtarget, decrease every epoch


#hiddenLayers = [2,3]
#neurons = [22,24,26]
#alpha = [0.25,0.5,0.75,1,1.5,2,3] #For alpha decrease every epoch
#alphaDecrease = [.00000001,.00000005,.0000001,.0000005,.000001] #for alpha decrease every epoch
#alpha = [0.9,1,1.1,2] #For alpha decrease every time it wins
#alphaDecrease = [0.5,0.7,0.9,0.99,0.999,0.99999] #For alpha decrease every time it wins
minAlpha = 0.0000001

np.random.seed(1)

maxEpisodes = 2000

saves = -1
filenum = 13000
attempts = 1
settingID = 0
#discount = 0
rewardResolution = 100

totalAttempts = -1

decreaseType = "Every Epoch"

hasData = False
reset = False
rewardSumList = []
errorAvgList = []

performanceDebug = True
if(performanceDebug): performance = PerformanceView.PerformanceView((1200,600))

for h in hiddenLayers:

    for n in neurons:

        for alp in alpha:

            for aD in alphaDecrease:

                for gamma in discount:

                    for attempt in range(attempts):

                        totalAttempts += 1

                        nn = NeuralNetwork.NeuralNetwork(1, inputs, h, n, 1)

                        learningRate = alp
                        learningRateDecrease = aD
                        print("\n")
                        print(Fore.LIGHTBLUE_EX)
                        print("######################### SETTINGS ###########################")
                        print("Hidden layers: ", h)
                        print("Neurons: ", n)
                        print("Alpha: ", alp)
                        print("Alpha Decrease: ", aD)
                        print(Fore.RESET)

                        for i in range(len(avgRewardLast100Episodes)):
                            avgRewardLast100Episodes[i] = 0

                        averageReward = 0

                        if performanceDebug:
                            if hasData:
                                if saves > 4:
                                    saves = 0
                                    filenum += 1
                                    reset = True
                                    performance.saveScreenshot("data_"+str(filenum))
                                    hasData = False
                                    del performance
                                    performance = PerformanceView.PerformanceView((1200, 600))

                                if reset:
                                    totalAttempts = 1
                                    reset = False

                                performance.addPerformance(settingID, episodesToWin,settings, rewardSumList, totalAttempts-1, attempts, errorAvgList)

                            performance.update(maxEpisodes,attempts,200,rewardResolution,attempt-1, attempts)

                            rewardSumList.clear()
                            errorAvgList.clear()

                        settings = [h, n, alp, aD, gamma, decreaseType]

                        if attempt == 0:
                            saves += 1
                            settingID += 1

                        for episode in range(maxEpisodes+1):
                            obs = env.reset()
                            state[0, 0:4] = obs

                            if episode % 500 == 0:
                                print("Episode: " + str(episode))
                                print("Avg Reward: ", averageReward)



                            if episode > 0:
                                for i in range(len(avgRewardLast100Episodes)):
                                    averageReward += avgRewardLast100Episodes[i]
                                averageReward /= len(avgRewardLast100Episodes)



                                if episode % 500 == 0:
                                    print("\nEpisode: ", episode)
                                    print("  ##### Average reward last 100 ep.: " +  str(averageReward) + " #####")
                                    print("ALPHA: ", learningRate)

                                if averageReward >= 195:
                                    print(Fore.LIGHTCYAN_EX + "")
                                    print("###################### WON WON WON ########################")
                                    print("################# After " + str(episode-100) + " episodes ######################")
                                    print("##### Average reward last 100 ep.: " + str(averageReward) + " #####")
                                    print(avgRewardLast100Episodes)
                                    print(Fore.RESET + "")
                                    episodesToWin = episode-100
                                    break

                                # if rewardSum >= 200:
                                #     print(Fore.LIGHTCYAN_EX + "")
                                #     print("###################### WON WON WON ########################")
                                #     print("################# After " + str(episode-100) + " episodes ######################")
                                #     print("##### Average reward last 100 ep.: " + str(averageReward) + " #####")
                                #     print(avgRewardLast100Episodes)
                                #     print(Fore.RESET + "")
                                #     episodesToWin = episode
                                #     break

                            rewardSum = 0

                            avgError = 0

                            for step in range(200):
                                #env.render()

                                # Feedforward current state and predict qval for all actions
                                for i in range(2):
                                    state[0, 4] = i
                                    qvals[i] = nn.feedForward(state)

                                #Determine action
                                if(random.random() < 0.95):
                                    a = round(float(np.argmax(qvals)))
                                else:
                                    a = round(random.random())

                                # if (step == 1 and episode % 1000 == 0 and episode > 1):
                                #     print(a)
                                #     print(random.random())

                                prevState = state

                                oldObs = obs
                                #Move and get new state
                                obs, reward, done, info = env.step((a)) # take a random action

                                if done: reward = -1

                                state[0,0:4] = obs

                                #Feedforward new state and calculate max q.
                                for i in range(2):
                                    state[0, 4] = i
                                    newQvals[i] = nn.feedForward(state)

                                qMax = max(newQvals)
                                qTarget = reward + gamma*qMax
                                errorVector = abs(qTarget - qvals[a])
                                #state[0,0:4] = state[0,0:4] / 20+0.5
                                #print(state)
                                nn.setData(state, qTarget)
                                error = nn.train(1, learningRate, showError=True)
                                avgError += error
                                #print("Error: ", error)

                                #if(step % 100 == 0 and step > 1): print("Error: ", error)

                                #state[0,5:9] = oldObs
                                #state[0,9:13] = oldOldObs

                                #averageReward += reward

                                rewardSum += reward

                                #Train using previous action, previous state and new reward
                                # if(step > 0):
                                #     state[0, 4] = a
                                #     nn.setData(prevState, reward)
                                #
                                #     error = nn.train(1, learningRate, showError=True)
                                #     #if(step % 25 == 0): print("Error: ", error)
                                #
                                # for i in range(2):
                                #     state[0,4] = i
                                #     qvals[i] = nn.feedForward(state)
                                #
                                # prevState = state
                                # a = round(float(np.argmax(qvals)))

                                if done or step >= 199:

                                    #if step > 100: print("Total steps: ", step)

                                    hasData = True

                                    episodesToWin = episode

                                    if len(avgRewardLast100Episodes) == 100:
                                        avgRewardLast100Episodes.remove(avgRewardLast100Episodes[0])

                                    if(episode % (maxEpisodes/rewardResolution) == 0 and episode > 0):
                                        rewardSumList.append(averageReward)
                                        errorAvgList.append(avgError/step)

                                    avgRewardLast100Episodes.append(rewardSum)


                                    #if(performanceDebug):
                                        #performance.addReward(rewardSum)

                                    #decreaseType = "Every Time The Agent Wins"
                                    learningRate = learningRate * 1 / (1 + learningRateDecrease * episode)
                                    # if (rewardSum > 195):
                                    #     learningRate *= learningRateDecrease
                                    #     learningRate = max(learningRate,minAlpha)

                                    if rewardSum >= 195:
                                        rewardStreak += 1
                                        if rewardStreak > bestRewardStreak:
                                            bestRewardStreak = rewardStreak
                                    else:
                                        rewardStreak = 0
                                        render = False

                                    if rewardStreak > 99:
                                        render = True

                                    if rewardSum > bestReward:
                                        bestReward = rewardSum

                                    #if(episode % 100 == 0): print("End of episode " + str(episode) + "\n reward: " + str(rewardSum) + "\n Highest reward: " + str(bestReward) + "\n Learning rate: " + str(learningRate)
                                    #                             + "\n Reward streak: " + str(rewardStreak) + "\n Best reward streak: " + str(bestRewardStreak))
                                    break;

                            #clock.tick(frameRate)