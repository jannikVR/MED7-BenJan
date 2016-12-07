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

hiddenLayers = [1,2,3,4,6]
neurons = [5,6,7,8,12]
alpha = [0.05,0.1,0.2,0.5,1]
alphaDecrease = [0.7,0.8,0.9,0.99,0.999]

minAlpha = 0.0000001

np.random.seed(1)

maxEpisodes = 2000

saves = 0
filenum = 200
attempts = 3
settingID = 0
discount = 0.1

performanceDebug = True
if(performanceDebug): performance = PerformanceView.PerformanceView((1200,600))

for h in hiddenLayers:

    for n in neurons:

        for alp in alpha:

            for aD in alphaDecrease:

                for attempt in range(attempts+1):

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

                    settings = [h, n, alp, aD]

                    print("ATTEMPT STATS: ", attempt, attempts)
                    if attempt == 0:
                        saves += 1
                        settingID += 1

                    if performanceDebug:
                        if attempt > 0:
                            if saves > 4:
                                saves = 0
                                filenum += 1
                                performance.saveScreenshot("data_"+str(filenum))
                                del performance
                                performance = PerformanceView.PerformanceView((1200, 600))

                            performance.addPerformance(settingID, episodesToWin,settings)

                        performance.update(maxEpisodes,attempts)

                    for episode in range(maxEpisodes+1):
                        obs = env.reset()
                        state[0, 0:4] = obs

                        if episode % 500 == 0:
                            print("Episode: " + str(episode))
                            print("Avg Reward: ", averageReward)

                        rewardSum = 0

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
                                averageReward = 0
                                break

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

                            qMax = round(float(np.argmax(newQvals)))
                            qTarget = reward + discount*qMax
                            errorVector = qTarget - qvals[a]
                            nn.setData(state, errorVector)
                            error = nn.train(1, learningRate, showError=True)
                            if(step % 100 == 0 and step > 1): print("Error: ", error)

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

                                if step > 100: print("Total steps: ", step)

                                episodesToWin = episode

                                if len(avgRewardLast100Episodes) == 100:
                                    avgRewardLast100Episodes.remove(avgRewardLast100Episodes[0])

                                avgRewardLast100Episodes.append(rewardSum)


                                if (rewardSum > 195):
                                    learningRate *= learningRateDecrease
                                    learningRate = max(learningRate,minAlpha)

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