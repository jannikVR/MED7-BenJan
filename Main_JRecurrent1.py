import RNN_J1
import numpy as np
import gym
import time
from colorama import Fore
import PerformanceView
import pygame

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
rewardDelta  = np.zeros((2))

action[0] = env.action_space.sample()

inputs = 5

state = np.zeros((1,inputs))
oldState = np.zeros((1,inputs))

env.reset()

performance = PerformanceView.PerformanceView((1200,600))

a = 0

totalReward = 0
bestReward = 0
rewardStreak = 0
bestRewardStreak = 0
averageReward = 0

learningRate = 2
render = False

avgRewardLast100Episodes = []



hiddenLayers = [2]
neurons = [6, 7, 8]
alpha = [0.5, 4, 5.5, 6]
alphaDecrease = [0.9995,0.999, 0.99, 0.95, 0.925]
minAlpha = 0.0000001

np.random.seed(1)

maxEpisodes = 2000

saves = 0
filenum = 0

attempts = 2
settingID = 0

for h in hiddenLayers:

    for n in neurons:

        for alp in alpha:

            for aD in alphaDecrease:

                for attempt in range(attempts+1):

                    # nn = NeuralNetwork.NeuralNetwork(1, inputs, h, n, 1)
                    rnn = RNN_J1.RecurrentNeuralNetwork(alp, aD,inputs, n, 1, 2)


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



                            #oldOldObs = oldObs
                            oldObs = obs
                            obs, reward, done, info = env.step((a)) # take a random action

                            state[0,0:4] = obs
                            oldState[0,0:4] = oldObs

                            #state[0,5:9] = oldObs
                            #state[0,9:13] = oldOldObs

                            if done: reward = -1

                            alphaDecreaseB = False
                            if reward > 1:
                                alphaDecreaseB = True

                            #averageReward += reward
                            oldState[0,4] = a
                            # print("OldState")
                            # print(oldState)
                            newReward = 0
                            if reward > 0:
                                newReward = 0
                            out = rnn.RNNSendInData([oldState[0]], newReward, alphaDecreaseB)

                            rewardSum += reward



                            for i in range(2):
                                state[0,4] = i

                                ##nn.setData(state,reward) // old state/ reward?
                                rewardDelta[i] = rnn.ForwardCleanThrough([state[0]])

                            a = round(float(np.argmax(rewardDelta)))


                            if done or step >= 199:

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