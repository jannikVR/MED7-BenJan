import NeuralNetwork
import numpy as np
import gym
import time

env = gym.make('CartPole-v0')
env.reset()
print(env.action_space)
print(env.observation_space)

#Take action
#Get reward
#Get new observations
#Update Weights using reward and observertions and choose action

action = [0]
rewardDelta  = np.zeros((2))

action[0] = env.action_space.sample()

nn = NeuralNetwork.NeuralNetwork(1,9,3,12,1)

state = np.zeros((1,9))

env.reset()

a = 0

totalReward = 0
bestReward = 0
rewardStreak = 0
bestRewardStreak = 0
averageReward = 0

learningRate = 2
render = False

for episode in range(100000):
    obs = env.reset()
    #oldObs = obs

    rewardSum = 0
    if episode % 100 == 0:
        print("##### Average reward: " +  str(averageReward/100) + " #####")
        if averageReward/100 >= 195:
            print("###################### WON WON WON ########################")
            print("################# After " + str(episode-100) + " episodes ######################")
        averageReward = 0

    for step in range(200):
        env.render()

        #oldOldObs = oldObs
        oldObs = obs
        obs, reward, done, info = env.step((a)) # take a random action

        state[0,0:4] = obs
        state[0,5:9] = oldObs
        #state[0,9:13] = oldOldObs

        if done: reward = -1

        averageReward += reward

        rewardSum += reward

        for i in range(2):
            state[0,4] = i
            nn.setData(state,reward)
            rewardDelta[i] = nn.train(1,learningRate)

        a = round(float(np.argmax(rewardDelta)))

        if done or step >= 199:

            if (rewardSum > 195): learningRate = 0

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

            if(episode % 10 == 0): print("End of episode " + str(episode) + "\n reward: " + str(rewardSum) + "\n Highest reward: " + str(bestReward) + "\n Learning rate: " + str(learningRate)
                                         + "\n Reward streak: " + str(rewardStreak) + "\n Best reward streak: " + str(bestRewardStreak))
            break;


        #nn.setData(state,reward)





#
# x = np.array([[0,0,1],
# [0,1,1],
# [1,0,1],
# [1,1,1]])
#
# y = np.array([[0],
# [1],
# [1],
# [0]])
#
# nn = NeuralNetwork.NeuralNetwork(x.shape[0],x.shape[1],1,4,1)
#
# nn.setData(x,y)
#
# nn.train(60000)