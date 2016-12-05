import NeuralNetwork
import RecurrentNN
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


totalReward = 0
bestReward = 0
rewardStreak = 0
bestRewardStreak = 0
averageReward = 0

learningRate = 2
learningRateMltiplier = 0.95

action = [0]
rewardDelta  = np.zeros((2))

action[0] = env.action_space.sample()

# nn = NeuralNetwork.NeuralNetwork(1,5,3,12,1)
rnn = RecurrentNN.RecurrentNeuralNetwork(learningRate, 5, 15, 1, 5)

state = np.zeros((1,5))




env.reset()

a = 0

stateList = list()
def AddStateToStateList(state):
    maxSteps = 5
    # create list

    # add new state to list
    stateList.append(state)
    # remove oldest if above max
    if stateList.__len__() > maxSteps:
        del stateList[0]

    # print(stateList)



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
        # env.render()


        obs, reward, done, info = env.step((a)) # take a random action

        if done: reward = -1
        averageReward += reward
        rewardSum += reward

        AddStateToStateList(state)
        AddStateToStateList(state)

        state2 = np.zeros([2, 5])
        state2[0] = stateList[0]
        state2[1] = stateList[1]
        rnn.Train(state2, reward)


        state[0,0:4] = obs






        for i in range(2):
            state[0,4] = i
            # nn.setData(state,reward)
            # rewardDelta[i] = nn.train(1,learningRate)
            state2 = np.zeros([2,5])
            state2[0] = stateList[0]
            state2[1] = stateList[1]
            rewardDelta[i], error = rnn.Forward(state2) # skal være to forskellige stateLists og så kun forward
            print(rewardDelta[i])
        a = round(float(np.argmax(rewardDelta))) # vælg den variabelpos med højest værdi

        state[0, 4] = a


        if done or step >= 199:

            if (rewardSum > 195): learningRate *= learningRateMltiplier

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





