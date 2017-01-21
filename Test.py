import NeuralNetwork
import numpy as np



nn = NeuralNetwork.NeuralNetwork(1,1,2,2,1)

state = np.zeros((1,1))

value = 10


for i in range(100000):

    state[0,0] = np.random.randint(10)
    state[0,0] /= 10
    reward = 0

    if( state > 5):
        reward = 1

    guess = nn.feedForward(state)
    print(guess)


    nn.setData(state, reward)
    guess = nn.train(1,2)



    print(guess)
    print("R= " + str(reward) + " S=" + str(state))
    print("Error= " + str(guess -reward))
    print()