import copy, numpy as np

np.random.seed(0)


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)





class RecurrentNeuralNetwork():


    def __init__(self, alpha, input_dim, hidden_dim, output_dim, n_steps):
        self.alpha = alpha
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # initialize neural network weights
        self.synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
        self.synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
        self.synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

        self.synapse_0_update = np.zeros_like(self.synapse_0)
        self.synapse_1_update = np.zeros_like(self.synapse_1)
        self.synapse_h_update = np.zeros_like(self.synapse_h)

        self.layer_2_deltas = list()
        self.layer_1_values = list()
        self.layer_1_values.append(np.zeros(self.hidden_dim))

        self.n_steps = n_steps

        self.future_layer_1_delta = np.zeros(self.hidden_dim)

        self.savedInputs = np.zeros((self.input_dim, self.n_steps))
        self.savedActual = np.zeros( self.n_steps)
        self.output = np.zeros(self.n_steps)
        self.savedOutputs = 2 * np.random.random((hidden_dim, output_dim)) - 1

        self.currStep = 0
        self.overallError = 0

    def ForwardCleanThrough(self, inputI):

        # output layer (new binary representation)
        x = sigmoid(np.dot(inputI, self.synapse_0) + np.dot(self.layer_1_values[-1], self.synapse_h))
        out = sigmoid(np.dot(x, self.synapse_1))
        return out


    def ForwardAndChange(self, inputI):

        # hidden layer (input ~+ prev_hidden)
        self.layer_1 = sigmoid(np.dot(inputI, self.synapse_0) + np.dot(self.layer_1_values[-1], self.synapse_h))

        # output layer (new binary representation)
        self.layer_2 = sigmoid(np.dot(self.layer_1, self.synapse_1))

        # store hidden layer so we can use it in the next timestep
        self.layer_1_values.append(copy.deepcopy(self.layer_1))

        return self.layer_2


    def BackProp(self, correct, position):
        #
        # print()
        # print("layer1Val2")
        # print(self.layer_1_values)

        layer_1 = self.layer_1_values[-position - 1]
        prev_layer_1 = self.layer_1_values[-position - 2]



        # error at output layer
        layer_2_delta = self.layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_1_delta = (self.future_layer_1_delta.dot(self.synapse_h.T) + layer_2_delta.dot(
            self.synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weight-changes so we can try again
        self.synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        self.synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        self.synapse_0_update += correct.T.dot(layer_1_delta)

        self.future_layer_1_delta = layer_1_delta


    def SetWeights(self):
        self.synapse_0 += self.synapse_0_update * self.alpha
        self.synapse_1 += self.synapse_1_update * self.alpha
        self.synapse_h += self.synapse_h_update * self.alpha

        # Reset
        self.synapse_0_update *= 0
        self.synapse_1_update *= 0
        self.synapse_h_update *= 0

    def RNNSendInData(self, input, actual):

        # add input to matrix
        self.savedInputs[:, self.n_steps - self.currStep-1] = input[0]

        # add actual to matrix
        self.savedActual[self.n_steps - self.currStep-1] = actual

        # # generate input and output
        # a = np.zeros([self.input_dim])
        # for i in range(self.input_dim):
        #     a[i] = self.savedInputs[i, self.currStep]
        #
        # X = np.array([a])
        #
        #

        # output = self.ForwardAndChange(input)

        if(self.currStep >= self.n_steps-1):
            # print("saved")
            # print(self.savedInputs)
            # print()
            # print(self.savedActual)

            outOut, overallError = self.Train(self.savedInputs, self.savedActual) #train network with batch data

            print("guess" + str(outOut))

            out = 0  #print
            for index, x in enumerate(reversed(outOut)):
                out += x * pow(2, index)

            print(" = " + str(out))
            print("overallError" + str(overallError))

            self.currStep = -1;   # reset to overwrite
            self.savedInputs = np.zeros((self.input_dim, self.n_steps))
            self.savedActual = np.zeros(self.n_steps)


        self.currStep += 1;  # increase stepNr and check if end of batch

        # return output;










    def RNNSendInData2(self, inputI, actual):
        # add input to matrix
        self.savedInputs[:, self.n_steps - self.currStep-1] = inputI[0]

        # add actual to matrix
        self.savedActual[self.n_steps - self.currStep-1] = actual


        X = np.array(inputI)
        #
        y = np.array([[actual]]).T

        output = self.ForwardAndChange(X)
        # print(X)
        # print(y)

        # Calc Deltas/Error for propagating
        self.layer_2_error = y - output
        self.layer_2_deltas.append((self.layer_2_error) * sigmoid_output_to_derivative(output))  # save/append error for each t
        self.overallError += np.abs(self.layer_2_error[0])

        # decode estimate so we can print it out
        self.output[self.n_steps - self.currStep - 1] = np.round(output[0][0])



        if(self.currStep >= self.n_steps-1):

            # Create / reset bufferDelta
            self.future_layer_1_delta = np.zeros(self.hidden_dim)  # reset

            # Go through each step and propagate
            for position in range(self.n_steps):
                # X = np.array([[input[0, position], input[1, position]]])

                # generate input and output
                a = np.zeros([self.input_dim])
                for i in range(self.input_dim):
                    a[i] = self.savedInputs[i, position]

                X = np.array([a])

                self.BackProp(X, position)

            # Finally change weights
            self.SetWeights()


            print("guess" + str(self.output))

            out = 0  #print
            for index, x in enumerate(reversed(self.output)):
                out += x * pow(2, index)

            print(" = " + str(out))
            print("overallError" + str(self.overallError))

            self.currStep = -1;   # reset to overwrite
            self.savedInputs = np.zeros((self.input_dim, self.n_steps))
            self.savedActual = np.zeros(self.n_steps)
            self.overallError = 0

        self.currStep += 1;  # increase stepNr and check if end of batch

        return output;






    def Train(self, inputs, actual):

        # print("Train Input")
        # print(input)
        # print()

        # where we'll store our best guess (binary encoded)
        d = np.zeros_like(actual)
        overallError = 0

        self.layer_2_deltas = list()  # reset lists
        self.layer_1_values = list()
        self.layer_1_values.append(np.zeros(self.hidden_dim))

        # moving along the positions in the binary encoding
        for position in range(self.n_steps):

            # generate input and output
            a = np.zeros([self.input_dim])
            for i in range(self.input_dim):
                a[i] = inputs[i, self.n_steps - position - 1] # baglæns! need last input first..

            X = np.array([a])


            y = np.array([[actual[self.n_steps - position - 1]]]).T
            # output = self.ForwardCleanThrough(X)
            # print(output)
            output = self.ForwardAndChange(X)
            print(output)
            #
            # print()
            print(X)
            print(y)

            # Calc Deltas/Error for propagating
            self.layer_2_error = y - output
            self.layer_2_deltas.append((self.layer_2_error) * sigmoid_output_to_derivative(output)) # save/append error for each t
            overallError += np.abs(self.layer_2_error[0])

            # decode estimate so we can print it out
            d[self.n_steps - position - 1] = np.round(output[0][0])


        # Create / reset bufferDelta
        self.future_layer_1_delta = np.zeros(self.hidden_dim)   # reset

        # Go through each step and propagate
        for position in range(self.n_steps):
            # X = np.array([[input[0, position], input[1, position]]])

            # generate input and output
            a = np.zeros([self.input_dim])
            for i in range(self.input_dim):
                a[i] = inputs[i, position]

            X = np.array([a])


            self.BackProp(X, position)

        # Finally change weights
        self.SetWeights()



        return d, overallError