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


    def Forward(self, input):

        print("Forward Input")
        print(input)
        print()

        # hidden layer (input ~+ prev_hidden)
        self.layer_1 = sigmoid(np.dot(input, self.synapse_0) + np.dot(self.layer_1_values[-1], self.synapse_h))

        # output layer (new binary representation)
        self.layer_2 = sigmoid(np.dot(self.layer_1, self.synapse_1))

        # store hidden layer so we can use it in the next timestep
        self.layer_1_values.append(copy.deepcopy(self.layer_1))

        return self.layer_2


    def BackProp(self, correct, position):

        layer_1 = self.layer_1_values[-position - 1]
        prev_layer_1 = self.layer_1_values[-position - 2]

        # error at output layer
        layer_2_delta = self.layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_1_delta = (self.future_layer_1_delta.dot(self.synapse_h.T) + layer_2_delta.dot(
            self.synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
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


    def Train(self, input, actual):

        print("Train Input")
        print(input)
        print()

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
                a[i] = input[i, self.n_steps - position - 1]

            X = np.array([a])
            y = np.array([[actual[self.n_steps - position - 1]]]).T
            output = self.Forward(X)

            # Calc Deltas/Error for propagating
            self.layer_2_error = y - output
            self.layer_2_deltas.append((self.layer_2_error) * sigmoid_output_to_derivative(output))
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
                a[i] = input[i, position]

            X = np.array([a])

            self.BackProp(X, position)

        # Finally change weights
        self.SetWeights()



        return d, overallError