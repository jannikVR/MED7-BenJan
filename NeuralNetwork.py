import numpy as np
import time

def sigmoid(xx, derivative = False):
    if (derivative == True ):
        return xx * (1-xx)
    return 1 / (1+np.exp(-xx))


class NeuralNetwork():

    def __init__(self, numberOfDatasets, inputs, hiddenLayers, neuronsPrHiddenLayer, outputs):

        np.random.seed(1)

        self.__nInputs = inputs
        self.__nOutputs = outputs
        self.__nHiddenLayers = hiddenLayers
        self.__neuronsPrHiddenLayer = neuronsPrHiddenLayer

        self.__layers = hiddenLayers+2

        self.__inputNeuronValues = np.zeros((numberOfDatasets,inputs),dtype = float)
        self.__hiddenNeuronValues = np.zeros((hiddenLayers,numberOfDatasets,neuronsPrHiddenLayer),dtype = float)
        self.__outputNeuronValues = np.zeros((numberOfDatasets, outputs),dtype = float)

        self.__inputErrors = np.zeros((numberOfDatasets,inputs*neuronsPrHiddenLayer),dtype = float)
        self.__inputDelta = np.zeros((numberOfDatasets,inputs*neuronsPrHiddenLayer),dtype = float)
        self.__inputWeights = 2 * np.random.random((inputs, neuronsPrHiddenLayer)) - 1
        self.__hiddenErrors = np.zeros((hiddenLayers-1, numberOfDatasets, neuronsPrHiddenLayer),dtype = float)
        self.__hiddenDelta = np.zeros((hiddenLayers-1, numberOfDatasets, neuronsPrHiddenLayer),dtype = float)
        self.__hiddenWeights = 2 * np.random.random((hiddenLayers-1, neuronsPrHiddenLayer, neuronsPrHiddenLayer)) - 1
        self.__outputErrors = np.zeros((numberOfDatasets, outputs),dtype = float)
        self.__outputDelta = np.zeros((numberOfDatasets, outputs),dtype = float)
        self.__outputWeights = 2 * np.random.random((neuronsPrHiddenLayer, outputs)) - 1

    def setData(self,inputData,outputData):
        self.__inputNeuronValues = inputData
        self.__outputData = outputData

    def train(self, episodes, learningRate, showError = False):

        oldEstTimeLeft = 0


        for episode in range(episodes):

            self.__hiddenNeuronValues[0, :, :] = sigmoid(np.dot(self.__inputNeuronValues, self.__inputWeights))

            for i in range(1,self.__layers-2):
                self.__hiddenNeuronValues[i, :, :] = sigmoid(np.dot(self.__hiddenNeuronValues[i-1], self.__hiddenWeights[i-1,:,:]))

            self.__outputNeuronValues = sigmoid(np.dot(self.__hiddenNeuronValues[self.__layers - 3,:,:],self.__outputWeights))

            self.__outputErrors = self.__outputData - self.__outputNeuronValues

            self.__outputDelta = self.__outputErrors * sigmoid(self.__outputNeuronValues,derivative=True)

            if (self.__layers > 3):
                self.__hiddenErrors[0,:,:] = self.__outputDelta.dot(self.__outputWeights.T)

                for i in range(self.__layers-4):
                    self.__hiddenDelta[i,:,:] = self.__hiddenErrors[i,:,:] * sigmoid(self.__hiddenNeuronValues[self.__layers - 3 - i,:,:] ,derivative=True)
                    self.__hiddenErrors[i + 1, :, :] = self.__hiddenDelta[i,:,:].dot(self.__hiddenWeights[i,:,:].T)

                self.__hiddenDelta[self.__layers-4,:,:] = self.__hiddenErrors[self.__layers-4,:,:] * sigmoid(self.__hiddenNeuronValues[1,:,:], derivative=True)
                self.__inputErrors = self.__hiddenDelta[self.__layers - 4, :, :].dot(self.__hiddenWeights[self.__layers - 4, :, :].T)
            else:
                self.__inputErrors = self.__outputDelta.dot(self.__outputWeights.T)

            self.__inputDelta = self.__inputErrors * sigmoid(self.__hiddenNeuronValues[0,:,:], derivative=True)
            self.__outputWeights += np.dot(self.__hiddenNeuronValues[self.__layers-3, :, :].T, self.__outputDelta) * learningRate


            if (self.__layers > 3):
                for i in range(self.__layers-4,-1,-1):
                    self.__hiddenWeights[i,:,:] += np.dot(self.__hiddenNeuronValues[i, :, :].T, self.__hiddenDelta[self.__layers - 4 - i,:,:]) * learningRate

            self.__inputWeights += (np.dot(self.__inputNeuronValues.T,self.__inputDelta)) * learningRate

            if(showError):
                if (episode > 1000):
                    estTimeLeft = int((time.clock() * (episodes / episode)) - time.clock())

                    if (abs(oldEstTimeLeft - estTimeLeft) >= 3):
                        oldEstTimeLeft = estTimeLeft
                        print("\n")
                        print ("Est. time left: " + str(estTimeLeft) + " sec")
                        if (self.__nHiddenLayers < 2):
                            print ("Error:" + str(np.mean(np.abs(self.__outputErrors)) + np.mean(np.abs(self.__inputErrors))))

                        else:
                            print ("Error:" + str(np.mean(np.abs(self.__outputErrors)) + np.mean(np.abs(self.__inputErrors)) + np.mean(np.abs(self.__hiddenErrors))))

                if (self.__nHiddenLayers < 2):
                    return np.mean(np.abs(self.__outputErrors)) + np.mean(np.abs(self.__inputErrors))

                else:
                    return np.mean(np.abs(self.__outputErrors)) + np.mean(np.abs(self.__inputErrors)) + np.mean(
                            np.abs(self.__hiddenErrors))

    def test(self, data):
        nDatasets = data.shape[0]
        inputNeuronValues = np.zeros((nDatasets, self.__nInputs), dtype=float)
        hiddenNeuronValues = np.zeros((self.__nHiddenLayers, nDatasets, self.__neuronsPrHiddenLayer), dtype=float)
        outputNeuronValues = np.zeros((nDatasets, self.__nOutputs), dtype=float)

        hiddenNeuronValues[0, :, :] = sigmoid(np.dot(data, self.__inputWeights))

        if (self.__nHiddenLayers > 1):
            for i in range(self.__layers - 3, self.__layers - 2):
                hiddenNeuronValues[i, :, :] = sigmoid(
                    np.dot(hiddenNeuronValues[i - 1], self.__hiddenWeights[i - 1, :, :]))

        outputNeuronValues = sigmoid(
            np.dot(hiddenNeuronValues[self.__layers - 3, :, :], self.__outputWeights))

    def feedForward(self, data):
        nDatasets = data.shape[0]

        self.__inputNeuronValues = np.zeros((nDatasets, self.__nInputs), dtype=float)
        self.__hiddenNeuronValues = np.zeros((self.__nHiddenLayers, nDatasets, self.__neuronsPrHiddenLayer), dtype=float)
        self.__outputNeuronValues = np.zeros((nDatasets, self.__nOutputs), dtype=float)

        self.__inputNeuronValues = data

        self.__hiddenNeuronValues[0, :, :] = sigmoid(np.dot(self.__inputNeuronValues, self.__inputWeights))

        for i in range(1, self.__layers - 2):
            self.__hiddenNeuronValues[i, :, :] = sigmoid(
                np.dot(self.__hiddenNeuronValues[i - 1], self.__hiddenWeights[i - 1, :, :]))

        self.__outputNeuronValues = sigmoid(
            np.dot(self.__hiddenNeuronValues[self.__layers - 3, :, :], self.__outputWeights))

        return self.__outputNeuronValues
