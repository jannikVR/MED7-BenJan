import numpy as np

def sigmoid(xx, derivative = False):
    if (derivative == True ):
        return xx * (1-xx)
    return 1 / (1+np.exp(-xx))

y = np.array([[4.],[2.],[6.],[1.],[5.],[3.],[7.],[10.],[14.],[9.],[13.],[20.],[18.],[22.],[17.],[21.],[19.],[23.]
            ,[28.], [26.], [30.], [25.], [29.], [27.], [31.], [8.], [16.]])

x = np.array(
            [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1],
             [0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 1], [0, 1, 1, 0, 1]
                , [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 1, 1],
             [1, 0, 1, 1, 1]
                , [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 1], [1, 1, 1, 0, 1], [1, 1, 0, 1, 1],
             [1, 1, 1, 1, 1], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]
             ])

#x = np.array(
#            [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0],[1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [1, 0, 1, 1, 1], [0, 1, 1, 1, 1]
#            ])

#y = np.array([[4.],[2.],[1.],[8.],[16.],[25.],[29.],[30.],[23.],[15.]])

maxY = 50 #y.max(axis=0)+1
alphas = [0.1,0.5,1,2,5,10,50]
y = y / maxY

np.random.seed(1)

class NeuralNetwork():

    def __init__(self, numberOfDatasets, inputs, hiddenLayers, neuronsPrHiddenLayer, outputs):
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

        print ("INITIATED")

    def setData(self,inputData,outputData):
        self.__inputNeuronValues = inputData
        self.__outputData = outputData

    def train(self, episodes):

        for episode in range(episodes):

            self.__hiddenNeuronValues[0, :, :] = sigmoid(np.dot(self.__inputNeuronValues, self.__inputWeights))

            for i in range(1,self.__layers-2):
                #print "## i ##" + str(i)
                self.__hiddenNeuronValues[i, :, :] = sigmoid(np.dot(self.__hiddenNeuronValues[i-1], self.__hiddenWeights[i-1,:,:]))

            self.__outputNeuronValues = sigmoid(np.dot(self.__hiddenNeuronValues[self.__layers - 3,:,:],self.__outputWeights))

            self.__outputErrors = self.__outputData - self.__outputNeuronValues

            self.__outputDelta = self.__outputErrors * sigmoid(self.__outputNeuronValues,derivative=True)

            if (self.__layers > 3):
                self.__hiddenErrors[0,:,:] = self.__outputDelta.dot(self.__outputWeights.T)

                for i in range(self.__layers-4):
                    #print "Enter loop " + str(i) + " " + str(self.__layers - 4)
                    self.__hiddenDelta[i,:,:] = self.__hiddenErrors[i,:,:] * sigmoid(self.__hiddenNeuronValues[self.__layers - 3 - i,:,:] ,derivative=True)
                    self.__hiddenErrors[i + 1, :, :] = self.__hiddenDelta[i,:,:].dot(self.__hiddenWeights[i,:,:].T)
                #self.__hiddenDelta[0, :, :] = self.__hiddenErrors[0, :, :] * sigmoid(self.__hiddenNeuronValues[1, :, :], derivative=True)
                #self.__hiddenErrors[1, :, :] = self.__hiddenDelta[0,:,:].dot(self.__hiddenWeights[0,:,:].T)

                #should it really be -4?
                self.__hiddenDelta[self.__layers-4,:,:] = self.__hiddenErrors[self.__layers-4,:,:] * sigmoid(self.__hiddenNeuronValues[1,:,:], derivative=True)
                self.__inputErrors = self.__hiddenDelta[self.__layers - 4, :, :].dot(self.__hiddenWeights[self.__layers - 4, :, :].T)
            else:
                self.__inputErrors = self.__outputDelta.dot(self.__outputWeights.T)

            self.__inputDelta = self.__inputErrors * sigmoid(self.__hiddenNeuronValues[0,:,:], derivative=True)
            self.__outputWeights += np.dot(self.__hiddenNeuronValues[self.__layers-3, :, :].T, self.__outputDelta)


            if (self.__layers > 3):
                for i in range(self.__layers-4,-1,-1):
                    #print "## i ##" + str(i)
                    self.__hiddenWeights[i,:,:] += np.dot(self.__hiddenNeuronValues[i, :, :].T, self.__hiddenDelta[self.__layers - 4 - i,:,:])

            #self.__hiddenWeights[0, :, :] += np.dot(self.__hiddenNeuronValues[0, :, :].T,self.__hiddenDelta[0, :, :])

            #self.__hiddenWeights[self.__layers-4, :, :] += np.dot(self.__hiddenNeuronValues[0, :, :].T,self.__hiddenDelta[1, :, :])

            self.__inputWeights += np.dot(self.__inputNeuronValues.T,self.__inputDelta)
            if (episode % (episodes/10)) == 0:
                print ("Error:" + str(np.mean(np.abs(self.__outputErrors)))) #not correct error!!!

        print (self.__outputNeuronValues * maxY)

    def test(self, data):
        inputs = 5
        hiddenLayers = 2
        neuronsPrHiddenLayer = 6
        outputs = 1
        inputNeuronValues = np.zeros((2, inputs), dtype=float)
        hiddenNeuronValues = np.zeros((hiddenLayers, 2, neuronsPrHiddenLayer), dtype=float)
        outputNeuronValues = np.zeros((2, outputs), dtype=float)


        hiddenNeuronValues[0, :, :] = sigmoid(np.dot(data, self.__inputWeights))

        if (hiddenLayers > 1):
            for i in range(self.__layers - 3, self.__layers - 2):
                hiddenNeuronValues[i, :, :] = sigmoid(
                    np.dot(hiddenNeuronValues[i - 1], self.__hiddenWeights[i - 1, :, :]))

        outputNeuronValues = sigmoid(
            np.dot(hiddenNeuronValues[self.__layers - 3, :, :], self.__outputWeights))
        print ("SECOND: " + str(outputNeuronValues * maxY))


nn = NeuralNetwork(x.shape[0],5,2,6,1)

nn.setData(x,y)

nn.train(100000)
x = np.array([[0, 1, 0, 1, 1], [1, 1, 0, 0, 0]])
nn.test(x)