import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

"""
intialising global variables
"""

maxEpochs = 2000
N = 4
costArr = []

learningRate = 1/N
regParam = 0.8

np.random.seed(10)

numofHiddenNodes = 8

W1 = np.random.normal(0,1, size=(2,2))
B1 = np.random.normal(0,1, size=2)
W2 = np.random.normal(0,1, size=(2,1))
B2 = np.random.normal(0,1, size=1)

"""
function to get data from the file
"""

def loadInputData():
	with open('spiral_train.dat','r') as f:
		inputData = [i.strip().split(',') for i in f.readlines()]
	inputData = np.array(inputData, dtype=np.float16)
	inputData = inputData.astype(float)
	return inputData 

"""
sigmoid function to get the sigmoid value
"""

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

"""
sigmoid dervitave function to get the sigmoid dervitive value
"""

def sigmoidDerivative(Z):
	derSigmo = 1 / (1 + np.exp(-Z))
	return derSigmo * (1 - derSigmo)

"""
softmax function will return the softmax of the given vector
"""

def softmax(X):
	exp = np.exp(X - np.max(X))
	return exp / exp.sum(axis=0, keepdims=True)

"""
forward function to forward pass the NN
"""

def forward(X,w1 = None, b1 = None, w2 = None, b2 = None):
	global W1,B1,W2,B2
	if(w1 is None or b1 is None or w2 is None or b2 is None):
		w1,b1,w2,b2 = W1,B1,W2,B2
	a0 = X.T
	z1 = w1.dot(a0) + b1
	a1 = sigmoid(z1)
	z2 = w2.dot(a1) + b2
	a2 = softmax(z2)
	a = [a0,a1,a2]
	z = [None,z1,z2]
	return a,z

"""
To calculate the cross entropy loss in the function
"""

def calculateLoss(output,target):
	global regParam
	crossEntropy = -(np.mean(target * np.log(output.T)))*2
	return crossEntropy

"""
to get the one hot enoding from the y array
"""

def oneHot(y, n_labels, dtype):
    mat = np.zeros((len(y), n_labels))
    for i, val in enumerate(y):
        mat[i, int(val)] = 1
    return mat.astype(dtype)    

"""
backward function to get the derivative change in weights an bias
"""

def backward(X,Y,a,z):
	global W2,learningRate,regParam
	dz2 = (a[2] - Y.T)
	dW2 = (np.dot(dz2,a[1].T) * learningRate)
	dB2 = (np.sum(dz2,axis=1, keepdims=True) * learningRate)
	da1 = np.dot(W2.T,dz2)
	dz1 = da1 * sigmoidDerivative(z[1])
	dW1 = learningRate * np.dot(dz1,a[0].T)
	dB1 = learningRate * np.sum(dz1,axis=1, keepdims=True)
	return dW1,dB1,dW2,dB2

"""
to get the gradient change in every epoch
"""

def getGradientChange(X,Y):
	a,z = forward(X)
	dW1,dB1,dW2,dB2 = backward(X,Y,a,z)
	return dW1,dB1,dW2,dB2

"""
to plot the scatter graph ans decision boundary
"""

def plotScatterData(X,Y):
	xData = X[:, 0]
	yData = X[:, 1]
	colour = ["red","blue","green"]
	scatterColour = ["red" for i in range(len(Y))]
	buffer = 0.1
	xLimits = [xData.min() - buffer, xData.max() + buffer]
	yLimits = [xData.min() - buffer, xData.max() + buffer]
	setSize = 0.005
	for i in range(len(Y)):
		scatterColour[i] = colour[int(Y[i])]
	xGirds, yGrids = np.meshgrid(np.arange(xLimits[0], xLimits[1] , setSize), np.arange(yLimits[0], yLimits[1], setSize))
	combinedData = np.c_[xGirds.ravel(), yGrids.ravel()]
	a,z = forward(combinedData)
	armx = np.argmax(a[len(a)-1].T, axis=1)
	armx = armx.reshape(xGirds.shape)
	plt.figure(1, figsize=(8, 6))
	cMap = c.ListedColormap(colour)
	plt.pcolormesh(xGirds, yGrids, armx, cmap=cMap,alpha = 0.6)
	plt.scatter(xData, yData, c=scatterColour, edgecolors='black', cmap=plt.cm.Paired)
	plt.title('Points and the Decision boundary')
	plt.ylabel('Y Axis')
	plt.xlabel('X Axis')
	plt.savefig("q1bscatter.png")
	plt.close()

"""
train function to train the model for a given number of epochs
"""

def train(X,y_enc):
	global W1,B1,W2,B2
	costArr = [0 for i in range(maxEpochs)] 
	for i in range(maxEpochs):
		dW1,dB1,dW2,dB2 = getGradientChange(X,y_enc)
		W1 -= dW1
		B1 -= dB1
		W2 -= dW2
		B2 -= dB2
		a,z = forward(X)
		cost = calculateLoss(a[2], y_enc)
		if(cost <= 0.01):
			costArr[i] = cost
			break
		costArr[i] = cost
	plt.title('Loss - Epoch')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.plot(costArr)
	plt.savefig("q1blossepoch.png")
	plt.close()

"""
predict function to call the compute the accuracy of the model with the updated weights
"""

def predict(X , Y):
	global W1,B1,W2,B2
	correct = 0
	a , z = forward(X)
	a2 = a[len(a)-1].T
	for i in range(len(a2)):
		if(Y[i] == a2[i].argmax()):
			correct += 1
	print("Final Accuracy: " + str((correct/len(a2))*100)+"%")

"""
main function to start the training and initialise the variables
"""

def main():
	global N,learningRate,W1,B1,W2,B2
	inputData = loadInputData()
	X = inputData[:, :-1]
	Y = inputData[:, -1]
	N,D = X.shape
	learningRate = 1/N
	k = len(set(Y))
	yEnc = oneHot(y=Y, n_labels=k, dtype=np.float)
	W1 = np.random.normal(0,2, size=(numofHiddenNodes,D))
	B1 = np.random.normal(0,2, size=(numofHiddenNodes,1))
	W2 = np.random.normal(0,2, size=(k,numofHiddenNodes))
	B2 = np.random.normal(0,2, size=(k,1))
	print(W1,B1,W2,B2)
	print("Starting to Train")
	train(X,yEnc)
	print("Training Over")
	print("Staring to plot scatter")
	plotScatterData(X,Y)
	print("Done ploting")
	predict(X,Y)
	print("Weights Hidden :",W1)
	print("Weights Output:",W2)
	print("Bias Hidden:",B1)
	print("Bias Output:",B2)

main()