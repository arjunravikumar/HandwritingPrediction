import numpy as np
import matplotlib.pyplot as plt
import sys


maxEpochs = 100
N = 4
costArr = []

learningRate = 1/N
regParam = 0.01

np.random.seed(10)

numofHiddenNodes = 8

W1 = np.random.normal(0,1, size=(2,2))
B1 = np.random.normal(0,1, size=2)
W2 = np.random.normal(0,1, size=(2,1))
B2 = np.random.normal(0,1, size=1)

np.random.seed(1)

def loadInputData(fName):
	with open(fName,'r') as f:
		inputData = [i.strip().split(',') for i in f.readlines()]
	inputData = np.array(inputData, dtype=np.float16)
	inputData = inputData.astype(float)
	return inputData 

def oneHot(y, n = 3, dtype=int):
	mat = np.zeros((len(y), n))
	for i, val in enumerate(y):
		mat[i, int(val)] = 1
	return mat.astype(dtype)    

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

def sigmoidDerivative(Z):
	derSigmo = 1 / (1 + np.exp(-Z))
	return derSigmo * (1 - derSigmo)

def softmax(X):
	exp = np.exp(X - np.max(X))
	return exp / exp.sum(axis=0, keepdims=True)

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

def backward(X,Y,a,z):
	global W2,learningRate
	dz2 = (a[2] - Y.T)
	dW2 = (np.dot(dz2,a[1].T) * learningRate)
	dB2 = (np.sum(dz2,axis=1, keepdims=True) * learningRate)
	da1 = np.dot(W2.T,dz2)
	dz1 = da1 * sigmoidDerivative(z[1])
	dW1 = learningRate * np.dot(dz1,a[0].T)
	dB1 = learningRate * np.sum(dz1,axis=1, keepdims=True)
	return dW1,dB1,dW2,dB2

def getGradientChange(X,Y):
	a,z = forward(X)
	dW1,dB1,dW2,dB2 = backward(X,Y,a,z)
	return dW1,dB1,dW2,dB2

def calculateLoss(output,target):
	global regParam
	crossEntropy = -(np.mean(target * np.log(output.T)))*2
	crossEntropy = crossEntropy + (regParam/2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	return crossEntropy

def getMiniBatch(data,splitVal):
	np.random.shuffle(data)
	return np.split(data, splitVal)

def train(inputData,Xvalid,Yvalid,XTrain,YTrain):
	global W1,B1,W2,B2
	lossTrainArr = [0 for i in range(maxEpochs)] 
	accuracyTrainArr = [0 for i in range(maxEpochs)] 
	lossTestArr = [0 for i in range(maxEpochs)] 
	accuracyTestArr = [0 for i in range(maxEpochs)] 
	k = len(set(inputData[:, -1]))
	yEncValid = oneHot(Yvalid,k,dtype=np.float)
	yEncTrain = oneHot(YTrain,k,dtype=np.float)
	for i in range(maxEpochs):
		batchData = getMiniBatch(inputData.copy(),10)
		for batch in batchData:
			X = batch[:, :-1]
			Y = batch[:, -1]
			yEnc = oneHot(Y,k,dtype=np.float)
			dW1,dB1,dW2,dB2 = getGradientChange(X,yEnc)
			W1 -= (dW1)
			B1 -= (dB1)
			W2 -= (dW2)
			B2 -= (dB2)
		accuracyTrainArr[i],lossTrainArr[i] = getAccuracyAndLost(XTrain,YTrain,yEncTrain)
		accuracyTestArr[i],lossTestArr[i] = getAccuracyAndLost(Xvalid,Yvalid,yEncValid)
	plt.plot(lossTrainArr, label='Train set loss',color = "green")
	plt.plot(lossTestArr, label='Test set loss',color = "orange")
	plt.plot(accuracyTrainArr, label='Train set accuracy',color = "blue")
	plt.plot(accuracyTestArr, label='Test set accuracy',color = "red")
	plt.legend(loc='best')
	plt.title('Loss,Accuracy - Epoch')
	plt.ylabel('Loss,Accuracy')
	plt.xlabel('Epochs')
	plt.savefig("q1cafterregplot.png")
	plt.close()

def getAccuracyAndLost(X,Y,yEnc):
	acc = predict(X,Y)
	a,z = forward(X)
	loss = calculateLoss(a[len(a)-1], yEnc)
	return acc,loss

def predict(X , Y):
	global W1,B1,W2,B2
	correct = 0
	a , z = forward(X)
	a2 = a[len(a)-1].T
	for i in range(len(a2)):
		if(Y[i] == a2[i].argmax()):
			correct += 1
	return correct/len(a2)

def main():
	global N,learningRate,W1,B1,W2,B2
	inputData = loadInputData('iris_train.dat')
	XTrain = inputData[:, :-1]
	YTrain = inputData[:, -1]
	N,D = XTrain.shape
	learningRate = 0.001
	k = len(set(YTrain))
	W1 = np.random.normal(0,2, size=(numofHiddenNodes,D))
	B1 = np.random.normal(0,2, size=(numofHiddenNodes,1))
	W2 = np.random.normal(0,2, size=(k,numofHiddenNodes))
	B2 = np.random.normal(0,2, size=(k,1))
	validateData = loadInputData('iris_test.dat')
	XValid = validateData[:, :-1]
	YValid = validateData[:, -1]
	print("Starting to Train")
	train(inputData,XValid,YValid,XTrain,YTrain)
	print("Finshed Training and saving the graphs")
	err,loss = getAccuracyAndLost(XTrain, YTrain, oneHot(YTrain))
	print("Accuracy on Train Data:",end = "")
	print(err * 100 ,"%")
	print("Loss on Train Data:",end = "")
	print(loss)
	err,loss = getAccuracyAndLost(XValid, YValid, oneHot(YValid))
	print("Accuracy on Test Data:",end = "")
	print(err * 100 ,"%")
	print("Loss on Test Data:",end = "")
	print(loss)
	print("Weight hidden:",W1)
	print("Weight output:",W2)
	print("Bias hidden:",B1)
	print("Bias output:",B2)

main()