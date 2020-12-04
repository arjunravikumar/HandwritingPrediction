import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sn

maxEpochs = 500
N = 4
costArr = []
learningRate = 1/N
regParam = 0.0001

np.random.seed(10)

numofHiddenNodes = 300
factor = 0.99 / 255
dataType = np.float32

W1 = np.random.normal(0,1, size=(2,2))
B1 = np.random.normal(0,1, size=2)
W2 = np.random.normal(0,1, size=(2,1))
B2 = np.random.normal(0,1, size=1)

bestData = None

np.random.seed(1)

def loadInputData(fName):
	with open(fName,'r') as f:
		inputData = [i.strip().split(',') for i in f.readlines()]
	inputData = np.array(inputData, dtype=dataType)
	inputData = inputData.astype(float)
	return inputData 

def oneHot(y, n = 10, dtype=int):
	mat = np.zeros((len(y), n))
	for i, val in enumerate(y):
		mat[i, int(val)] = 1
	return mat.astype(dtype)    

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

def sigmoidDerivative(Z):
	derSigmo = sigmoid(Z)
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
	crossEntropy = crossEntropy + ((regParam/2) * (np.sum(np.square(W1)) + np.sum(np.square(W2))))
	return crossEntropy

def getMiniBatch(data,splitVal):
	np.random.shuffle(data)
	return np.split(data, splitVal)

def train(inputData,Xvalid,Yvalid,XTrain,YTrain):
	global W1,B1,W2,B2,factor,bestData
	lossTrainArr = [0 for i in range(maxEpochs)] 
	accuracyTrainArr = [0 for i in range(maxEpochs)] 
	lossTestArr = [0 for i in range(maxEpochs)] 
	accuracyTestArr = [0 for i in range(maxEpochs)] 
	k = len(set(inputData[:, 0]))
	yEncValid = oneHot(Yvalid,k)
	yEncTrain = oneHot(YTrain,k)
	for i in range(maxEpochs):
		batchData = getMiniBatch(inputData.copy(),100)
		for batch in batchData:
			X = np.asfarray(batch[:, 1:], dtype=dataType) * factor + 0.01
			Y = batch[:, 0]
			yEnc = oneHot(Y,k)
			dW1,dB1,dW2,dB2 = getGradientChange(X,yEnc)
			W1 -= (dW1)
			B1 -= (dB1)
			W2 -= (dW2)
			B2 -= (dB2)
		accuracyTestArr[i],lossTestArr[i] = getAccuracyAndLost(Xvalid,Yvalid,yEncValid)
		accuracyTrainArr[i],lossTrainArr[i] = getAccuracyAndLost(XTrain,YTrain,yEncTrain)
		if(bestData is None and i > 50):
			previousData = {"W1":W1.copy(),"B1":B1.copy(),"W2":W2.copy(),"B2":B2.copy()}
			if(accuracyTestArr[i]>accuracyTestArr[i-1]):
				bestData = previousData.copy()
		print("Progress", int(((i+1)/maxEpochs)*100), "%", end="\r")
	print("Progress", int(((i+1)/maxEpochs)*100))
	plt.plot(lossTrainArr, label='Train set loss',color = "green")
	plt.plot(lossTestArr, label='Test set loss',color = "orange")
	bestEpoch = lossTestArr.index(min(lossTestArr))
	print("The value with the least loss was found at epoch", bestEpoch, "with value", lossTestArr[bestEpoch])
	plt.plot(bestEpoch,lossTestArr[bestEpoch], marker = '.', markerfacecolor='black', markersize=12)
	plt.annotate("Lowest loss at epoch", (bestEpoch,lossTestArr[bestEpoch]))
	plt.legend(loc='best')
	plt.title('Loss - Epoch')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.savefig("q2plotLoss.png")
	plt.close()
	plt.plot(accuracyTrainArr, label='Train set Accuracy',color = "green")
	plt.plot(accuracyTestArr, label='Test set Accuracy',color = "orange")
	plt.plot(bestEpoch,accuracyTestArr[bestEpoch], marker = '.', markerfacecolor='black', markersize=12)
	plt.annotate("Lowest loss at epoch", (bestEpoch,accuracyTestArr[bestEpoch]))
	plt.legend(loc='best')
	plt.title('Loss - Epoch')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.savefig("q2plotAcc.png")
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

def confusionMatrix(X , Y, fNAme):
    cm = np.zeros((10, 10), int)
    a , z = forward(X)
    a2 = a[len(a)-1].T
    for i in range(len(a2)):
        res_max = a2[i].argmax()
        target = Y[i]
        cm[res_max, int(target)] += 1

    df_c = pd.DataFrame(cm, index = range(0,10),
                        columns= range(0,10))
    plt.figure(figsize=(8,6))
    chart = sn.heatmap(df_c,
               annot=True,
               annot_kws={"fontsize" : 8},
               linewidths=0.5,
               cmap="coolwarm",
               fmt='0.2f')
    chart.set_title(label="Accuracy Matrix")
    plt.yticks(rotation=0)
    plt.savefig(fNAme)

def saveTheValues():
	global W1,B1,W2,B2,bestData
	datasetName = 'trainedNN.pkl'
	NNobj = {
	"lastValues":
	{
	"W1" : W1,
	"W2" : W2,
	"B1" : B1,
	"B2" : B2
	},
	"bestValues": bestData
	}

	with open( datasetName, 'wb' ) as f:
	    joblib.dump( NNobj, f, compress=3 )

def main():
	global N,learningRate,W1,B1,W2,B2,factor
	inputData = loadInputData('mnist_train.csv')
	XTrain = np.asfarray(inputData[:, 1:], dtype=dataType) * factor + 0.01
	YTrain = inputData[:, 0]
	N,D = XTrain.shape
	learningRate = 0.001
	k = len(set(YTrain))
	W1 = np.random.normal(0,2, size=(numofHiddenNodes,D))
	B1 = np.random.normal(0,2, size=(numofHiddenNodes,1))
	W2 = np.random.normal(0,2, size=(k,numofHiddenNodes))
	B2 = np.random.normal(0,2, size=(k,1))
	validateData = loadInputData('mnist_test.csv')
	XValid = np.asfarray(validateData[:, 1:], dtype=dataType) * factor + 0.01
	YValid = validateData[:, 0]
	print("Starting to Train")
	train(inputData,XValid,YValid,XTrain,YTrain)
	print("Finshed Training and saving the graphs")
	err,loss = getAccuracyAndLost(XTrain, YTrain, oneHot(YTrain))
	print("Final accuracy on Train Data:",end = "")
	print(err * 100 ,"%")
	err,loss = getAccuracyAndLost(XValid, YValid, oneHot(YValid))
	print("Final accuracy on Test Data:",end = "")
	print(err * 100 ,"%")
	confusionMatrix(XValid,YValid,"q2ValidAccuracyMatrix.png")
	confusionMatrix(XTrain,YTrain,"q2TrainAccuracyMatrix.png")
	print("Saving the values")
	saveTheValues()
main()