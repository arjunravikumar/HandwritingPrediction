import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sn
from PIL import Image

"""
intialising global variables
"""

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

"""
to convert the given byte data to csv for accessing it later
"""

def makeCSV(imageFileName, labelFileName, csvFileName, numberOfRecords):
    inputImageFile = open(imageFileName, "rb")
    csvFile = open(csvFileName, "w")
    inputLabelFile = open(labelFileName, "rb")

    inputImageFile.read(16)
    inputLabelFile.read(8)
    images = []

    for i in range(numberOfRecords):
        image = [ord(inputLabelFile.read(1))]
        for j in range(28*28):
            image.append(ord(inputImageFile.read(1)))
        images.append(image)

    for image in images:
        csvFile.write(",".join(str(pix) for pix in image)+"\n")
    inputImageFile.close()
    csvFile.close()
    inputLabelFile.close()

"""
function to get data from the csv file
"""

def loadInputData(fName):
	with open(fName,'r') as f:
		inputData = [i.strip().split(',') for i in f.readlines()]
	inputData = np.array(inputData, dtype=dataType)
	inputData = inputData.astype(float)
	return inputData 

"""
to get the one hot enoding from the y array
"""

def oneHot(y, n = 10, dtype=int):
	mat = np.zeros((len(y), n))
	for i, val in enumerate(y):
		mat[i, int(val)] = 1
	return mat.astype(dtype)    

"""
sigmoid function to get the sigmoid value
"""

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

"""
sigmoid dervitave function to get the sigmoid dervitive value
"""

def sigmoidDerivative(Z):
	derSigmo = sigmoid(Z)
	return derSigmo * (1 - derSigmo)

"""
softmax function will return the softmax of the given vector
"""

def softmax(X):
	exp = np.exp(X - np.max(X))
	return exp / exp.sum(axis=0, keepdims=True)

"""
to get the L2 penality value
"""

def getL2penality(W):
	return (np.sum(np.square(W)))/2

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
backward function to get the derivative change in weights an bias
"""

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

"""
to get the gradient change in every epoch
"""

def getGradientChange(X,Y):
	a,z = forward(X)
	dW1,dB1,dW2,dB2 = backward(X,Y,a,z)
	return dW1,dB1,dW2,dB2

"""
To calculate the cross entropy loss in the function
"""

def calculateLoss(output,target):
	global regParam
	crossEntropy = -(np.mean(target * np.log(output.T)))*2
	crossEntropy = crossEntropy + ((regParam/2) * (np.sum(np.square(W1)) + np.sum(np.square(W2))))
	return crossEntropy

"""
To get minibatch from the whole data using the given split val
"""

def getMiniBatch(data,splitVal):
	np.random.shuffle(data)
	return np.split(data, splitVal)

"""
train function to train the model for a given number of epochs
"""

def train(inputData,Xvalid,Yvalid,XTrain,YTrain):
	global W1,B1,W2,B2,factor,bestData,regParam
	lossTrainArr = [0 for i in range(maxEpochs)] 
	accuracyTrainArr = [0 for i in range(maxEpochs)] 
	lossTestArr = [0 for i in range(maxEpochs)] 
	accuracyTestArr = [0 for i in range(maxEpochs)] 
	k = len(set(inputData[:, 0]))
	yEncValid = oneHot(Yvalid,k)
	yEncTrain = oneHot(YTrain,k)
	reg_Param = 0
	weightsAtEveryEpoch = []
	for i in range(maxEpochs):
		batchData = getMiniBatch(inputData.copy(),100)
		for batch in batchData:
			X = np.asfarray(batch[:, 1:], dtype=dataType) * factor + 0.01
			Y = batch[:, 0]
			yEnc = oneHot(Y,k)
			dW1,dB1,dW2,dB2 = getGradientChange(X,yEnc)
			W1 = W1 - (dW1) + (reg_Param * getL2penality(W1))
			B1 -= (dB1)
			W2 = W2 - (dW2) + (reg_Param * getL2penality(W2))
			B2 -= (dB2)
		accuracyTestArr[i],lossTestArr[i] = getAccuracyAndLost(Xvalid,Yvalid,yEncValid)
		accuracyTrainArr[i],lossTrainArr[i] = getAccuracyAndLost(XTrain,YTrain,yEncTrain)
		weightsAtEveryEpoch.append({"W1" : W1.copy(),"W2" : W2.copy(),"B1" : B1.copy(),"B2" : B2.copy()})
		print("Progress", int(((i+1)/maxEpochs)*100), "%", end="\r")
	print("Progress", int(((i+1)/maxEpochs)*100))
	plt.plot(lossTrainArr, label='Train set loss',color = "green")
	plt.plot(lossTestArr, label='Test set loss',color = "orange")
	bestEpoch = lossTestArr.index(min(lossTestArr))
	bestData = weightsAtEveryEpoch[bestEpoch]
	print("The value with the least loss was found at epoch", bestEpoch, "with value", lossTestArr[bestEpoch])
	plt.plot(bestEpoch,lossTestArr[bestEpoch], marker = '.', markerfacecolor='black', markersize=12)
	plt.axvline(bestEpoch)
	plt.annotate("Overfitting barrier", (bestEpoch,lossTestArr[bestEpoch]))
	plt.legend(loc='best')
	plt.title('Loss - Epoch')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.savefig("q2plotLoss.png")
	plt.close()
	plt.plot(accuracyTrainArr, label='Train set Accuracy',color = "green")
	plt.plot(accuracyTestArr, label='Test set Accuracy',color = "orange")
	plt.plot(bestEpoch,accuracyTestArr[bestEpoch], marker = '.', markerfacecolor='black', markersize=12)
	plt.axvline(bestEpoch)
	plt.annotate("Overfitting barrier", (bestEpoch,accuracyTestArr[bestEpoch]))
	plt.legend(loc='best')
	plt.title('Accuracy - Epoch')
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.savefig("q2plotAcc.png")
	plt.close()

"""
To get the accuracy and loss of the model using the updated weights
"""

def getAccuracyAndLost(X,Y,yEnc):
	acc = predict(X,Y)
	a,z = forward(X)
	loss = calculateLoss(a[len(a)-1], yEnc)
	return acc,loss

"""
To get the accuracy of the model
"""

def predict(X , Y):
	global W1,B1,W2,B2
	correct = 0
	a , z = forward(X)
	a2 = a[len(a)-1].T
	for i in range(len(a2)):
		if(Y[i] == a2[i].argmax()):
			correct += 1
	return correct/len(a2)

"""
To create the histogram for accuracy of all classes
"""

def createHistogram(X , Y, fNAme):
    correct = np.zeros(10)
    total = np.zeros(10)
    a , z = forward(X)
    a2 = a[len(a)-1].T
    for i in range(len(a2)):
        res_max = a2[i].argmax()
        target = Y[i]
        target = int(target)
        res_max = int(res_max)
        if(res_max == target):
            correct[res_max] += 1
        total[target] += 1
    for index in range(len(total)):
            correct[index] /= total[index]
    plt.bar([str(i) for i in range(0,10)],correct)
    plt.title('Class accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.savefig(fNAme)
    plt.close()

"""
to create a confusion matrix of the data prediction disrtibution
"""

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
               cmap="Greens",
               fmt='0.2f')
    chart.set_title(label="Accuracy Matrix")
    plt.yticks(rotation=0)
    plt.savefig(fNAme)
    plt.close()

"""
to save one correct and one incorrect image prediction for each class
"""

def saveCorrectAndInCorrect(X,Y):
	correct = np.zeros(10)
	incorrect = np.zeros(10)
	a , z = forward(X)
	a2 = a[len(a)-1].T
	for i in range(len(a2)):
		res_max = a2[i].argmax()
		target = Y[i]
		res_max = int(res_max)
		target = int(target)
		if(res_max == target):
			if(correct[res_max] == 0):
				img = X[i].reshape((28,28))
				plt.figure(figsize=(8,6))
				plt.imshow(img, cmap="Greys")
				plt.savefig("OutputImages/Correct"+str(res_max)+".png")
				plt.close()
				correct[res_max] += 1
		else:
			if(incorrect[res_max] == 0):
				img = X[i].reshape((28,28))
				plt.figure(figsize=(8,6))
				plt.imshow(img, cmap="Greys")
				plt.savefig("OutputImages/Incorrect"+str(res_max)+".png")
				plt.close()
				incorrect[res_max] += 1

"""
to pickle save the values for future runs
"""

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

"""
to split the data into train and validation
"""

def splitTrainAndValidation(inputDataSet):
	return inputDataSet[len(inputDataSet)//5:] , inputDataSet[:len(inputDataSet)//5]

"""
main function to start the training and initialise the variables
"""

def main():
	global N,learningRate,W1,B1,W2,B2,factor
	makeCSV("MNIST/train-images-idx3-ubyte","MNIST/train-labels-idx1-ubyte","MNIST/mnist_train.csv",60000)
	inputData = loadInputData('MNIST/mnist_train.csv')
	inputData , validateData = splitTrainAndValidation(inputData)
	XTrain = np.asfarray(inputData[:, 1:], dtype=dataType) * factor + 0.01
	YTrain = inputData[:, 0]
	N,D = XTrain.shape
	learningRate = 0.001
	k = len(set(YTrain))
	W1 = np.random.normal(0,2, size=(numofHiddenNodes,D))
	B1 = np.random.normal(0,2, size=(numofHiddenNodes,1))
	W2 = np.random.normal(0,2, size=(k,numofHiddenNodes))
	B2 = np.random.normal(0,2, size=(k,1))
	XValid = np.asfarray(validateData[:, 1:], dtype=dataType) * factor + 0.01
	YValid = validateData[:, 0]
	print("Starting to Train")
	train(inputData,XValid,YValid,XTrain,YTrain)
	print("Saving the values")
	saveTheValues()
	print("Finshed Training and saving the graphs")
	err,loss = getAccuracyAndLost(XTrain, YTrain, oneHot(YTrain))
	print("Final accuracy on Train Data:",end = "")
	print(err * 100 ,"%")
	err,loss = getAccuracyAndLost(XValid, YValid, oneHot(YValid))
	print("Final accuracy on Test Data:",end = "")
	print(err * 100 ,"%")
	confusionMatrix(XValid,YValid,"q2ValidAccuracyMatrix.png")
	confusionMatrix(XTrain,YTrain,"q2TrainAccuracyMatrix.png")
	createHistogram(XValid,YValid,"q2ValidAccuracyMatrix.png")
	saveCorrectAndInCorrect(XValid,YValid)

main()