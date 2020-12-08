import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sn

"""
intialising global variables
"""

W1 = None
W2 = None
B1 = None
B2 = None
factor = 0.99 / 255


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
function to get data from the csv file
"""

def loadInputData(fName):
	global W1,W2,B1,B2
	with open(fName,'r') as f:
		inputData = [i.strip().split(',') for i in f.readlines()]
	inputData = np.array(inputData, dtype=np.float16)
	inputData = inputData.astype(float)
	datasetName = 'trainedNN.pkl'
	
	with open( datasetName, 'rb' ) as f:
	    NNobj = joblib.load( f )
	    
	W1, B1 = NNobj["bestValues"]["W1"], NNobj["bestValues"]["B1"]
	W2, B2 = NNobj["bestValues"]["W2"], NNobj["bestValues"]["B2"]
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
main function to start the training and initialise the variables
"""

def main():
	global factor
	makeCSV("MNIST/t10k-images-idx3-ubyte","MNIST/t10k-labels-idx1-ubyte","MNIST/mnist_test.csv",10000)
	validateData = loadInputData('MNIST/mnist_test.csv')
	XValid = validateData[:, 1:] * factor + 0.01
	YValid = validateData[:, 0]
	confusionMatrix(XValid,YValid,"q2ValidAccuracyMatrix.png")
	createHistogram(XValid,YValid,"q2Histogram.png")
	saveCorrectAndInCorrect(XValid,YValid)
	acc = predict(XValid,YValid)
	print("Final Accuracy "+ str(acc*100) + "%")

main()