import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sn

W1 = None
W2 = None
B1 = None
B2 = None
factor = 0.99 / 255

def confusionMatrix(X , Y):
    cm = np.zeros((10, 10), int)
    a , z = forward(X)
    a2 = a[len(a)-1].T
    for i in range(len(a2)):
        res_max = a2[i].argmax()
        target = Y[i]
        cm[res_max, int(target)] += 1

    df_c = pd.DataFrame(cm, index = range(1,11),
                        columns= range(1,11))
    plt.figure(figsize=(8,6))
    chart = sn.heatmap(df_c,
               annot=True,
               annot_kws={"fontsize" : 8},
               linewidths=0.5,
               cmap="coolwarm",
               fmt='0.2f')
    chart.set_title(label="Accuracy Matrix")
    plt.yticks(rotation=0)
    plt.savefig("confusionMatrix.png")

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

def oneHot(y, n = 10, dtype=int):
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
	global factor
	validateData = loadInputData('mnist_test.csv')
	XValid = validateData[:, 1:] * factor + 0.01
	YValid = validateData[:, 0]
	confusionMatrix(XValid,YValid)
	acc = predict(XValid,YValid)
	print("Final Accuracy "+ str(acc*100) + "%")

main()