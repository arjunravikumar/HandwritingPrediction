import numpy as np
import matplotlib.pyplot as plt

"""
intialising global variables
"""

maxEpochs = 1000
N = 4
costArr = []

learningRate = 1/N
regParam = 0.8

np.random.seed(10)

numofHiddenNodes = 2

W1 = np.random.normal(0,1, size=(2,2))
B1 = np.random.normal(0,1, size=2)
W2 = np.random.normal(0,1, size=(2,1))
B2 = np.random.normal(0,1, size=1)

"""
function to get data from the file
"""

def loadDataFromFile():
	with open('xor.dat','r') as file:
		inputData = [s.strip().split(',') for s in file.readlines()]
	inputData = np.array(inputData, dtype=np.float16)
	inputData = inputData.astype(int)
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
this function will reshape the weights and the bias to the original shape
input will be 1D array output will be 2D array
"""

def reshapeWtBs(w,b,orgW,orgB):
	w = w.reshape(orgW.shape)
	b = b.reshape(orgB.shape)
	return w,b

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
        mat[i, val] = 1
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
Gradient check function to get the numeric gradient change to compare with the analytical gradient change
"""

def gradientCheck(xloc,yloc,epsilon = 1e-4):
	global W1,B1,W2,B2
	gradentCheckArr = []
	w1 = W1.flatten()
	b1 = B1.flatten()
	w2 = W2.flatten()
	b2 = B2.flatten()
	thetaOriginal = np.concatenate((w1,b1,w2,b2))
	limits = [len(w1),len(w1)+len(b1),len(w1)+len(b1)+len(w2)]
	for j in range(len(thetaOriginal)):
		theta = thetaOriginal.copy()
		theta[j] += epsilon
		wt1,bs1 = reshapeWtBs(w=theta[:limits[0]], b=theta[limits[0]:limits[1]],orgW = W1,orgB = B1)
		wt2,bs2 = reshapeWtBs(w=theta[limits[1]:limits[2]], b=theta[limits[2]:],orgW = W2,orgB = B2)
		a,z = forward(xloc, wt1, bs1,wt2,bs2)
		jPlus = calculateLoss(a[2], yloc)
		theta[j] -= 2*epsilon
		wt1,bs1 = reshapeWtBs(w=theta[:limits[0]], b=theta[limits[0]:limits[1]],orgW = W1,orgB = B1)
		wt2,bs2 = reshapeWtBs(w=theta[limits[1]:limits[2]], b=theta[limits[2]:],orgW = W2,orgB = B2)
		a,z = forward(xloc, wt1, bs1, wt2, bs2)
		jMinus = calculateLoss(a[2], yloc)
		deltaTheta = (jPlus - jMinus)/(2 * epsilon)
		gradentCheckArr.append(deltaTheta)
	dwt1 = np.array(gradentCheckArr[:limits[0]])
	dbs1 = np.array(gradentCheckArr[limits[0]:limits[1]])
	dwt2 = np.array(gradentCheckArr[limits[1]:limits[2]])
	dbs2 = np.array(gradentCheckArr[limits[2]:])
	dw1,db1 = reshapeWtBs(w = dwt1, b = dbs1, orgW = W1,orgB = B1)
	dw2,db2 = reshapeWtBs(w = dwt2, b = dbs2, orgW = W2,orgB = B2)
	return dw1,db1,dw2,db2

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
	plt.savefig("lossepochq1a.png")
	plt.close()

"""
function to compare the analytic and number gradeints and then start to train the model if both of them are correct
"""

def gradientComparison(X,Y):
	global W1,B1,W2,B2,N
	N, D = X.shape
	k = len(set(Y))
	Y = oneHot(y=Y, n_labels=2, dtype=np.float)
	W1 = np.random.normal(0,2, size=(numofHiddenNodes,D))
	B1 = np.random.normal(0,2, size=(numofHiddenNodes,1))
	W2 = np.random.normal(0,2, size=(k,numofHiddenNodes))
	B2 = np.random.normal(0,2, size=(k,1))
	dAW1,dAB1,dAW2,dAB2 = getGradientChange(X, Y)
	dNW1,dNB1,dNW2,dNB2 = gradientCheck(X, Y)
	print('Gradient checking for weight1 is ',end = "")
	numerator = np.linalg.norm(dAW1 - dNW1)
	denominator = np.linalg.norm(dAW1) + np.linalg.norm(dNW1)
	difference = numerator / denominator
	if(difference <= 1e-4):
		print('correct',difference)
	else:
		print('incorrect',difference)
		return

	print('Gradient checking for weight2 is ',end = "")
	numerator = np.linalg.norm(dAW2 - dNW2)
	denominator = np.linalg.norm(dAW2) + np.linalg.norm(dNW2)
	difference = numerator / denominator
	if(difference <= 1e-4):
		print('correct',difference)
	else:
		print('incorrect',difference)
		return

	print('Gradient checking for bias1 is ',end = "")
	numerator = np.linalg.norm(dAB1 - dNB1)
	denominator = np.linalg.norm(dAB1) + np.linalg.norm(dNB1)
	difference = numerator / denominator
	if(difference <= 1e-4):
		print('correct',difference)
	else:
		print('incorrect',difference)
		return

	print('Gradient checking for bias2 is ',end = "")
	numerator = np.linalg.norm(dAB2 - dNB2)
	denominator = np.linalg.norm(dAB2) + np.linalg.norm(dNB2)
	difference = numerator / denominator
	if(difference <=1e-4):
		print('correct',difference)
	else:
		print('incorrect',difference)
		return
	train(X,Y)
	print('Trained model values:')
	print("Weights layer1:" + str(W1))
	print("Bias layer1:"+ str(B1))
	print("Weights layer2:" + str(W2))
	print("Bias layer2:"+ str(B2))

"""
predict function to call the compute the accuracy of the model with the updated weights
"""

def predict(X , Y):
	global W1,B1,W2,B2
	correct = 0
	a , z = forward(X)
	a2 = a[2].T
	for i in range(len(a2)):
		print(Y[i],a2[i])
		if(Y[i] == a2[i].argmax()):
			correct += 1
	print("Final Accuracy: " + str((correct/len(a2))*100)+"%")

"""
main function to start the whole program
"""

def main():
	fileData = loadDataFromFile()
	X = fileData[:, :-1]
	Y = fileData[:, -1]
	gradientComparison(X , Y)
	predict(X , Y)

main()
