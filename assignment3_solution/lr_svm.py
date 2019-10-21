import numpy as np
from numpy import linalg as LA
import pandas as pd
train_FILE = 'trainingSet.csv'
test_FILE = 'testSet.csv'

trainingSet = pd.read_csv(train_FILE)
testSet = pd.read_csv(test_FILE)
# def sigmoid(z):
# 		return 1 / (1 + np.exp(-z))
# trainX = trainingSet.iloc[:,:-1]
# trainY = trainingSet['decision']
# W = np.zeros(trainX.shape[1])
# gradient = np.zeros(trainX.shape[1])
# z = np.dot(trainX, W)
# h = sigmoid(z)
# reg_constant = 0.01
# # print(h)

# for j in range(len(W)):
# 	gradient[j] = np.dot(trainX.iloc[:,j].T,(h-trainY)) + reg_constant * W[j]

# print(gradient)


def lr(trainingSet, testSet):
	max_iter = 500
	threshold = 1e-6
	trainX = trainingSet.iloc[:,:-1]
	trainY = trainingSet['decision']
	testX = testSet.iloc[:,:-1]
	testY = testSet['decision']
	W = np.zeros(trainX.shape[1])
	gradient = np.zeros(trainX.shape[1])
	reg_constant = 0.01
	step = 0.01

	def sigmoid(z):
		return 1 / (1 + np.exp(-z))
	weight_difference = float('inf')
	i = 0
	while i < max_iter and weight_difference>=threshold:
		W_new = np.zeros(trainX.shape[1])
		z = np.dot(trainX, W)
		h = sigmoid(z)
		for j in range(len(W)):
			gradient[j] = np.dot(trainX.iloc[:,j].T,(h-trainY)) + reg_constant * W[j]
			W_new[j]  = W[j] - step*gradient[j]
		curr_distance = LA.norm(W_new-W)
		if curr_distance < weight_difference:
			weight_difference = curr_distance
		W = W_new
		i+=1

	def predict(X):
		prob = sigmoid(np.dot(X, W))
		if prob[i]>=0.5:
			prob[i] = 1
		else:
			prob[i] = 0
		return prob
	def calculate_accuracy(prediction, label):
		count = 0
		for i in range(len(prediction)):
			if prediction[i] == label[i]:
				count+=1
		return round(float(count)/len(prediction),2)

	train_pred = predict(trainX)
	print('Training accuracy LR: ',calculate_accuracy(train_pred,trainY))
	test_pred = predict(testX)

	print('Test accuracy LR: ',calculate_accuracy(test_pred,testY))


def svm(trainingSet,testSet):
	max_iter = 500
	threshold = 1e-6
	trainX = trainingSet.iloc[:,:-1]
	trainY = trainingSet['decision']
	testX = testSet.iloc[:,:-1]
	testY = testSet['decision']
	W = np.zeros(trainX.shape[1])
	gradient = np.zeros(trainX.shape[1])
	reg_constant = 0.01
	step = 0.5
	weight_difference = float('inf')
	i = 0
	while i < max_iter and weight_difference>=threshold:
		W_new = np.zeros(trainX.shape[1])
		h = np.dot(trainX,W)
		for j in range(trainX.shape[1]):
			sum_gradient = 0
			for i in range(trainX.shape[0]):
				if h[i]*trainY[i]<0:
					sum_gradient+=reg_constant*W[j]-trainX[i]*trainX.iloc[i,j]
				else:
					sum_gradient+=reg_constant*W[j]

			gradient[j] = sum_gradient/trainX.shape[0]
			W_new[j]  = W[j] - step*gradient[j]
		curr_distance = LA.norm(W_new-W)
		if curr_distance < weight_difference:
			weight_difference = curr_distance
		W = W_new
		i+=1


	def predict(X):
		prob = np.sign(np.dot(X, W))
		return prob.astype(int)
	def calculate_accuracy(prediction, label):
		count = 0
		for i in range(len(prediction)):
			if prediction[i] == label[i]:
				count+=1
		return round(float(count)/len(prediction),2)

	train_pred = predict(trainX)
	print('Training accuracy SVM: ',calculate_accuracy(train_pred,trainY))
	test_pred = predict(testX)

	print('Test accuracy SVM: ',calculate_accuracy(test_pred,testY))

# svm(trainingSet, testSet)
lr(trainingSet,testSet)
