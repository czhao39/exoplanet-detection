import numpy as np
from numpy import genfromtxt
import os
import random

oversampling_factor = 135

def upsample(dataInstance):
	allData = []
	dataLabels = []
	for i in range(oversampling_factor):
		rotation = random.randint(1,dataInstance.shape[0])
		dataInstance = np.roll(dataInstance, rotation)
		allData.append(dataInstance)
		dataLabels.append(1)
	return [np.array(allData), dataLabels]

trainData_filePath = os.path.join('data', 'exoTrain.csv')
trainData = genfromtxt(trainData_filePath, delimiter=',')
trainData = trainData[1:,:] # get rid of labels
trainLabels = trainData[:,0]
trainLabels = [0 if x == 1.0 else 1 for x in trainLabels]
trainData = np.delete(trainData, 0, 1)

# testData_filePath = os.path.join('data', 'exoTest.csv')
# testData = genfromtxt(testData_filePath, delimiter=',')
# testData = testData[1:,:] # get rid of labels
# testLabels = testData[:,0]
# testLabels = [0 if x == 1.0 else 1 for x in testLabels]
# testData = np.delete(testData, 0, 1)

isFirst = 1
newALLXTrain = []
newALLyTrain = []
for i in range(len(trainLabels)):
	if trainLabels[i] == 1:
		[newTrainInstances, newLabels] = upsample(trainData[i])
		if isFirst:
			newALLXTrain, newALLyTrain = newTrainInstances, newLabels
			isFirst = 0
		else:
			newALLXTrain = np.concatenate((newALLXTrain, newTrainInstances), axis = 0)
			newALLyTrain.extend(newLabels)
print("UPSAMPLED STUFF: ")
print(newALLXTrain.shape)
print(len(newALLyTrain))

print("ORIGINAL DATA: ")
print(trainData.shape)
print(len(trainLabels))

trainData = np.concatenate((trainData, newALLXTrain), axis = 0)
trainLabels.extend(newALLyTrain)

print("NEW DATA: ")
print(trainData.shape)
print(len(trainLabels))

np.save(os.path.join('data', 'newUpsampledXTrain.npy'), trainData)
np.save(os.path.join('data', 'newUpsampledYTrain.npy'), trainLabels)