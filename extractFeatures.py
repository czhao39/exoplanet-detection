import numpy as np
import pywt
from numpy import genfromtxt
import os



def getFeatures(signal):

	signal = np.array(signal)

	# Fourier transform
	fft = np.real(np.fft.rfft(signal))
	a = fft[0]
	b = fft[1]
	c = fft[2]
	d = fft[3]

	# Discrete Wavelet transform
	cA, _ = pywt.dwt(signal, 'db1')
	cB, _ = pywt.dwt(signal, 'db4')
	wt1 = cA[1]
	wt2 = cA[2]
	wt3 = cB[1]
	wt4 = cB[2]

	return [a, b, c, d, wt1, wt2, wt3, wt4]



trainData = np.load(os.path.join('data', 'exoTrainData_upsampled.npy'))

testData_filePath = os.path.join('data', 'exoTest.csv')
testData = genfromtxt(testData_filePath, delimiter=',')
testData = testData[1:,:] # get rid of labels

allFeaturifiedData =[]
for i in range(trainData.shape[0]):
	currSample = trainData[i][1:]
	currExtractedFeatures = getFeatures(currSample)
	allFeaturifiedData.append(currExtractedFeatures)

np.save(os.path.join('data', 'featureExtractedTrainData.npy'), np.array(allFeaturifiedData))

allFeaturifiedData =[]
for i in range(testData.shape[0]):
	currSample = testData[i][1:]
	currExtractedFeatures = getFeatures(currSample)
	allFeaturifiedData.append(currExtractedFeatures)

np.save(os.path.join('data', 'featureExtractedTestData.npy'), np.array(allFeaturifiedData))