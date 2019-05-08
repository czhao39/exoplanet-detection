import numpy as np
import pywt
from numpy import genfromtxt
import os
from astropy.stats import BoxLeastSquares
from scipy.signal import medfilt


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

stat_names = ["depth", "depth_odd", "depth_even", "depth_half", "depth_phased"]
def get_bls_features(x, t):
    features = []
    # print(x[0])
    for i in range(x.shape[0]):
        # adapted from http://docs.astropy.org/en/stable/stats/bls.html#peak-statistics
        bls = BoxLeastSquares(t, x[i])
        periodogram = bls.autopower(40, minimum_n_transit=5)  # arg is the granularity of considered durations
        max_power = np.argmax(periodogram.power)
        stats = bls.compute_stats(periodogram.period[max_power],
                                    periodogram.duration[max_power],
                                    periodogram.transit_time[max_power])
        # TODO: use dataframe?
        features.append([stats[s][0] / stats[s][1] for s in stat_names])
        # based on https://arxiv.org/pdf/astro-ph/0206099.pdf
#         ratios.append(stats["depth"][0] / stats["depth"][1])  # depth over uncertainty
        if (i + 1) % 10 == 0:
            print(".", end="")
        if (i + 1) % 500 == 0:
            print()
    print()
    return np.array(features)

trainData = np.load(os.path.join('data', 'exoTrainData_upsampled.npy'))

testData_filePath = os.path.join('data', 'exoTest.csv')
testData = genfromtxt(testData_filePath, delimiter=',')
testData = testData[1:,:] # get rid of labels

filteredTrain = []
for i in range(trainData.shape[0]):
	rslt = medfilt(trainData[i],kernel_size=17)
	filteredTrain.append(rslt)

trainData = np.array(filteredTrain)

filteredTest = []
for i in range(testData.shape[0]):
	rslt = medfilt(testData[i],kernel_size=17)
	filteredTest.append(rslt)

testData = np.array(filteredTest)

allFeaturifiedData = []
for i in range(trainData.shape[0]):
	currSample = trainData[i][1:]

	currExtractedFeatures = getFeatures(currSample)
	allFeaturifiedData.append(currExtractedFeatures)

np.save(os.path.join('data', 'featureExtractedFilteredTrainData.npy'), np.array(allFeaturifiedData))
# blsFeatures = get_bls_features(trainData, np.arange(trainData.shape[1]))
# np.save(os.path.join('data', 'blsFilteredTrainData.npy'), np.array(blsFeatures))


allFeaturifiedData =[]
for i in range(testData.shape[0]):
	currSample = testData[i][1:]
	currExtractedFeatures = getFeatures(currSample)
	allFeaturifiedData.append(currExtractedFeatures)

np.save(os.path.join('data', 'featureExtractedFilteredTestData.npy'), np.array(allFeaturifiedData))

# blsFeatures = get_bls_features(testData, np.arange(testData.shape[1]))
# np.save(os.path.join('data', 'blsFilteredTestData.npy'), np.array(blsFeatures))