import numpy as np
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
import matplotlib.pyplot as plt
import timeit

plt.rcParams.update({'font.size': 36})

batch_size = 128
num_classes = 2
epochs = 20

def defineModel():
	model = Sequential()
	# Input - Layer
	model.add(Dense(500, activation = "relu", input_dim=X_train.shape[1]))
	# Hidden - Layers
	# model.add(Dropout(0.3, noise_shape=None, seed=None))
	model.add(BatchNormalization())
	model.add(Dense(200, activation = "relu"))
	# model.add(Dropout(0.2, noise_shape=None, seed=None))
	model.add(BatchNormalization())
	model.add(Dense(50, activation = "relu"))	
	# model.add(Dropout(0.2, noise_shape=None, seed=None))
	model.add(BatchNormalization())
	model.add(Dense(10, activation = "relu"))
	# Output- Layer
	model.add(Dense(num_classes, activation = "softmax"))
	return model


X_originalFeatures_train = np.load(os.path.join('data', 'featureExtractedTrainData.npy'))
# X_originalFeatures_train = np.load(os.path.join('data', 'exoTrainData_upsampled.npy'))
X_bls_train = np.load(os.path.join('data', 'bls_train.npy'))
X_train = np.concatenate((X_originalFeatures_train,X_bls_train),axis = 1)
# X_train = np.load(os.path.join('data', 'featureExtractedTrainData.npy'))

y_train = np.load(os.path.join('data', 'exoTrainLabels_upsampled.npy'))
y_train = keras.utils.to_categorical(y_train, num_classes)

numTrain = int(X_train.shape[0]*0.8)
X_train, y_train = X_train[0:numTrain,:], y_train[0:numTrain,]
X_valid, y_valid = X_train[numTrain:,:], y_train[numTrain:,]

X_originalFeatures_test = np.load(os.path.join('data', 'featureExtractedTestData.npy'))
# X_test = np.load(os.path.join('data', 'exoTestData.npy'))
X_bls_test = np.load(os.path.join('data', 'bls_test.npy'))
X_test = np.concatenate((X_originalFeatures_test,X_bls_test),axis = 1)
# X_test = np.load(os.path.join('data', 'featureExtractedTestData.npy'))

y_test = np.load(os.path.join('data', 'exoTestLabels.npy'))
y_test = keras.utils.to_categorical(y_test, num_classes)

model = defineModel()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

print(model.summary())

history  = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_valid, y_valid))

prediction = model.predict(X_test)
print("True Breakdown: " + str(np.sum(y_test[:,0])) + " " + str(np.sum(y_test[:,1])))
print("Predicted Breakdown: " + str(np.sum(prediction[:,0])) + " " + str(np.sum(prediction[:,1])))

y_hat = model.predict(X_test)[:,0]

print(prediction)

numExoplanets = 0 
for i in range(prediction.shape[0]):
	if prediction[i,1] > prediction[i,0]:
		numExoplanets += 1

print(numExoplanets)

start_time = timeit.default_timer()
score = model.evaluate(X_test, y_test, verbose=0)
elapsed = timeit.default_timer() - start_time

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Time: " + str(elapsed) + " on " + str(y_test.shape[0]) + " tests\n")

plt.plot(history.history['acc'], label="Training Accuracy", linewidth=2)
# plt.plot(history.history['val_acc'], label="Validation Accuracy", linewidth=2)

plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label="Training Loss", linewidth=2)
# plt.plot(history.history['val_loss'], label="Validation Loss", linewidth=2)

plt.title('Model Losses')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()