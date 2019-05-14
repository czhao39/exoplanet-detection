import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from scipy.signal import medfilt
from numpy import genfromtxt
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import timeit

plt.rcParams.update({'font.size': 55})

batch_size = 128
num_classes = 2
epochs = 11

img_rows, img_cols = 47,68
input_shape = (img_rows, img_cols, 1)

# Load training data
X_train = np.load(os.path.join('data', 'exoTrainData_upsampled.npy'))
X_train = X_train[:,0:X_train.shape[1]-1] # get rid of last col to make it easy to convert to img
y_train = np.load(os.path.join('data', 'exoTrainLabels_upsampled.npy'))

# Randomize the order of the training instances
numImages = y_train.shape[0]
indices = random.sample(range(0,numImages), numImages)
X_train = X_train[indices]
y_train = y_train[indices]

# Load testing data
testData_filePath = os.path.join('data', 'exoTest.csv')
testData = genfromtxt(testData_filePath, delimiter=',')
X_test = testData[1:,:] # get rid of labels
print(X_test.shape)
X_test = X_test[:,0:X_test.shape[1]-2] # get rid of last col to make it easy to convert to img
y_test = np.load(os.path.join('data', 'exoTestLabels.npy'))

print(X_train.shape)
print(X_test.shape)
# Smoothen the training/test instances by applying median filtering:
for i in range(X_train.shape[0]):
	X_train[i] = medfilt(X_train[i],kernel_size=17)

for i in range(X_test.shape[0]):
	X_test[i] = medfilt(X_test[i],kernel_size=17)

# Standardize the data
scaler = StandardScaler()
scaler.fit(X_train) # Fit on training set only
X_train = scaler.transform(X_train) # Apply transform to both the training set and the test set
X_test = scaler.transform(X_test)

train_size, test_size = X_train.shape[0], X_test.shape[0]

# Convert training and test instances into images
X_train_temp = np.zeros((train_size, img_rows, img_cols))
X_test_temp = np.zeros((test_size, img_rows, img_cols))

for i in range(train_size):
  X_train_temp[i,:,:]=np.reshape(X_train[i],(img_rows, img_cols))

X_train = X_train_temp.reshape(X_train_temp.shape[0], img_rows, img_cols, 1)

for i in range(test_size):
  X_test_temp[i,:,:]=np.reshape(X_test[i],(img_rows, img_cols))

X_test = X_test_temp.reshape(X_test_temp.shape[0], img_rows, img_cols, 1)

X_train, X_test = X_train.astype('float32'), X_test.astype('float32')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)





###########################
# Define Model:
###########################

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))    
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print("Input 2D with Shape: 47x68")
print("CNN Architecture: ")
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

# print(confusion_matrix(y_test, y_pred))
y_test_temp = []
for i in range(y_test.shape[0]):
	if y_test[i][0] == 1:
		y_test_temp.append(0)
	else:
		y_test_temp.append(1)

prediction = model.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test_temp, prediction[:,1])
print("AUC: " + str(metrics.auc(fpr, tpr)))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

predictions = model.predict_classes(X_test)

y_test_temp = []
for i in range(y_test.shape[0]):
	if y_test[i][0] == 1:
		y_test_temp.append(0)
	else:
		y_test_temp.append(1)
print(confusion_matrix(y_test_temp, predictions))

start_time = timeit.default_timer()
score = model.evaluate(X_test, y_test, verbose=0)
elapsed = timeit.default_timer() - start_time

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Time: " + str(elapsed) + " on " + str(y_test.shape[0]) + " tests")

# Plot training & validation accuracy values
plt.plot(history.history['acc'], linewidth=6)
plt.plot(history.history['val_acc'], linewidth=6)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], linewidth=6)
plt.plot(history.history['val_loss'], linewidth=6)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()