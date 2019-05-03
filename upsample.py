import numpy as np


DATA_PATH = "data/"
TRAIN_FILE = "exoTrain.csv"
TEST_FILE = "exoTest.csv"
UPSAMPLE_MULT = 10  # must be a whole number


train_data = np.loadtxt("{}{}".format(DATA_PATH, TRAIN_FILE), skiprows=1, delimiter=',')
x_train = train_data[:, 1:]
print("x_train shape:", x_train.shape)
y_train = (train_data[:, 0] - 1).flatten().astype(int)
print("y_train freqs:")
print(np.unique(y_train, return_counts=True))

x_train_exo = x_train[y_train == 1]
x_train = np.vstack((x_train, np.repeat(x_train_exo, UPSAMPLE_MULT - 1, axis=0)))
print("upsampled x_train shape:", x_train.shape)
y_train_exo = np.ones(x_train_exo.shape[0])
y_train = np.concatenate((y_train, np.repeat(y_train_exo, UPSAMPLE_MULT - 1, axis=0)))
print("upsampled y_train freqs:")
print(np.unique(y_train, return_counts=True))

print("Writing upsampled train files...")
filename_base = TRAIN_FILE[:TRAIN_FILE.rindex(".")]
np.save("{}{}Data_upsampled".format(DATA_PATH, filename_base), x_train)
np.save("{}{}Labels_upsampled".format(DATA_PATH, filename_base), y_train)


test_data = np.loadtxt("{}{}".format(DATA_PATH, TEST_FILE), skiprows=1, delimiter=',')
x_test = test_data[:, 1:]
print("x_test shape:", x_test.shape)
y_test = (test_data[:, 0] - 1).flatten().astype(int)
print("y_test freqs:")
print(np.unique(y_test, return_counts=True))

print("Writing non-upsampled test files...")
filename_base = TEST_FILE[:TEST_FILE.rindex(".")]
np.save("{}{}Data".format(DATA_PATH, filename_base), x_test)
np.save("{}{}Labels".format(DATA_PATH, filename_base), y_test)
