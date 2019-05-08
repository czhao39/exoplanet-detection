from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
from sklearn.metrics import accuracy_score

#############################
# Import train and test data
#############################

X_originalFeatures_train = np.load(os.path.join('data', 'featureExtractedTrainData.npy'))
# X_originalFeatures_train = np.load(os.path.join('data', 'exoTrainData_upsampled.npy'))
X_bls_train = np.load(os.path.join('data', 'bls_train.npy'))
X_train = np.concatenate((X_originalFeatures_train,X_bls_train),axis = 1)
# X_train = np.load(os.path.join('data', 'featureExtractedTrainData.npy'))

y_train = np.load(os.path.join('data', 'exoTrainLabels_upsampled.npy'))

X_originalFeatures_test = np.load(os.path.join('data', 'featureExtractedTestData.npy'))
# X_test = np.load(os.path.join('data', 'exoTestData.npy'))
X_bls_test = np.load(os.path.join('data', 'bls_test.npy'))
X_test = np.concatenate((X_originalFeatures_test,X_bls_test),axis = 1)
# X_test = np.load(os.path.join('data', 'featureExtractedTestData.npy'))

y_test = np.load(os.path.join('data', 'exoTestLabels.npy'))


models = [SVC(gamma='auto', probability = True), 
    RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), 
    DecisionTreeClassifier(random_state=0), 
    MLPClassifier(hidden_layer_sizes=(30,30,30)),
    KNeighborsClassifier(n_neighbors=10)]

modelNames = ["SVC", "Random Forest", "Decision Tree", "MLP", "KNN"]

accs = []
for i in range(len(models)):
    model = models[i]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    # maxVals = np.argmax(probabilities, axis=1)
    # brier_losses.append(brier_score_loss(y_test, maxVals))

    # accuracies.append(100*(np.sum([a == b for a,b in zip(y_test, prediction)])/(y_test.shape[0])))
    # reports.append(str(classification_report(y_test, prediction, target_names=target_names)))

    accs.append(accuracy_score(y_test, prediction)*100)

for acc, name in zip(accs, modelNames):
    print(name + ": " + str(acc))