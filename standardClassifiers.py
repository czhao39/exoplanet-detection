from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
#############################
# Import train and test data
#############################

X_originalFeatures_train = np.load(os.path.join('data', 'featureExtractedTrainData.npy'))
X_bls_train = np.load(os.path.join('data', 'bls_train.npy'))
X_train = np.concatenate((X_originalFeatures_train,X_bls_train),axis = 1)
y_train = np.load(os.path.join('data', 'newUpsampledYTrain.npy'))

X_originalFeatures_test = np.load(os.path.join('data', 'featureExtractedFilteredTestData.npy'))
X_bls_test = np.load(os.path.join('data', 'bls_test.npy'))
X_test = np.concatenate((X_originalFeatures_test,X_bls_test),axis = 1)
y_test = np.load(os.path.join('data', 'exoTestLabels.npy'))


models = [SVC(gamma='auto', probability = True),
    LinearSVC(random_state=0, tol=1e-5),
    RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), 
    DecisionTreeClassifier(random_state=0), 
    MLPClassifier(hidden_layer_sizes=(30,30,30)),
    KNeighborsClassifier(n_neighbors=10),
    LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')]

modelNames = ["SVC", "LinearSVC", "Random Forest", "Decision Tree", "MLP", "KNN", "Logistic"]


accs = []
reports = []
target_names =['Negative', 'Positive']

for i in range(len(models)):
    model = models[i]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    accs.append(accuracy_score(y_test, prediction)*100)
    print()
    reports.append(str(classification_report(y_test, prediction, target_names=target_names)))

    

    print(modelNames[i])
    print(accuracy_score(y_test, prediction)*100)

    print(f1_score(y_test, prediction, average='binary'))
    print(confusion_matrix(y_test, prediction))

