import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

import itertools


plt.rcParams.update({'font.size': 60})
plt.rcParams["font.family"] = "fantasy"

class_names  = np.array(['Non-Exoplanet', 'Exoplanet'])

#############################
# Import train and test data
#############################

X_originalFeatures_train = np.load(os.path.join('data', 'featureExtractedTrainData.npy'))
X_bls_train = np.load(os.path.join('data', 'bls_train.npy'))
X_train = np.concatenate((X_originalFeatures_train,X_bls_train),axis = 1)
y_train = np.load(os.path.join('data', 'exoTrainLabels_upsampled.npy'))

X_originalFeatures_test = np.load(os.path.join('data', 'featureExtractedTestData.npy'))
X_bls_test = np.load(os.path.join('data', 'bls_test.npy'))
X_test = np.concatenate((X_originalFeatures_test,X_bls_test),axis = 1)
y_test = np.load(os.path.join('data', 'exoTestLabels.npy'))

def plot_confusion_matrix(cm, classes,
						  title, index,
                          normalize=True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.subplot(2,3,index)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.3)

models = [SVC(gamma='auto', probability = True),
    LinearSVC(random_state=0, tol=1e-5),
    RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), 
    DecisionTreeClassifier(random_state=0), 
    MLPClassifier(hidden_layer_sizes=(30,30,30)),
    KNeighborsClassifier(n_neighbors=10),
    LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')]

modelNames = ["SVC", "LinearSVC", "Random Forest", "Decision Tree", "MLP", "KNN", "Logistic"]

for i in range(len(modelNames)):
    classifier = models[i]
    name = modelNames[i]
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cnf_matrix, classes=class_names, title=name, index=(i+1),  normalize=False)

    plt.show()

    precision, recall, fscore, support = score(y_test, y_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))