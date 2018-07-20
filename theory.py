# http://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
data, labels = make_blobs(n_samples=100, n_features=2, centers=2,cluster_std=4,random_state=2)
plt.scatter(data[:,0], data[:,1], c = labels, cmap='coolwarm')
plt.show()

#Import LinearSVC
from sklearn.svm import LinearSVC

#Create instance of Support Vector Classifier
svc = LinearSVC()

#Fit estimator to 70% of the data
svc.fit(data[:70], labels[:70])

#Predict final 30%
y_pred = svc.predict(data[70:])

#Establish true y values
y_true = labels[70:]

# TP – True Positives
# FP – False Positives
# Precision – Accuracy of positive predictions.
# Precision = TP/(TP + FP)
from sklearn.metrics import precision_score
print("Precision score: {}".format(precision_score(y_true,y_pred)))

# FN – False Negatives
# Recall (aka sensitivity or true positive rate): Fraction of positives That were correctly identified.
# Recall = TP/(TP+FN)
from sklearn.metrics import recall_score
print("Recall score: {}".format(recall_score(y_true,y_pred)))

# F1 Score (aka F-Score or F-Measure) – A helpful metric for comparing two classifiers. 
# F1 Score takes into account precision and the recall. 
# It is created by finding the the harmonic mean of precision and recall.
# F1 = 2 x (precision x recall)/(precision + recall)
from sklearn.metrics import f1_score
print("F1 Score: {}".format(f1_score(y_true,y_pred)))

#Classification Report
#Report which includes Precision, Recall and F1-Score.
from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))

#Confusion Matrix
#Confusion matrix allows you to look at the particular misclassified examples yourself and 
#perform any further calculations as desired.
from sklearn.metrics import confusion_matrix
import pandas as pd

confusion_df = pd.DataFrame(confusion_matrix(y_true,y_pred),
             columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
             index = ["Class " + str(class_name) for class_name in [0,1]])

print(confusion_df)



