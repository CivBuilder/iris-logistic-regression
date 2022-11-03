import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline

X = np.array([
[7.0,3.2,4.7,1.4],
[6.4,3.2,4.5,1.5],
[6.9,3.1,4.9,1.5],
[5.5,2.3,4.0,1.3],
[6.5,2.8,4.6,1.5],
[5.7,2.8,4.5,1.3],
[6.3,3.3,4.7,1.6],
[4.9,2.4,3.3,1.0],
[6.6,2.9,4.6,1.3],
[5.2,2.7,3.9,1.4],
[5.0,2.0,3.5,1.0],
[5.9,3.0,4.2,1.5],
[6.0,2.2,4.0,1.0],
[6.1,2.9,4.7,1.4],
[5.6,2.9,3.6,1.3],
[6.7,3.1,4.4,1.4],
[5.6,3.0,4.5,1.5],
[5.8,2.7,4.1,1.0],
[6.2,2.2,4.5,1.5],
[5.6,2.5,3.9,1.1],
[5.9,3.2,4.8,1.8],
[6.1,2.8,4.0,1.3],
[6.3,2.5,4.9,1.5],
[6.1,2.8,4.7,1.2],
[6.4,2.9,4.3,1.3],
[6.6,3.0,4.4,1.4],
[6.8,2.8,4.8,1.4],
[6.7,3.0,5.0,1.7],
[6.0,2.9,4.5,1.5],
[5.7,2.6,3.5,1.0],
[5.5,2.4,3.8,1.1],
[5.5,2.4,3.7,1.0],
[5.8,2.7,3.9,1.2],
[6.0,2.7,5.1,1.6],
[5.4,3.0,4.5,1.5],
[6.0,3.4,4.5,1.6],
[6.7,3.1,4.7,1.5],
[6.3,2.3,4.4,1.3],
[5.6,3.0,4.1,1.3],
[5.5,2.5,4.0,1.3],
[5.5,2.6,4.4,1.2],
[6.1,3.0,4.6,1.4],
[5.8,2.6,4.0,1.2],
[5.0,2.3,3.3,1.0],
[5.6,2.7,4.2,1.3],
[5.7,3.0,4.2,1.2],
[5.7,2.9,4.2,1.3],
[6.2,2.9,4.3,1.3],
[5.1,2.5,3.0,1.1],
[5.7,2.8,4.1,1.3],
[6.3,3.3,6.0,2.5],
[5.8,2.7,5.1,1.9],
[7.1,3.0,5.9,2.1],
[6.3,2.9,5.6,1.8],
[6.5,3.0,5.8,2.2],
[7.6,3.0,6.6,2.1],
[4.9,2.5,4.5,1.7],
[7.3,2.9,6.3,1.8],
[6.7,2.5,5.8,1.8],
[7.2,3.6,6.1,2.5],
[6.5,3.2,5.1,2.0],
[6.4,2.7,5.3,1.9],
[6.8,3.0,5.5,2.1],
[5.7,2.5,5.0,2.0],
[5.8,2.8,5.1,2.4],
[6.4,3.2,5.3,2.3],
[6.5,3.0,5.5,1.8],
[7.7,3.8,6.7,2.2],
[7.7,2.6,6.9,2.3],
[6.0,2.2,5.0,1.5],
[6.9,3.2,5.7,2.3],
[5.6,2.8,4.9,2.0],
[7.7,2.8,6.7,2.0],
[6.3,2.7,4.9,1.8],
[6.7,3.3,5.7,2.1],
[7.2,3.2,6.0,1.8],
[6.2,2.8,4.8,1.8],
[6.1,3.0,4.9,1.8],
[6.4,2.8,5.6,2.1],
[7.2,3.0,5.8,1.6],
[7.4,2.8,6.1,1.9],
[7.9,3.8,6.4,2.0],
[6.4,2.8,5.6,2.2],
[6.3,2.8,5.1,1.5],
[6.1,2.6,5.6,1.4],
[7.7,3.0,6.1,2.3],
[6.3,3.4,5.6,2.4],
[6.4,3.1,5.5,1.8],
[6.0,3.0,4.8,1.8],
[6.9,3.1,5.4,2.1],
[6.7,3.1,5.6,2.4],
[6.9,3.1,5.1,2.3],
[5.8,2.7,5.1,1.9],
[6.8,3.2,5.9,2.3],
[6.7,3.3,5.7,2.5],
[6.7,3.0,5.2,2.3],
[6.3,2.5,5.0,1.9],
[6.5,3.0,5.2,2.0],
[6.2,3.4,5.4,2.3],
[5.9,3.0,5.1,1.8]])

Y = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
y_pred_train = clf.predict(X_train)

calibrator = CalibratedClassifierCV(clf, cv='prefit')
model_train= calibrator.fit(X_train, Y_train)
model_test = calibrator.fit(X_test, Y_test)

y_pred_train = model_train.predict(X_train)
y_pred_test = model_test.predict(X_test)

confusion_train = metrics.confusion_matrix(Y_train, y_pred_train)
x_prob_train = model_train.predict_proba(X_train)
x_prob_test = model_test.predict_proba(X_test)

tn = confusion_train[0,0]
fp = confusion_train[0,1]
specificity_train = tn/(fp+tn)
print(confusion_train)

print("Accuracy: ",accuracy_score(Y_train, y_pred_train))
print("Sensitivity : ",recall_score(Y_train, y_pred_train))
print("Precision: ",precision_score(Y_train, y_pred_train))
print("F1: ",f1_score(Y_train, y_pred_train))
print("Log loss: ",log_loss(Y_train, x_prob_train))
print("Specificity: ", specificity_train)
print("Parameter vector:", clf.named_steps['sgdclassifier'].coef_)
print("Bias:", clf.named_steps['sgdclassifier'].intercept_)

confusion_test = metrics.confusion_matrix(Y_test, y_pred_test)
print(confusion_test)

tn = confusion_test[0,0]
fp = confusion_test[0,1]
specificity_test = tn/(fp+tn)

print("Accuracy: ",accuracy_score(Y_test, y_pred_test))
print("Sensitivity : ",recall_score(Y_test, y_pred_test))
print("Precision: ",precision_score(Y_test, y_pred_test))
print("F1: ",f1_score(Y_test, y_pred_test))
print("Log loss: ",log_loss(Y_test, x_prob_test))
print("Specificity: ", specificity_test)
print("Parameter vector:", clf.named_steps['sgdclassifier'].coef_)
print("Bias:", clf.named_steps['sgdclassifier'].intercept_)