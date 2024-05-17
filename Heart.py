import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('C:/Users/sawan/Downloads/multiple-disease-prediction-streamlit-app-main/dataset/heart.csv')
# print first 5 rows of the dataset
heart_data.head()
# print last 5 rows of the dataset
heart_data.tail()
# number of rows and columns in the dataset
heart_data.shape
# getting some info about the data
heart_data.info()
# checking for missing values
heart_data.isnull().sum()
# statistical measures about the data
heart_data.describe()
# checking the distribution of Target Variable
heart_data['target'].value_counts()
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = svm.SVC(kernel='linear')
#model = LogisticRegression(max_iter=1000)
#model = RandomForestClassifier(n_estimators=100, random_state=2)
#model = GradientBoostingClassifier(n_estimators=100, random_state=2)
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)
# Precision
training_precision = precision_score(Y_train, X_train_prediction)
test_precision = precision_score(Y_test, X_test_prediction)

# Recall
training_recall = recall_score(Y_train, X_train_prediction)
test_recall = recall_score(Y_test, X_test_prediction)

# F1 Score
training_f1 = f1_score(Y_train, X_train_prediction)
test_f1 = f1_score(Y_test, X_test_prediction)

# Confusion Matrix
conf_matrix_train = confusion_matrix(Y_train, X_train_prediction)
conf_matrix_test = confusion_matrix(Y_test, X_test_prediction)

print('Precision on Training data : ', training_precision)
print('Precision on Test data : ', test_precision)

print('Recall on Training data : ', training_recall)
print('Recall on Test data : ', test_recall)

print('F1 Score on Training data : ', training_f1)
print('F1 Score on Test data : ', test_f1)

print('Confusion Matrix on Training data:\n', conf_matrix_train)
print('Confusion Matrix on Test data:\n', conf_matrix_test)
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
import pickle
filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))
for column in X.columns:
  print(column)
# Data Exploration
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='target', data=heart_data)
plt.title('Distribution of Target Variable')
plt.show()

# Correlation Matrix
correlation_matrix = heart_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(Y_test,model.decision_function(X_test))
#fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, X_test_prediction)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Heart Disease'], yticklabels=['No Disease', 'Heart Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
