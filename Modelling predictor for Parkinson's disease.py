import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('C:/Users/sawan/Downloads/multiple-disease-prediction-streamlit-app-main/dataset/parkinsons.csv')
# printing the first 5 rows of the dataframe
parkinsons_data.head()
# number of rows and columns in the dataframe
parkinsons_data.shape
# getting more information about the dataset
parkinsons_data.info()
# getting some statistical measures about the data
parkinsons_data.describe()
# distribution of target Variable
parkinsons_data['status'].value_counts()
# grouping the data bas3ed on the target variable
parkinsons_data.groupby('status').mean()
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = GradientBoostingClassifier(n_estimators=100, random_state=2)
#odel = RandomForestClassifier(n_estimators=100, random_state=2)
#model = LogisticRegression()
#model = svm.SVC(kernel='linear')
# training the SVM model with training data
model.fit(X_train, Y_train)
# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
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
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")
import pickle
filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))
for column in X.columns:
  print(column)
# Data Exploration
# Histogram for distribution of target variable
sns.countplot(x='status', data=parkinsons_data)
plt.title('Distribution of Parkinson\'s Disease Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Correlation matrix heatmap
correlation_matrix = parkinsons_data.drop(columns=['name', 'status']).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
# Model Evaluation
# Confusion Matrix
cm = confusion_matrix(Y_test, X_test_prediction)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(Y_test, model.decision_function(X_test))
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
