import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('C:/Users/sawan/Downloads/multiple-disease-prediction-streamlit-app-main/dataset/diabetes.csv') 
# printing the first 5 rows of the dataset
diabetes_dataset.head()
# number of rows and Columns in this dataset
diabetes_dataset.shape
# getting the statistical measures of the data
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(x)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
#classifier = GradientBoostingClassifier(n_estimators=100, random_state=2)
#classifier = GaussianNB()
#classifier = RandomForestClassifier(n_estimators=100, random_state=2)
#classifier = KNeighborsClassifier()
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
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
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
import pickle
filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
for column in X.columns:
  print(column)
import seaborn as sns
import matplotlib.pyplot as plt
# Data Exploration
# Histogram for distribution of target variable
sns.countplot(x='Outcome', data=diabetes_dataset)
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# Correlation matrix heatmap
correlation_matrix = diabetes_dataset.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
# Confusion Matrix
cm = confusion_matrix(Y_test, X_test_prediction)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, auc
# Computes the False and True Positive Rate (fpr & tpr) for various threshold values using the decision scores and true labels.
fpr, tpr, _ = roc_curve(Y_test, classifier.decision_function(X_test))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Save the model
filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Prediction on new data
input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = loaded_model.predict(input_data_reshaped)
