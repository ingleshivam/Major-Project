import numpy as np
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from svm_classifier import svm_train_and_predict
from sklearn.metrics import accuracy_score

# Load and preprocess data
X_train, X_test, Y_train, Y_test, scaler = load_and_preprocess_data('diabetes.csv')

# Train and predict using SVM
Y_test_pred = svm_train_and_predict(X_train, Y_train.values, X_test)

# Calculate accuracy
test_data_accuracy = accuracy_score(Y_test, Y_test_pred)

input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)

# Predict on new input
prediction = svm_train_and_predict(X_train, Y_train, std_data)

if prediction[0] == 1:
    print('The person is diabetic')
else:
    print('The person is not diabetic')
