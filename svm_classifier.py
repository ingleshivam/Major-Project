import numpy as np

def svm_train_and_predict(X_train, Y_train, X_test, kernel='linear'):
    n_samples, n_features = X_train.shape

    # Initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0

    # Set learning rate and number of iterations
    learning_rate = 0.01
    num_iterations = 1000

    # Regularization parameter (C)
    C = 1.0

    # Training the SVM model
    for iteration in range(num_iterations):
        for i in range(n_samples):
            condition = Y_train[i] * (np.dot(X_train[i], weights) + bias)
            if condition >= 1:
                weights -= learning_rate * (2 * 1 / num_iterations * C * weights)
            else:
                weights -= learning_rate * (2 * 1 / num_iterations * C * weights - np.dot(X_train[i], Y_train[i]))
                bias -= learning_rate * Y_train[i]

    # Making predictions on test data
    Y_pred = np.sign(np.dot(X_test, weights) + bias)

    return Y_pred
