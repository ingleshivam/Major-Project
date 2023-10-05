import numpy as np
import pandas as pd
from custom_scaler import CustomStandardScaler
from custom_train_test_split import custom_train_test_split


def load_and_preprocess_data(data_file):
    diabetes_dataset = pd.read_csv(data_file)

    X = diabetes_dataset.drop(columns='Outcome', axis=1).values
    Y = diabetes_dataset['Outcome'].values

    scaler = CustomStandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)

    X = standardized_data
    Y = diabetes_dataset['Outcome']

    X_train, X_test, Y_train, Y_test = custom_train_test_split(X, Y, test_size=0.2, random_state=2)

    return X_train, X_test, Y_train, Y_test, scaler
