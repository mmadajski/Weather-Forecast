import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Project1WeatherDataset.csv")
data.drop("Weather", axis=1, inplace=True)
data["Date/Time"] = pd.to_datetime(data["Date/Time"])

data_train = data.iloc[:8000].copy()
data_test = data.iloc[8000:].copy()
data_train.drop("Date/Time", axis=1, inplace=True)
data_test.drop("Date/Time", axis=1, inplace=True)

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

hours_future = 24
hours_past = 48

# Train sequences
train_sequences = np.array([data_train[i - hours_past:i] for i in range(hours_past, data_train.shape[0] - hours_future)])
train_answer_sequences = np.array([data_train[i:i + hours_future] for i in range(hours_past, data_train.shape[0] - hours_future)])

# Test sequences
test_sequences = np.array([data_test[i - hours_past:i] for i in range(hours_past, data_test.shape[0] - hours_future)])
test_answer_sequences = np.array([data_test[i:i + hours_future] for i in range(hours_past, data_train.shape[0] - hours_future)])


