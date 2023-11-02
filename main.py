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
test_answer_sequences = np.array([data_test[i:i + hours_future] for i in range(hours_past, data_test.shape[0] - hours_future)])


# Model
lstm_net = tf.keras.Sequential()
lstm_net.add(tf.keras.layers.InputLayer(input_shape=(48, 6)))
lstm_net.add(tf.keras.layers.LSTM(128, activation="relu", return_sequences=True))
lstm_net.add(tf.keras.layers.LSTM(64, activation="relu"))

lstm_net.add(tf.keras.layers.Dense(64))
lstm_net.add(tf.keras.layers.BatchNormalization())
lstm_net.add(tf.keras.layers.Activation("relu"))

lstm_net.add(tf.keras.layers.Dense(144))
lstm_net.add(tf.keras.layers.Reshape([24, 6]))


lstm_net.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
lstm_net.summary()


hist = lstm_net.fit(train_sequences, train_answer_sequences, batch_size=30, epochs=10)

train_prediction = lstm_net.predict(train_sequences)

mse_train = np.mean((train_prediction - train_answer_sequences) ** 2)
print("Mse train: ", mse_train)

test_predict = lstm_net.predict(test_sequences)
mse_test = np.mean((test_predict - test_answer_sequences) ** 2)
print("Mse test: ", mse_test)

