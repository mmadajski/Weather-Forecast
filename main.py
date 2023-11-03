import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Data read
data = pd.read_csv("Project1WeatherDataset.csv")

"""
Removing the "Weather" column because it describes the weather based on other columns that I am trying to predict.
Similarly, the "Dew Point Temp_C" and "Rel Hum_%" columns have been removed,
as they depend heavily on temperature and atmospheric pressure.
"""

data.drop("Weather", axis=1, inplace=True)
data.drop("Dew Point Temp_C", axis=1, inplace=True)
data.drop("Rel Hum_%", axis=1, inplace=True)
data["Date/Time"] = pd.to_datetime(data["Date/Time"])

# Data split
data_train = data.iloc[:8000].copy()
data_test = data.iloc[8000:].copy()
data_train.drop("Date/Time", axis=1, inplace=True)
data_test.drop("Date/Time", axis=1, inplace=True)
names = data_train.columns

# Scaling data
scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

hours_future = 24
hours_past = 72

# Train sequences
train_sequences = np.array([data_train[i - hours_past:i] for i in range(hours_past, data_train.shape[0] - hours_future)])
train_answer_sequences = np.array([data_train[i:i + hours_future] for i in range(hours_past, data_train.shape[0] - hours_future)])

# Test sequences
test_sequences = np.array([data_test[i - hours_past:i] for i in range(hours_past, data_test.shape[0] - hours_future)])
test_answer_sequences = np.array([data_test[i:i + hours_future] for i in range(hours_past, data_test.shape[0] - hours_future)])


# Model
lstm_net = tf.keras.Sequential()
lstm_net.add(tf.keras.layers.InputLayer(input_shape=(72, 4)))
lstm_net.add(tf.keras.layers.LSTM(100, activation="relu", return_sequences=True))
lstm_net.add(tf.keras.layers.LSTM(50, activation="relu"))

lstm_net.add(tf.keras.layers.Dense(64))
lstm_net.add(tf.keras.layers.BatchNormaliation())
lstm_net.add(tf.keras.layers.Activation("relu"))

lstm_net.add(tf.keras.layers.Dense(96))
lstm_net.add(tf.keras.layers.Reshape([24, 4]))


lstm_net.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
lstm_net.summary()

# Training
hist = lstm_net.fit(train_sequences, train_answer_sequences, batch_size=30, epochs=13)

plt.plot(hist.history["loss"])
plt.savefig("loss.png")
plt.clf()

train_prediction = lstm_net.predict(train_sequences)

mse_train = np.mean((train_prediction - train_answer_sequences) ** 2)
print("Mse train: ", mse_train)
print("Rmse train: ", np.sqrt(mse_train))

test_predict = lstm_net.predict(test_sequences)
mse_test = np.mean((test_predict - test_answer_sequences) ** 2)
print("Mse test: ", mse_test)
print("Rmse train: ", np.sqrt(mse_test))

for i in range(0, 10):
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout(pad=2.0)

    sequence = test_answer_sequences[i * 50]
    predicted_sequence = test_predict[i * 50]

    for j in range(0, 2):
        for k in range(0, 2):
            axes[k][j].set_title(names[j * 2 + k])
            axes[k][j].plot(sequence[:, j * 2 + k], "b", label="True")
            axes[k][j].plot(predicted_sequence[:, j * 2 + k], "r", label="Predicted")
            axes[k][j].legend()

    fig.savefig("sample_" + str(i) + ".png")
    fig.clf()