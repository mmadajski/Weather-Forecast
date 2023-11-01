import pandas as pd
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


