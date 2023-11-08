# Weather Forecasting with LSTM Neural Network
## Project Overview
This project uses Long Short-Term Memory 
(LSTM) neural networks implemented in TensorFlow
to predict weather conditions such as temperature,
air pressure, visibility,
and wind speed for the next 24 hours.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)

--- 

## Dataset

To train and evaluate the LSTM model, I used a dataset
that includes historical weather data. This dataset contains
information about temperature, air pressure, visibility, wind
speed, dev point and relative humidity measured hourly for a year.
Model was build to predict temperature, air pressure, visibility and wind
speed, because dev point and relative humidity highly depends on temperature and air pressure.
 
Data source: [Kaggle](https://www.kaggle.com/datasets/vidhisrivastava/weather-dataset/data)

---

## Model Architecture
The LSTM model architecture consists of two layers of LSTM cells,
which can capture sequential patterns in the input data, followed by two dense layers.
The final output layer provides predictions for the target weather parameters.

The model takes as input 72 sequences, each with 4 variables and predicts atmospheric conditions
for the next 24. 

---

## Training
The model has been trained using adam optimizer, mean square as loss function, 
13 epoch and batch size equal to 30.

---

## Results

The performance of model was evaluated using MSE and RMSE. 
On train dataset MSE was equal to 0.17 while RMSE 0.42.
On test dataset MSE was equal to 0.89 while RMSE 0.95.

Such MSE and RMSE values suggest that the model is severely over fitted,
although reducing the network size results in the worst performance in the test.

This may be due to the fact that the data from December was used as a test set,
there was no data for that month in the learning set.
This may make the test not completely reliable.
To properly verify the model, data from several years would 
be needed so that it would be possible to use data from several
years to train the model and data from one year as a test set.

In addition, looking at the sample graphs of the model's predictions,
one can see that the model seems to struggle mainly with predicting
visibility and wind speed, which can be quite random and therefore difficult to predict. 

![image](https://github.com/mmadajski/Weather-Forecast/blob/main/images/sample_1.png?raw=true)
---

## Requirements

To run this project, you will need the following dependencies:

- Python v3.11
- TensorFlow v2.14.0
- NumPy v1.26.1
- Pandas v2.1.2
- Matplotlib v3.8.1

You can install the required packages using pip:

```bash
pip install requirements.txt
```

---

## Installation

1. clone this repository to your local computer:
```bash
git clone https://github.com/mmadajski/Weather-Forecast
```
or by downloading the source code from the release version and unzipping it.

2. Install requirements.txt. If you're using pip: 
```bash
pip install requirements.txt
```
3. Download and unzip the data inside projects directory: [Kaggle](https://www.kaggle.com/datasets/vidhisrivastava/weather-dataset/data)
4. You're ready to go.

