import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


print('fetching data ....')

data = np.genfromtxt(r'D:\google_ai_ml\ethusd_price\ETHUSD_1m_Binance.csv' , delimiter=',' , dtype=None , names=True , encoding='utf-8')

open_time =data['Open_time']
close_price = data['Close']

print('fetching data complete.....\nprocessing data......')
# Convert close_time to a datetime object
open_time_datetime = pd.to_datetime(open_time,format=r'%Y-%m-%d %H:%M:%S')



# Extract components (hour and day as an example)
year = open_time_datetime.year
month = open_time_datetime.month
day = open_time_datetime.day
hour = open_time_datetime.hour

filtered_data = np.column_stack((close_price,year,month,hour,day))
scaler = MinMaxScaler(feature_range=(0,1))
normalized_data = scaler.fit_transform(filtered_data,)
#split_index = int(len(filtered_data)*0.8)
split_index = 160000
test_data=normalized_data[170000:190000]
train_data =normalized_data[:split_index]
print(train_data.shape)

sequence_size = 70

train_data_shaped = TimeseriesGenerator(train_data,train_data[:,0],length=sequence_size,batch_size=32)
test_data_shaped = TimeseriesGenerator(test_data,test_data[:,0],length=sequence_size,batch_size=32)

layer = tf.keras.layers

model = tf.keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(sequence_size, 5)),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_squared_error'],
)

model.fit(train_data_shaped , epochs = 5)


predictions = model.predict(test_data_shaped)
tf.keras.models.save_model(
    model, r'D:\google_ai_ml\ethusd_price', overwrite=True,
    include_optimizer=True, save_format=None)

# Reverse scaling for predictions and actual close prices
# We need to pad predictions and actual prices to fit the scaler's expected shape
predicted_prices = scaler.inverse_transform(
    np.hstack((predictions, np.zeros((predictions.shape[0], 4))))
)[:, 0]

actual_prices = scaler.inverse_transform(
    np.hstack((test_data[sequence_size:, 0].reshape(-1, 1), test_data[sequence_size:, 1:]))
)[:, 0]

# Plot the predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label="Actual Prices", color="blue", alpha=0.7)
plt.plot(predicted_prices, label="Predicted Prices", color="red", alpha=0.7)
plt.title("Actual vs. Predicted ETH/USD Prices")
plt.xlabel("Time Step")
plt.ylabel("ETH/USD Price")
plt.legend()
plt.grid()
plt.show()

