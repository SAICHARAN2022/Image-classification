import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense,LSTM

# df = pd.read_csv("train_data.csv")


# df["Date"] = pd.to_datetime(df["Date"])


# df = df.sort_values("Date")



# scaler = MinMaxScaler()

# df.rename(columns={"Petrol (USD)":"price"},inplace=True)

# df = df.dropna(subset=["price"])

# # print(df.head())

# prices = df[["price"]].values

# print("Missing prices:", df["price"].isna().sum())

# scaled_prices = scaler.fit_transform(prices)

# # print(scaled_prices)

# sequence_length = 30

# x = [] # 30 days analsis
# y = [] # actual 31 values

# for i in range(sequence_length, len(scaled_prices)):

#     x.append(scaled_prices[i-sequence_length:i])

#     y.append(scaled_prices[i])

# x = np.array(x)
# y = np.array(y)

# # print("X shape:", x.shape)
# # print("y shape:", y.shape)

# # print(len(x[0]))
# # total 783 recors
# train_size = int(len(x) * 0.8)

# X_train = x[:train_size]
# X_test = x[train_size:]
https://github.com/SAICHARAN2022/Image-classification.git

# # print(X_test)

# y_train = y[:train_size]
# y_test = y[train_size:]

# # print(X_train.shape)
# # print(X_test.shape)

# model = Sequential([
#     SimpleRNN(50, activation="tanh", input_shape=(sequence_length, 1)),
#     Dense(1)
# ])

# model.compile(
#     optimizer="adam",
#     loss="mse"
# )

# model.summary()

# # print(model)
# history = model.fit(
#     X_train,
#     y_train,
#     epochs=50,
#     batch_size=32,
#     validation_split=0.1,
#     shuffle=False
# )

# # predictions = model.predict(X_test)

# # predictions = scaler.inverse_transform(predictions)
# # # 
# # actual_prices = scaler.inverse_transform(y_test)

# predictions = model.predict(X_test)

# predictions = scaler.inverse_transform(predictions)
# actual_prices = scaler.inverse_transform(y_test)


# result = pd.DataFrame({
#     "Actual Price": actual_prices.flatten(),
#     "Predicted Price": predictions.flatten()
# })

# print(result)


# LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# -----------------------------
# 1. Load data
# -----------------------------

df = pd.read_csv("train_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df.rename(
    columns={"Petrol (USD)": "price"},
    inplace=True
)

df = df.dropna(subset=["price"])


# -----------------------------
# 2. Get prices
# -----------------------------

prices = df[["price"]].values

print("Missing prices:", df["price"].isna().sum())


# -----------------------------
# 3. Scaling
# -----------------------------

scaler = MinMaxScaler()

scaled_prices = scaler.fit_transform(prices)


# -----------------------------
# 4. Create sequences
# -----------------------------

sequence_length = 30

x = []   # previous 30 days
y = []   # next day

for i in range(sequence_length, len(scaled_prices)):

    x.append(
        scaled_prices[i-sequence_length:i]
    )

    y.append(
        scaled_prices[i]
    )


x = np.array(x)
y = np.array(y)

print("X shape:", x.shape)
print("y shape:", y.shape)


# -----------------------------
# 5. Train/Test split
# -----------------------------

train_size = int(len(x) * 0.8)

X_train = x[:train_size]
X_test = x[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# -----------------------------
# 6. Build LSTM model
# -----------------------------

model = Sequential([
    LSTM(
        50,
        activation="tanh",
        input_shape=(sequence_length, 1)
    ),
    Dense(1)
])


# -----------------------------
# 7. Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="mse"
)


model.summary()


# -----------------------------
# 8. Train
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


# -----------------------------
# 9. Prediction
# -----------------------------

predictions = model.predict(X_test)


# -----------------------------
# 10. Convert back to
#     original petrol prices
# -----------------------------

predictions = scaler.inverse_transform(predictions)

actual_prices = scaler.inverse_transform(y_test)


# -----------------------------
# 11. Create result table
# -----------------------------

result = pd.DataFrame({
    "Actual Price": actual_prices.flatten(),
    "Predicted Price": predictions.flatten()
})


result["Error"] = (
    result["Actual Price"] -
    result["Predicted Price"]
)


print(result)