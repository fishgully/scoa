import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load data
df = yf.download('AAPL', '2020-01-01', '2023-01-01')
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Prepare data
X = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
X = MinMaxScaler().fit_transform(X)

y = df['Target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = Sequential([
    Dense(64, input_dim=5, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

# plt.figure(figsize=(10,4))
plt.plot(y_test[:100],  marker='o')
plt.plot(y_pred[:100],  marker='x')
# plt.title('Stock Market Trend Prediction (Up=1, Down=0)')
# plt.xlabel('Days (Test Sample)')
# plt.ylabel('Trend')
# plt.legend()
plt.show()