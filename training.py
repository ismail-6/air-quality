from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('training.csv')


X = df.drop(columns=['AQI'])
y = df['AQI']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model= XGBRegressor(subsample=0.8,
                           n_estimators= 1100,
                           min_child_weight=3,
                           max_depth=30,
                           learning_rate=0.05)

history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=80)

model.save('Predictor')
