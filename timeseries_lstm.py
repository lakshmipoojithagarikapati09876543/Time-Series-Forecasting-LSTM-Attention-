import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention

# Dummy data: sequences of length 10, 1 feature, predicting next step
X = np.random.rand(1000, 10, 1)
y = np.random.rand(1000, 1)

input_seq = Input(shape=(10,1))
lstm_out = LSTM(64, return_sequences=True)(input_seq)
attention = Attention()([lstm_out, lstm_out])
context_vector = tf.reduce_sum(attention, axis=1)
output = Dense(1)(context_vector)

model = Model(inputs=input_seq, outputs=output)
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10)
