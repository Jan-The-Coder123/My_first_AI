import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x_train = tf.constant([5, 10, 15, 20, 25], dtype=tf.float32)
y_train = tf.constant([50, 100, 150, 200, 250], dtype=tf.float32)

model = keras.Sequential([
    layers.Dense(200, activation='relu', input_shape=(1,)),
    layers.Dense(100),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, verbose=0)
print('learning done')

x_test = tf.constant([500, 1000, 1500, 2000, 2500], dtype=tf.float32)
predictions = model.predict(x_test)

print('inputs:', x_test)
tf.print('predictions:', predictions.flatten())
