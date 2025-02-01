import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = 'dataset/letters.csv'

# x is list of x, y, z coordinates -- y is letter value
x = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 3) + 1)))
y = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 3, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(30, activation='relu'),
    # 24 possibilities / 24 classes
    tf.keras.layers.Dense(24, activation='softmax')
])

# model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x_train,
    y_train,
    epochs=160,
    batch_size=128,
    validation_data=(x_test, y_test)
)
# ~87% accuracy on average

model.save('/Users/josephdavis/PycharmProjects/MediaPipe_Test/model.keras')