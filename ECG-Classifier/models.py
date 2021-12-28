import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import LSTM, Bidirectional

# A simple LSTM classifier
def LSTMModel(seq_length=187, depth=1, n_class=5):
    
    model = tf.keras.Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=(seq_length, depth))))
    model.add(Dropout(rate=0.25))
    model.add(Dense(n_class, activation="softmax"))
    
    return model

 
# Ref: https://arxiv.org/pdf/1805.00794.pdf
def DeepResidualModel(seq_length=187, depth=1, n_class=5):
    
    inputs = tf.keras.layers.Input(shape=(seq_length, depth))
    
    out1 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1)(inputs)
    
    for _ in range(4):
    
        out = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(out1)
        out = tf.keras.layers.Activation("relu")(out)
        out = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(out)
        out = tf.keras.layers.Add()([out, out1])
        out = tf.keras.layers.Activation("relu")(out)
        out1 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2)(out)
        
    out = tf.keras.layers.Flatten()(out1)
    out = tf.keras.layers.Dense(32)(out)
    out = tf.keras.layers.Activation("relu")(out)
    out = tf.keras.layers.Dense(32)(out)
    out = tf.keras.layers.Dense(n_class)(out)
    out = tf.keras.layers.Softmax()(out)
    
    return tf.keras.Model(inputs=inputs, outputs=out)
