import tensorflow as tf


# A simple LSTM classifier
def LSTMModel(seq_length=seq_length, n_class=n_class):
    model = tf.keras.Sequential([
                                 tf.keras.layers.LSTM(256, input_shape=(seq_length, 1)),
                                 tf.keras.layers.Dense(n_class, activation="softmax"),
                                ]
                            )
    return model

 
# Ref: https://arxiv.org/pdf/1805.00794.pdf
def DeepResidualModel(seq_length=seq_length, depth=1, n_class=5):
    
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
