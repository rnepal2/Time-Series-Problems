import logging
from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

# tf warning suppression
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class TrackMetrics(tf.keras.callbacks.Callback):
    """
      Callback to plot the learning curves of model during the training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
   
        f, axs = plt.subplots(1, len(metrics), figsize=(12,4))
        clear_output(wait=True)
        
        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if metric != "lr":        # no separate validation plot for learning_rate
                if logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2), 
                                self.metrics['val_' + metric], 
                                label='val_' + metric)
            
            axs[i].set_xlabel("Epoch")
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()
        
        
        
class Trainer:
    '''Custom Trainer class'''
    
    def __init__(self, model, optimizer="adam", learning_rate=0.001, callbacks=None):
        self.model = model
        if optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.model.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        
        if not callbacks:
            self.callbacks = [tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
                              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=5, min_lr=0.0001),
                              tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1),
                              TrackMetrics()]
        else:
            self.callbacks = callbacks
            
    def run(self, X, y, epochs=5, batch_size=256, validation_data: Tuple=None):
        
        if not validation_data:  
            split_ratio = 0.2     # this has problem: takes only last rows of data as validation
        else:
            split_ratio = 0.0
        
        self.model.fit(X, y, 
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=1,
                       validation_split=split_ratio, 
                       validation_data=validation_data,
                       callbacks=self.callbacks
                      )
        return self.model        
 
