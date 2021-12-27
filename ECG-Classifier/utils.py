import os
import itertools
import numpy as np   
import matplotlib.pyplot as plt   


def plot_confusion_matrix(cm, classes, normalize=False):
    """
        Input: 
            cm = confusion matrix, list of classes, 
            If normalize = True: plots the normalized confusion matrix, 
                           Otherwise absolute numbers
        Output:
            Returns the confusion matrix
        
    """
    cm = np.array(cm)
    n_class = len(classes)
    if normalize:
        np.set_printoptions(precision=3)       
        ncm = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                ncm[i, j] = cm[i, j]/sum(cm[i, :])
        cm = ncm

    vmin, vmax = min(cm.flatten()), max(cm.flatten())   
    plt.figure(figsize=(8, 8))
    img = plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=vmin, vmax=vmax)
    plt.title("Confusion matrix")
    plt.colorbar(img, shrink=0.7)
    
    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, classes, rotation=0, fontsize=14)
    plt.yticks(tick_marks, classes, rotation=90, fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 20.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=14, 
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
