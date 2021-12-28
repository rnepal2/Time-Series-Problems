import numpy as np
from scipy.signal import butter, sosfilt

def scaling(signal, scale=0.05):
    '''Introduces some noise into the signal'''
    scale_factor = np.random.normal(loc=1.0, scale=scale, size=(1, signal.shape[1]))
    noise = np.matmul(np.ones((signal.shape[0], 1)), scale_factor)
    return signal * noise

def vertical_flip(signal):
    '''
    Input: signal
    Return: vertically flipped signal
    '''
    return signal[::-1, :]

def shift(signal, interval=20):
    '''
    Input: signal
    Return: shited signal by interval
    '''
    for col in range(signal.shape[1]):
        offset = np.random.choice(range(-interval, interval))/100
        signal[:, col] += offset
    return signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog = False, btype="band", output="sos")
    return sos

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    transformed_signal = np.zeros([signal.shape[0], signal.shape[1]])
    for i in range(signal.shape[1]): 
        transformed_signal[:, i] = sosfilt(sos, signal[:, i])  
    return transformed_signal


def transform(signal, train=True):
    '''
    Input: a signal to transform
    Output: for a training signla: it is randomly transformed and returned 
    '''
    if train:
        rn = np.random.randn()

        if rn < 0.25:
            signal = scaling(signal, scale=0.07)
        elif rn >= 0.25 and rn < 0.50:
            signal = vertical_flip(signal)
        elif rn >= 0.5 and rn < 0.75:
            signal = shift(signal, interval=20)
        else:
            signal = butter_bandpass_filter(signal, 0.05, 48, 256)

    return signal

# multiple transformations at a time
def transform2(signal, train=True):
    '''
    Input: a signal to transform
    Output: for a training signla: it is randomly transformed and returned 
    '''
    if train:
        if np.random.randn() > 0.5: signal = scaling(signal)
        if np.random.randn() > 0.3: signal = vertical_flip(signal)
        if np.random.randn() > 0.5: signal = shift(signal)
        if np.random.randn() > 0.3: signal = butter_bandpass_filter(signal, 0.05, 46, 256)
    return signal
