### ECG Heartbeat Classification


<p align="center">
  <img src="https://github.com/rnepal2/Time-Series-Problems/blob/main/ECG-Classifier/ECG%20Signal.png" width="300" height="300">
</p>


Data source: [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/), taken in a curated form from [Kaggle](https://www.kaggle.com/shayanfazeli/heartbeat). Each ECG signal with a window of PQRST wave is classified as one of the following five different classes:
- N (0): Normal, 
- S (1): Supra-ventricular premature, 
- V (2): Ventricular premature, 
- F (3): Fusion of Ventricular and Normal
- Q (4): Unclassifiable

Models: Two classification models - Bidirectional LSTM Model and Deep Residual Model, are built that classify similar ECG heartbeat signals into corresponding class type.
