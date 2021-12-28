### ECG Heartbeat Classification


<p align="center">
  <img src="https://github.com/rnepal2/Time-Series-Problems/blob/main/ECG-Classifier/ECG%20Signal.png" width="300" height="300">
</p>


Data source is: [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/), taken in a curated form from a kaggle competition. Each ECG window of a PQRS wave is classified as one of the five different classes:
- N (0): Normal, 
- S (1): Supra-ventricular premature, 
- V (2): Ventricular premature, 
- F (3): Fusion of Ventricular and Normal
- Q (4): Unclassifiable

Classification models (Bidirectional LSTM and Deep Residual) are built that can classify similar ECG heartbeat signals into the corresponding class type.
