# TextClassification
Text Classification in Python Using PyDev

### Data:
Amazon Reviews for Alexa are being categorised into 3 categories to indicate where the user sentiment were Positive, Negative or Neutral

### Preprocessing:
So far the problem is treated in a very simple manner therefore only a few basic preprocessing techinques are practiced. To convert the textual data into numerical data,TF-IDF was used.

### ANN:
For Classification ANN is text classification.
Confusion Matrix: Neural Network

### Results:
          +    =   -
Positive [[28  0  2]
Neutral  [ 7  8 13]
Negative [ 0  0 30]]

             precision    recall  f1-score   support

 Positive       0.80      0.93      0.86        30
 Neutral        1.00      0.29      0.44        28
 Negative       0.67      1.00      0.80        30

avg / total     0.82      0.75      0.71        88
