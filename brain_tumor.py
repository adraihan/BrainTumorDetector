# -*- coding: utf-8 -*-
"""Brain Tumor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-xR6sM0ovp3hI4rLuueSahtPXymHAZUG

**DATA PRAPROSES**

*Import Library*
"""

import sklearn
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""*Upload Data*"""

df = pd.read_csv('brain_tumor.csv')
df.head()

"""*Normalisasi Data*"""

# drop kolom yang tidak diperlukan
data = df.drop(columns=['Image'])


"""*Split Datasets*"""

from sklearn.model_selection import train_test_split

X = data.iloc[:,1:14]
Y = data.iloc[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


"""**PELATIHAN MODEL**

*logistic Regression*
"""

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)
sv = log.fit(X_train, Y_train)

"""*Test Akurasi*"""

log.score(X_train, Y_train)

log.predict([[7.341095,	1143.808219,	33.820234,	0.001467,	5.061750,	26.479563,	81.867206,	0.031917,	0.001019,	0.268275,	5.981800,	0.978014,	7.458341e-155]])


pickle.dump(sv, open('brain_tumor.pkl', 'wb'))