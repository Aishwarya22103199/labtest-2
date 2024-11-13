#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import Iris_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrices import recall_score, f1_score, precision_score, accuracy_score

Iris=Iris_dataset()

data=pd.DataFrames(data=Iris.data, coloumn=Iris.feartues_name)
data['target']=Iris.target

print(data)

correlation_matrix=data.corr()
plt.figure(figuresize=(10,8))
sns.heatmap(correlation_matrix, annot='TRUE' cmap="coolwarm" ,fmt='.2f', width='1')
plt.title("correlation matrix")
pli.show()




def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }


    


# In[ ]:


import numpy as np

def 
array=['priya','pandas','sudhw','riyanka','sriyaam']

