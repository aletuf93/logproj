# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import seaborn as sns

# load iris sample dataset
iris = datasets.load_iris()
df=pd.DataFrame(iris.data)
df['y']=iris.target
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid','y']
df=df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']]

#data cleaning
df.dropna(how="all", inplace=True) 

sns.set(style="ticks", color_codes=True)
fig=sns.pairplot(df,diag_kind="kde",markers="o")
