import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.datasets import fetch_openml



# load the dataset
mnist = fetch_openml('mnist_784')
# view the shape of the dataset
print(mnist.data.shape)
X=mnist.data
y=mnist.target

scaler=StandardScaler()
scaler.fit_transform(X)

pca=PCA()


