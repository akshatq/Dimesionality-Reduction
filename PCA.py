# Please use Python > 3.7.5
"""
Created on Sun Jan 25 2029

@author: Akshat Chauhan
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as skpp
import scipy.sparse.linalg as ll

np.random.RandomState(seed=30)

# show image function
df=pd.read_csv('food-consumption.csv',delimiter=',',header=0)
#removing \Sweden", \Finland", and \Spain".

df = df[df.Country.isin([i for i in df['Country'] if i not in ['Sweden','Finland','Spain']])]

df=df.set_index('Country')
data=df.values

# normalizing data 
stdA = np.std(data,axis = 0)
stdA = skpp.normalize(stdA.reshape(1,-1)) # the normalize is different from MATLAB's

data = data @ np.diag(np.ones(stdA.shape[1])/stdA[0])

# making data centered around mean 
data=data-np.mean(data,axis = 0)

#PCA using SVD

U, Sigma, VT = np.linalg.svd(data, full_matrices=False) 
print("U:", U.shape)
print("Sigma:", Sigma.shape)
print("VT:", VT.shape)



#Plotting first two components
m, d =data.shape

k = 2

Y_k = np.abs(VT[0:k].T)


#Plotting first two Principal component
fig = plt.figure(figsize=(9, 9))
plt.scatter(Y_k[:,0], Y_k[:,1])
plt.title('Plotting first two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
for x, y, label in zip(Y_k[:,0], Y_k[:,1], df.columns):
    plt.annotate(label, xy=(x, y))





# Plotting projections of data points on first two principal components
fig = plt.figure(figsize=(9, 9))
Y_k = data.dot(VT[0:2, :].T)
plt.scatter(Y_k[:, 0], Y_k[:, 1])
for x, y, label in zip(Y_k[:, 0], Y_k[:, 1], df.index):
    plt.annotate(label, xy=(x, y))
ax = plt.axes()
ax.axis('square')


#####
## Following runs PCA using Eeigen decomposition as shown in demo
#####

data1=df.values

stdA = np.std(data1,axis = 0)
stdA = skpp.normalize(stdA.reshape(1,-1)) # the normalize is different from MATLAB's

data1 = data1 @ np.diag(np.ones(stdA.shape[1])/stdA[0])



mu = np.mean(data1.T,axis = 1)
xc = data1.T - mu[:,None]

# Covariance Matrix

C = np.dot(xc,xc.T)/m

K = 2
S,W = ll.eigs(C,k = K)



dim1 = np.dot(W[:,0].T,xc)/math.sqrt(S[0].real)
dim2 = np.dot(W[:,1].T,xc)/math.sqrt(S[1].real)


dim1=dim1.real
dim2=dim2.real

fig = plt.figure(figsize=(9, 9))

plt.scatter(dim1, dim2)
for x, y, label in zip(dim1, dim2, df.index):
    plt.annotate(label, xy=(x, y))
ax = plt.axes()
ax.axis('square')


