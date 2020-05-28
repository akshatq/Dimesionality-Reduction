######################################################################
# The codes are based on Python3.7.6

# @version 1.0
# @author Akshat Chauhan
######################################################################
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, find
import matplotlib.image as mpimg
from PIL import Image
from os.path import dirname, join as pjoin
import os
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
from sklearn.utils import graph_shortest_path
import scipy.sparse.linalg as ll
from scipy.spatial import distance


# Please change the p value to 1 for manhattan distance and p=2 for eucledian distance

P=1
#P=2

# Please change eps=22.4 for 100 nearest neighbor threshold and eps=13 for min 6 neighbors

#eps=22.4 # for eucledian with 100 nearest neighbors
#eps=13 # for eucledian with 6 nearest neighbors
eps=1010 # for manhattan with 100 nearest neighbors
#eps= 555 # for manhattan with 6 nearest neighbors

print("Current Working Directory " , os.getcwd())

cur_data_dir = os.getcwd()
mat_fname = pjoin(cur_data_dir, 'isomap.mat')
matFile1 = sio.loadmat(mat_fname)


data = matFile1['images']

data.shape
pixelno = data.shape[0]
imageno = data.shape[1]
data=data.T



A=radius_neighbors_graph(data, eps, mode='connectivity',metric='minkowski',p=P,include_self=False)
A.toarray()
MIN=np.sum(A.toarray(),axis=1)
min(MIN)
MIN.shape
MAX=np.sum(A.toarray(),axis=1)
max(MAX)


x,y=A.toarray().nonzero()[0],A.toarray().nonzero()[1]


edges=[(i,j) for i,j in zip(x,y)]

nodename=range(0,len(data))


G = nx.Graph()
G.add_nodes_from(nodename)
G.add_edges_from(edges)


pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,node_size=7)

## Plotting the images
data1=np.asarray([list(pos.values())])

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
ax.set_title('Representation of Facial Images')


x_size = (max(data1[0][:,0]) - min(data1[0][:,0]))* 0.08
y_size = (max(data1[0][:,1]) - min(data1[0][:,1]))* 0.08
for i in range(50):
    img_num = np.random.randint(0, imageno)
    #img_num=i
    x0 = data1[0][img_num,0] - (x_size / 2.)
    y0 = data1[0][img_num,1] - (y_size / 2.)
    x1 = data1[0][img_num,0] + (x_size / 2.)
    y1 = data1[0][img_num,1] + (y_size / 2.)
    img = data[img_num,:].reshape(64,64).T
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
              interpolation='nearest',zorder=100000,extent=(x0, x1, y0, y1))
ax.scatter(data1[0][:,0],data1[0][:,1], marker='.',alpha=0.7)


plt.show()




### isomap algo

# recalculating radius neighbors but this time with distance matrix
A=radius_neighbors_graph(data, eps, mode='distance',metric='minkowski',p=P,include_self=False)

temp=A.toarray()


D_g=graph_shortest_path.graph_shortest_path(temp,directed=False)
I=np.identity(imageno, dtype=None)
one_m=np.ones((imageno,imageno))
H=I-(one_m/imageno)
H.shape

C=(H@(D_g*D_g)@H)*(-1/2)
s, v = np.linalg.eigh(C)
top_2=np.concatenate((v[:,-1].reshape(imageno,1),v[:,-2].reshape(imageno,1)),axis=1)
Evalue=np.diag(1/np.sqrt(s[[-1,-2]]))


new_data=top_2@Evalue

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
ax.set_title('2D transformation from Isomap of Facial Images')


# Showing 40 of the images on the plot
x_size = (max(new_data[:,0]) - min(new_data[:,0]))* 0.08
y_size = (max(new_data[:,1]) - min(new_data[:,1]))* 0.08



for i in range(300):
    img_num = np.random.randint(0, imageno)
    #img_num=i
    x0 = new_data[img_num,0] - (x_size / 2.)
    y0 = new_data[img_num,1] - (y_size / 2.)
    x1 = new_data[img_num,0] + (x_size / 2.)
    y1 = new_data[img_num,1] + (y_size / 2.)
    img = data[img_num,:].reshape(64,64).T
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
              interpolation='nearest',zorder=100000,extent=(x0, x1, y0, y1))


ax.scatter(new_data[:,0].T,new_data[:,1].T, marker='.',alpha=0.7)

#ax.set_ylabel('Down-Up Pose')
#ax.set_xlabel('Left-Right Pose')

plt.show()

#  find nearest three only

fig = plt.figure()
fig.set_size_inches(10, 10)
#ax=fig.plot()
ax = fig.add_subplot()
ax.set_title('2D transformation from Isomap of Facial Images')

# Showing 40 of the images on the plot
x_size = (max(new_data[:,0]) - min(new_data[:,0]))* 0.08
y_size = (max(new_data[:,1]) - min(new_data[:,1]))* 0.08


# please change for eucledian and manhattan as mentioned below

Y0=-.0000060 # For Manhattan
Y1=-.0000075 # For Manhattan
#Y0=-.0005 # For eucledian
#Y1=-.00045 # For eucledian
three=np.random.randint(0, imageno);
d=distance.cdist(new_data, new_data[three,:].reshape(1,2), 'minkowski', p=P)
d=list(d)
d1=sorted(range(len(d)), key=lambda i: d[i], reverse=False)[:3]
c=1
for i in d1:
    #img_num = np.random.randint(0, imageno)
    img_num=i
    
    x0 = new_data[img_num,0] - (x_size / 2.)
    y0 = new_data[img_num,1] - (y_size / 2.)
    x1 = new_data[img_num,0] + (x_size / 2.)
    y1 = new_data[img_num,1] + (y_size / 2.)
    img = data[img_num,:].reshape(64,64).T
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
              interpolation='nearest',zorder=100000,extent=(x0+c*x_size, x1+c*x_size, Y0,Y1))
    c=c+1
ax.scatter(new_data[:,0].T,new_data[:,1].T, marker='.',alpha=0.7)
ax.scatter(new_data[d1,0].T,new_data[d1,1].T, marker='.',alpha=0.7,color='red')

#ax.set_ylabel('Down-Up Pose')
#ax.set_xlabel('Left-Right Pose')

plt.show()