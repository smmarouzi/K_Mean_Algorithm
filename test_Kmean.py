# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:54:13 2017

@author: mm
"""
import pandas as pd
import numpy as np
from numpy import linalg as LA

#read data
df=pd.read_csv('wine.dat', delimiter=" ")

#frequency table for result
df.region.value_counts()

c=df.shape[1]#Number of rows
r=df.shape[0]#Number of columns

#split data
y=df.region #Cluster Labels
x=df.iloc[:,1:c] #Data for Clustering

#data stotastical description
x.describe()#data is normalized

## ---------------------------------------
#how many clusters?
k=input("How many Clusters? ")
k=int(k)
## ---------------------------------------
#create k random centers
center = pd.DataFrame(np.random.normal(0, 1,k*(c-1)).reshape(k,c-1))#c-1 Rows and k columns

## ---------------------------------------
## Eucleadian Distance function
def dist(X,Y):
    #Eucleadian distance between two vectors
    dist = [(a - b)**2 for a, b in zip(X,Y)]
    dist=np.sqrt(sum(dist))
    return dist

## ---------------------------------------
## Cluster Function
def newCluster(x,k,center):
    #x: Data for Clustering
    #k: Number of Clusters
    #center: k Centers vectors
    clusterGroup=list()
    
    for i in range(r):
        distFromCenters=list()
        for j in range(k):
            distFromCenters.append(dist(x.iloc[i,:].values,center.iloc[j,:].values))
        clusterNum=np.where(distFromCenters==min(distFromCenters))[0]
        clusterGroup.extend(clusterNum.tolist())
    
    return clusterGroup

## ---------------------------------------
## Main program
while True: #Iterate the loop until there is just tiny diffrence between Old and New Centers
    
    df['Cluster']=newCluster(x,k,center)
    # find new center
    newCenter=df.groupby('Cluster').mean().drop('region',axis=1)
    #calculate the distance between Old Center and New Center
    distBetweenNewOldCenter=dist(center.values,newCenter.values)
    
    if LA.norm(distBetweenNewOldCenter)<0.06:
        break
    #replace Old center by New center
    center=newCenter

df.Cluster=df['Cluster'].replace(0,3)

## ---------------------------------------
## Compare our Clusters with Regions
# Compute confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cnf_matrix = confusion_matrix(df.region, df.Cluster)

# Plot non-normalized confusion matrix
plt.matshow(cnf_matrix)
print(cnf_matrix)


## ---------------------------------------
## visualization just for 2 dimension
regionlabel=list(set(df.region))
colors=['red','blue','green']

for regionid,color in zip(regionlabel,colors):
    plt.scatter(df[df.region==df.Cluster].iloc[:,2], \
                df[df.region==df.Cluster].iloc[:,5] , c=colors, label=regionid)
plt.show()





















