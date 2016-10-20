# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 01:03:56 2016

@author: haopingbai
"""
from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from scipy.spatial.distance import cdist, pdist

import matplotlib.cm as cm

logs = open("level2_log.txt", "w+")

#data = pd.read_csv("ThermoData.csv")
data = pd.read_csv("level2.csv")

data["min_e"] = (data["min_k"] * data["min_d"] * data["min_c"]) ** 0.5
data['max_e'] = (data["max_k"] * data["max_d"] * data["max_c"]) ** 0.5

pos = np.arange(0.5,22.5,0.5)
# Initialise plot
f = plt.figure(1)
 
# ax = fig.add_axes([0.15,0.2,0.75,0.3]) #[left,bottom,width,height]
ax = f.add_subplot(111)
 
# Plot the data
for i in range(0, data['Material'].size):
    ax.barh((i*0.5)+0.5, data['max_e'][i] - data['min_e'][i], left=data['min_e'][i], height=0.3, align='center', color='blue', alpha = 0.75)
 
# Format the y-axis
 
locsy, labelsy = plt.yticks(pos,data['Material'])
plt.setp(labelsy, fontsize = 14)
 
# Format the x-axis
 
#ax.set_xlim(xmin = 0, xmax = 1.9e-4)
#ax.set_xlim(xmin = 1e-8, xmax = 1e-3)
ax.set_ylim(ymin = -0.1, ymax = 40.5)
ax.grid(color = 'g', linestyle = ':')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

#ax.set_xscale('log')
ax.set_title('Thermal Effusivity of Materials')
ax.set_xlabel('Thermal Effusivity')
ax.set_ylabel('Materials')

f.show()

pos = np.arange(0.5,14.5,0.5)
# Initialise plot
g = plt.figure(2)
 
# ax = fig.add_axes([0.15,0.2,0.75,0.3]) #[left,bottom,width,height]
ax = g.add_subplot(111)
 
# Plot the data
y_pos = 0
ylabels = []
for i in range(0, data['Material'].size):
    if data['max_e'][i] < 5e3:
        ax.barh((y_pos*0.5)+0.5, data['max_e'][i] - data['min_e'][i], left=data['min_e'][i], height=0.3, align='center', color='blue', alpha = 0.75)
        y_pos += 1
        ylabels.append(data['Material'][i])
 
# Format the y-axis
 
locsy, labelsy = plt.yticks(pos,ylabels)
plt.setp(labelsy, fontsize = 14)
 
# Format the x-axis
 
#ax.set_xlim(xmin = 0, xmax = 3e-6)
ax.set_ylim(ymin = -0.1, ymax = 14.5)
ax.grid(color = 'g', linestyle = ':')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

ax.set_title("Thermal Effusivity of Selected Materials")
ax.set_xlabel('Thermal Effusivity')
ax.set_ylabel('Materials')

g.show()

X = []
y = []

for i in range(0, data['Material'].size):
    for samples in range(1, 25):
        eff = data['min_e'][i] + np.random.sample() * (data['max_e'][i] - data['min_e'][i])
        X.append([eff])
        y.append(i)
        
X = np.array(X)

range_n_clusters = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 30, 50]
silhouette_scores = []
centroids = []

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_scores.append(silhouette_avg)
          
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X, X, marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    centroids.append(centers)
    # Draw white circles at cluster centers
    ax2.scatter(centers, centers,
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c, c, marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for effusivity")
    ax2.set_ylabel("Feature space for effusivity")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.show()
    
D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/X.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X)**2)/X.shape[0]
bss = tss-wcss

kIdx = 10-1

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range_n_clusters, avgWithinSS, 'b*-')
ax.plot(range_n_clusters[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range_n_clusters, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range_n_clusters, silhouette_scores, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for n Clusters')

for elbow in range_n_clusters:
    print("------------------------------------------------------------", file=logs)
    print("----------------", elbow, " CLUSTERS:", "-----------------------", file=logs)
    #Train the clustering model
    clusterer = KMeans(n_clusters=elbow, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    #Test the model
    distribution = []
    cluster_info = []
    max_list = []
    SAMPLE_NUMBER = 200

    for i in range(0, elbow):
        arr = np.zeros(data['Material'].size)
        cluster_info += [arr]
    for i in range(0, data['Material'].size):
        temp = np.zeros(elbow)
        for samples in range(1, SAMPLE_NUMBER + 1):
            label = clusterer.predict(data['min_e'][i] + np.random.sample() * (data['max_e'][i] - data['min_e'][i]))
            temp[label] += 1
        for j in range(0, len(temp)):
            cluster_info[j][i] = temp[j]

        distribution += [temp]
        max_list += [np.argmax(temp)]

    ranking = sorted(range(len(max_list)), key=lambda k: distribution[k][max_list[k]])

    print("Material ranking on", elbow, "- cluster model:", file=logs)
    print("\n", file=logs)
    for i in range(0, len(ranking)):
        material = data['Material'][ranking[i]]
        print("  #", i + 1, ": ", material, "\nMaximum distribution in one cluster: ", 100 * distribution[ranking[len(ranking) - i - 1]][max_list[ranking[len(ranking) - i - 1]]] / 200.0, "%", file=logs)
        
        for j in range(0, elbow):
            print("    Cluster #", j, ": ", 100 * distribution[ranking[len(ranking) - i - 1]][j] / 200.0, "%", file=logs)
        
    print("\n", file=logs)

    print("Materials in each cluster:", file=logs)
    for i in range(0, elbow):
        print("  Cluster #", i, ": ", file=logs)
        total = 0
        for k in range(0, len(cluster_info[i])):
            total += cluster_info[i][k]
        for k in range(0, len(cluster_info[i])):
            if not cluster_info[i][k] == 0:
                print("    Material ", data['Material'][k], ": ", cluster_info[i][k] / total * 100, "%", file=logs)
                
logs.close()