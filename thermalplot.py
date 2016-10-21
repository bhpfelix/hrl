# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:03:13 2016

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
import matplotlib.patches as mpatches

_plot_lim_y = 51
fig_num = 1

#data = pd.read_csv("ThermoData.csv")
data = pd.read_csv("data.csv")

colordict = {}
colors = ['blue','green','red','cyan','magenta','yellow','black','white']
for i, cate in enumerate(set(data['category'])):
    colordict[cate] = colors[i]

data["min_e"] = (data["min_k"] * data["min_d"] * data["min_c"]) ** 0.5
data['max_e'] = (data["max_k"] * data["max_d"] * data["max_c"]) ** 0.5

pos = np.arange(0.5,_plot_lim_y,0.5)
# Initialise plot
f = plt.figure(fig_num)
 
# ax = fig.add_axes([0.15,0.2,0.75,0.3]) #[left,bottom,width,height]
ax = f.add_subplot(111)

# Plot the data
for i in range(0, data['Material'].size):
    ax.barh((i*0.5)+0.5, data['max_e'][i] - data['min_e'][i], 
            left=data['min_e'][i], height=0.3, align='center', 
            color=colordict[data['category'][i]], alpha = 0.75, 
            label=data['category'][i])

h = [mpatches.Patch(color=colordict[key], label=key) for key in colordict.keys()]
ax.legend(handles=h, loc=2)


# Format the y-axis
 
locsy, labelsy = plt.yticks(pos,data['Material'])
plt.setp(labelsy, fontsize = 5)
 
# Format the x-axis
 
#ax.set_xlim(xmin = 0, xmax = 1.9e-4)
ax.set_xlim(xmin = 0.08, xmax = 115)
ax.set_ylim(ymin = -0.1, ymax = _plot_lim_y)
ax.grid(color = 'g', linestyle = ':')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

ax.set_xscale('log')
ax.set_title('Thermal Effusivity of Materials')
ax.set_xlabel('Thermal Effusivity')
ax.set_ylabel('Materials')

f.show()

#pos = np.arange(0.5,14.5,0.5)
#
#for cate in set(data['category']):
#    fig_num += 1
#    
#    # Initialise plot
#    g = plt.figure(fig_num)
#     
#    # ax = fig.add_axes([0.15,0.2,0.75,0.3]) #[left,bottom,width,height]
#    ax = g.add_subplot(111)
#     
#    # Plot the data
#    y_pos = 0
#    ylabels = []
#    new_f = data.loc[data['category'] == cate]
#    for index, row in new_f.iterrows():
#        ax.barh((y_pos*0.5)+0.5, row['max_e'] - row['min_e'], left=row['min_e'], height=0.3, align='center', color='blue', alpha = 0.75)
#        y_pos += 1
#        ylabels.append(row['Material'])
#     
#    # Format the y-axis
#     
#    y_poses = np.arange(0.5,y_pos/2.+1,0.5)
#    locsy, labelsy = plt.yticks(y_poses,ylabels)
#    plt.setp(labelsy, fontsize = 14)
#     
#    # Format the x-axis
#     
#    #ax.set_xlim(xmin = 0, xmax = 3e-6)
##    ax.set_ylim(ymin = -0.1, ymax = new_f.shape[0]/2. + 1)
#    ax.grid(color = 'g', linestyle = ':')
#    
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    
#    figManager = plt.get_current_fig_manager()
#    figManager.window.showMaximized()
#    
#    ax.set_title("Thermal Effusivity of %s" % cate)
#    ax.set_xlabel('Thermal Effusivity')
#    ax.set_ylabel('Materials')
#    
#    g.show()