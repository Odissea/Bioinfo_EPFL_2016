import os

import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='jet') # ex. of cmap : "hsv", "jet", "spectral"
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color

###############################################################################
###############################################################################
###############################################################################

matplotlib.style.use('ggplot')
file_name = os.path.join(os.path.dirname(__file__), "plink.pca.evec")
nb_of_clusters = 6

alpha_points = 0.7
size_points = 20

background = (0.98, 0.98, 0.98)
fontsize_axis = 14
fontsize_titles = 20

show_centroids = False
adapt_scale = "custom" # can be None "force_same" or "zoom" or "custom"

###############################################################################
###############################################################################
###############################################################################
# do Kmeans
df = pd.read_csv(file_name, sep='\s+').drop("2.673", axis=1)

kmeans = KMeans(nb_of_clusters).fit(df.iloc[:, [0,1,2]])
KMeans(copy_x=True, init='kmeans++', max_iter=3000, n_clusters=5, n_init=10,
	n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
	verbose=0)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
centroids = list(zip(*centroids))
centroids_X = centroids[0]
centroids_Y = centroids[1]
if len(centroids) == 3:
	centroids_Z = centroids[2]
else:
	centroids_Z = [0 for i in range(len(centroids))]
###############################################################################
###############################################################################
###############################################################################
# calculate variance explained


###############################################################################
###############################################################################
###############################################################################
# Plot everywhere !
colormap = get_cmap(nb_of_clusters)
list_colors = labels.tolist()

for index, value in enumerate(list_colors):
	list_colors[index] = colormap(value)


x = df.iloc[:,0]
y = df.iloc[:,1]
z = df.iloc[:,2]


if (adapt_scale == "zoom") or (adapt_scale == "custom"):
	x_min, x_max = (-0.02, 0.13)
	y_min, y_max = (-0.05, 0.30)
	z_min, z_max = (-0.25, 0.20)

############# 3D

loc = plticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
data_points = ax.scatter(x, y, z, c=list_colors, alpha=alpha_points, s=size_points, edgecolor='k', linewidth='0.2')
if show_centroids:
	centroid = ax.scatter(centroids_X, centroids_Y, centroids_Z, c="r",marker="x",s=400,edgecolor='k', linewidth='3')
ax.set_xlabel('PC1', fontsize=fontsize_axis)
ax.set_ylabel('PC2', fontsize=fontsize_axis)
ax.set_zlabel('PC3', fontsize=fontsize_axis)
if adapt_scale == "force_same" or adapt_scale == "zoom":
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)
	ax.set_zlim(z_min, z_max)

ax.set_axis_bgcolor(background)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.zaxis.set_major_locator(loc)

############ 2D projections

if (adapt_scale == "force_same") or (adapt_scale == "custom"):
	x_min, x_max = (-0.25, 0.30)
	y_min, y_max = (-0.25, 0.30)
	z_min, z_max = (-0.25, 0.30)

fig2, ((ax1, ax2, ax3)) = plt.subplots(1,3)

data_points = ax1.scatter(x, y, c=list_colors, alpha=alpha_points, s=size_points, edgecolor='k', linewidth='0.2')
if show_centroids:
	centroid = ax1.scatter(centroids_X, centroids_Y, c="r",marker="x",s=400,edgecolor='k', linewidth='3')
ax1.set_xlabel('PC1', fontsize=fontsize_axis)
ax1.set_ylabel('PC2', fontsize=fontsize_axis)
if adapt_scale == "force_same" or adapt_scale == "zoom":
	ax1.set_xlim(x_min, x_max)
	ax1.set_ylim(y_min, y_max)
ax1.set_axis_bgcolor(background)
ax1.xaxis.set_major_locator(loc)
ax1.yaxis.set_major_locator(loc)

data_points = ax2.scatter(x, z, c=list_colors, alpha=alpha_points, s=size_points, edgecolor='k', linewidth='0.2')
if show_centroids:
	centroid = ax2.scatter(centroids_X, centroids_Z, c="r",marker="x",s=400,edgecolor='k', linewidth='3')
ax2.set_xlabel('PC1', fontsize=fontsize_axis)
ax2.set_ylabel('PC3', fontsize=fontsize_axis)
if adapt_scale == "force_same" or adapt_scale == "zoom":
	ax2.set_xlim(x_min, x_max)
	ax2.set_ylim(z_min, z_max)
ax2.set_axis_bgcolor(background)
ax2.xaxis.set_major_locator(loc)
ax2.yaxis.set_major_locator(loc)

data_points = ax3.scatter(y, z, c=list_colors, alpha=alpha_points, s=size_points, edgecolor='k', linewidth='0.2')
if show_centroids:
	centroid = ax3.scatter(centroids_X, centroids_Z, c="r",marker="x",s=400,edgecolor='k', linewidth='3')
ax3.set_xlabel('PC2', fontsize=fontsize_axis)
ax3.set_ylabel('PC3', fontsize=fontsize_axis)
if adapt_scale == "force_same" or adapt_scale == "zoom":
	ax3.set_xlim(y_min, y_max)
	ax3.set_ylim(z_min, z_max)
ax3.set_axis_bgcolor(background)
ax3.xaxis.set_major_locator(loc)
ax3.yaxis.set_major_locator(loc)

fig2.suptitle("Principal components after smartPCA", fontsize=fontsize_titles)

plt.show()
