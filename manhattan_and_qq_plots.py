#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import random
import sys

import pylab 
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def get_cmap(N):
	'''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
	RGB color.'''
	color_norm  = colors.Normalize(vmin=0, vmax=N-1)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap="Set1") # ex. of cmap : "hsv", "jet", "spectral", "prism"
	def map_index_to_rgb_color(index):
		return scalar_map.to_rgba(index)

	return map_index_to_rgb_color

def print_progress(
		iteration,
		total,
		prefix='',
		suffix='',
		decimals=1,
		bar_length=100):
	"""Function to print a progress bar.

	This function simply prints a progress bar corresponding to the ratio
	between the number of iterations and the expected total number of iterations.
	The percentage is also shown in a numeric form
	This function can also display some custom information in two fields (prefix
	and suffix).

	Args:
		iteration (int): the number of iterations already past
		total (int): the expected total number of iterations
		prefix (str): field to display custom information (default is '')
		suffix (str): field to display custom information (default is '')
		decimals (int): number of decimals to show
		bar_length (int)

	Returns:
		None

	Raises:
		None
	"""
	format_str = "{0:." + str(decimals) + "f}"
	percents = format_str.format(100 * (iteration / float(total)))
	filled_length = int(round(bar_length * iteration / float(total)))
	if iteration / float(total) < 0.333:
		filler = "░"
	elif iteration / float(total) < 0.666:
		filler = "▒"
	elif iteration / float(total) < 1:
		filler = "▓"
	else:
		filler = "█"
	bar = filler * filled_length + '-' * (bar_length - filled_length)
	sys.stdout.write(
		'\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()


path_to_source = os.path.dirname(__file__)
source_name = "plink.assoc.linear.ADD"
source_name_light = "plink.assoc.linear.ADD.light"

if not os.path.isfile(os.path.join(path_to_source, source_name_light)):
	data = pd.read_csv(os.path.join(path_to_source, source_name), sep="\s+", usecols=["CHR", "BP", "TEST", "P"])
	data = data.loc[data.TEST == "ADD"]
	data = data[["CHR", "BP", "P"]]
	data.to_csv(os.path.join(path_to_source, source_name_light), sep=" ", index=False)
else:
	data = pd.read_csv(os.path.join(path_to_source, source_name_light), sep="\s+")

data["minus_log10_P"] = - np.log10(data["P"])
data["Normalised_BP"] = 0
list_chromosomes = data["CHR"].unique().tolist()
print("Calculate the x position for the different points ...")
for chromosome in list_chromosomes:
	data.loc[data.CHR == chromosome, "Normalised_BP"] = data.loc[data.CHR == chromosome, "BP"] / (1.2*data.loc[data.CHR == chromosome, "BP"].max())
	print_progress(int(chromosome), len(data["CHR"].unique()))

pd.to_numeric(data["Normalised_BP"], errors='coerce')
pd.to_numeric(data["BP"], errors='coerce')
pd.to_numeric(data["CHR"], errors='coerce')
data["X_value"] = 0
data["X_value"] += data["Normalised_BP"] + data["CHR"] - 0.5

#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
alpha_points = 1
size_points = 20
marker_points = "D"

background = (0.95, 0.95, 0.95)
fontsize_axis = 14
fontsize_titles = 20

filter_high_p_values = False # don't show points with p > threshold
threshold = 0.005
matplotlib.style.use('ggplot')



#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

if filter_high_p_values:
	data = data.loc[data.P <= threshold]
if True:
	for i in list_chromosomes:
		to_shift = random.sample(list_chromosomes, 2)

		data.loc[data.CHR == to_shift[0], "CHR"] = 200
		data.loc[data.CHR == to_shift[1], "CHR"] = to_shift[0]
		data.loc[data.CHR == 200, "CHR"] = to_shift[1]

print(data["CHR"].unique().tolist())
colormap = get_cmap(len(list_chromosomes))
#colormap = colors.ListedColormap(np.random.rand ( 256,3))

print("Adapting the color map ...")
list_colors = data["CHR"].apply(colormap).tolist()
total_points = len(list_colors)
"""
print("Adapting the color map ...")
for index, value in enumerate(list_colors):
	list_colors[index] = colormap(value)
	print_progress(index, total_points-1, decimals=3)"""

print("Plotting figure ...")
loc = plticker.MultipleLocator(base=1)
fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_xlim(0, 23)
ax.set_ylim(0, data["minus_log10_P"].max() + 1)
x = data["X_value"]
y = data["minus_log10_P"]
data_points = ax.scatter(x, y, c=list_colors, marker=marker_points, alpha=alpha_points, s=size_points, linewidth=0)
ax.set_xlabel('Chromosome', fontsize=fontsize_axis)
ax.set_ylabel('-log10(p)', fontsize=fontsize_axis)
ax.set_axis_bgcolor(background)
#ax.xaxis.set_major_locator(loc)
ax.grid(b=False)

plt.setp(ax, xticks=[i for i in range(1,23)], xticklabels=[i for i in range(1,23)],
        yticks=[1, 2, 3])

significance = 5 * math.pow(10,-8)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.axhline(y=-math.log10(significance), xmin=0, xmax=1, linewidth=2, color = 'k')

fig.savefig("manhattan.png")
plt.show()
