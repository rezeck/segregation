#!/usr/bin/env python
# Plot for Segregation into Cluster without noise
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import math
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

#fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
fig, ax = plt.subplots(num=None, facecolor='w', edgecolor='k')
filenames = [("exp_150_5_cluster_0_noise.npy", "150 robots (5 types) 0% Noise", "b"), ("exp_150_5_cluster_1_noise.npy", "150 robots (5 types) 1% Noise", "r"), ("exp_150_5_cluster_5_noise.npy", "150 robots (5 types) 5% Noise", "g"), ("exp_150_5_cluster_10_noise.npy", "150 robots (5 types) 10% Noise", "y"), ("exp_150_5_cluster_20_noise.npy", "150 robots (5 types) 20% Noise", "c"),]

axins = zoomed_inset_axes(ax, 4000 , loc=6)	
axins2 = zoomed_inset_axes(ax, 2500 , loc=5)	


for filename in filenames:
	print "Loading Experiments", filename[0]
	experiments = np.matrix(np.load(filename[0]))
	print "Check:", experiments.shape
	print "Done"

	mean = []
	lower_bound = []
	upper_bound = []
	
	for it in range(experiments.shape[1]):
		data = experiments[:,it]
		m, l, u = mean_confidence_interval(data=data)
		mean.append(m)
		upper_bound.append(u)
		lower_bound.append(l)

	mean = np.array(mean)/33
	upper_bound = np.array(upper_bound)/33
	lower_bound = np.array(lower_bound)/33
	x = range(1, len(mean)+1)
	ax.semilogx(x, mean, filename[2]+'-', label=filename[1], linewidth=1)
	ax.semilogx(x, upper_bound, filename[2]+'--', linewidth=0.8)
	ax.semilogx(x, lower_bound, filename[2]+'--', linewidth=0.8)

	axins.semilogx(x, mean, filename[2]+'-')
	axins2.semilogx(x, mean, filename[2]+'-')

axins.set_xlim(6.9810,6.98452)
axins.set_ylim(17798.1/33, 17799.4/33)
axins.set_yticks([])
axins.set_xticks([])

axins2.set_xlim(56.321, 56.3675)
axins2.set_ylim(8178.48/33, 8180.63/33)
axins2.set_yticks([])
axins2.set_xticks([])

ax.set_ylim( 0 ) 
ax.set_xlim( 0 ) 

plt.rc('font', family='serif', size=22)
ax.set_title('Cluster Segregation Metric', size = 28)
ax.set_xlabel('Iterations (log scale)', size = 24)
ax.set_ylabel(r'$M_{clu}(q,\tau))$', size = 24)
ax.grid(False)
ax.legend(loc=0, prop={'size': 24})

mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
mark_inset(ax, axins2, loc1=2, loc2=3, fc="none", ec="0.5")

plt.xticks(visible=False)
plt.yticks(visible=False)


#plt.savefig('cluster_noise_mod.eps')
plt.show()








