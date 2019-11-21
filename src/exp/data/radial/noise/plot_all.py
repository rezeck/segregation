#!/usr/bin/env python
# Plot for Segregation into radial without noise
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

fig, ax = plt.subplots(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
#fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
filenames = [("exp_150_5_radial_0_noise.npy", "150 robots (5 types) 0% Noise", "b"), ("exp_150_5_radial_1_noise.npy", "150 robots (5 types) 1% Noise", "r"), ("exp_150_5_radial_5_noise.npy", "150 robots (5 types) 5% Noise", "g"), ("exp_150_5_radial_10_noise.npy", "150 robots (5 types) 10% Noise", "y"), ("exp_150_5_radial_20_noise.npy", "150 robots (5 types) 20% Noise", "c"),]

axins = zoomed_inset_axes(ax, 3500 , loc=6)	
axins2 = zoomed_inset_axes(ax, 3500 , loc=5)	

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

	mean = np.array(mean)
	upper_bound = np.array(upper_bound)
	lower_bound = np.array(lower_bound)
	x = range(1, len(mean)+1)
	ax.semilogx(x, mean, filename[2]+'-', label=filename[1], linewidth=2)
	ax.semilogx(x, upper_bound, filename[2]+'--', linewidth=0.8)
	ax.semilogx(x, lower_bound, filename[2]+'--', linewidth=0.8)
	
	axins.semilogx(x, mean, filename[2]+'-')
	axins2.semilogx(x, mean, filename[2]+'-')

axins.set_xlim(20.8563, 20.8727)
axins.set_ylim(0.387213, 0.387253)
axins.set_yticks([])
axins.set_xticks([])

axins2.set_xlim(22.2179, 22.2355)
axins2.set_ylim(0.386448, 0.386487)
axins2.set_yticks([])
axins2.set_xticks([])

ax.set_ylim(0) 
ax.set_xlim(0) 
ax.set_title('Radial Segregation Metric')
ax.set_xlabel('Iterations (log scale)')
ax.set_ylabel(r'$M_{rad}(q,\tau)$')
ax.legend(loc='upper right')

mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
mark_inset(ax, axins2, loc1=2, loc2=1, fc="none", ec="0.5")

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.savefig('radial_noise_mod.png')
plt.show()








