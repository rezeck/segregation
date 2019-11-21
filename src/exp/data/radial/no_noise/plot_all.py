#!/usr/bin/env python
# Plot for Segregation into Cluster without noise
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import math

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
max_it = 10000
x = (range(1, max_it+1))
filenames = [("exp_150_5_radial_no_noise.npy", "150 robots (5 types)", "b"), ("exp_150_10_radial_no_noise.npy", "150 robots (10 types)", "r"), ("exp_150_15_radial_no_noise.npy", "150 robots (15 types)", "g")]

for filename in filenames:
	print "Loading Experiments", filename[0]
	experiments = np.matrix(np.load(filename[0]))
	print "Check:", experiments.shape
	print "Done"

	mean = []
	lower_bound = []
	upper_bound = []
	
	for it in range(max_it):
		data = experiments[:,it]
		m, l, u = mean_confidence_interval(data=data)
		mean.append(m)
		upper_bound.append(u)
		lower_bound.append(l)

	mean = np.array(mean)
	upper_bound = np.array(upper_bound)
	lower_bound = np.array(lower_bound)

	plt.semilogx(x, mean, filename[2]+'-', label=filename[1],linewidth=1)
	plt.semilogx(x, upper_bound, filename[2]+'--', linewidth=0.8)
	plt.semilogx(x, lower_bound, filename[2]+'--', linewidth=0.8)

plt.ylim( 0 ) 
plt.xlim( 0 ) 

plt.title('Radial Segregation Metric')
plt.xlabel('Iterations (log scale)')
plt.ylabel(r'$M_{rad}(q,\tau)$')
plt.legend()
plt.savefig('radial_no_noise.png')
plt.show()








