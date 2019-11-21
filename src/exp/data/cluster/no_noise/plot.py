#!/usr/bin/env python
# Plot for Segregation into Cluster without noise
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import math

filename = "exp_150_5_cluster_no_noise.npy"
print "Loading Experiments"
experiments = np.matrix(np.load(filename))
print "Check:", experiments.shape
print "Done"

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


mean = []
lower_bound = []
upper_bound = []
max_it = 10000
for it in range(max_it):
	data = experiments[:,it]
	m, l, u = mean_confidence_interval(data=data)
	mean.append(m)
	upper_bound.append(u)
	lower_bound.append(l)


fig = plt.figure()
x = (range(1, max_it+1))
mean = np.array(mean)

upper_bound = np.array(upper_bound)
lower_bound = np.array(lower_bound)

plt.semilogx(x, mean, 'r-')
plt.semilogx(x, upper_bound, 'r--')
plt.semilogx(x, lower_bound, 'r--')
plt.ylim( 0 ) 
plt.xlim( 0 ) 

plt.title('Cluster Segregation Metric')
plt.xlabel('Iterations (log scale)')
plt.ylabel(r'$M_{clu}(q,\tau)$')

plt.show()








