#!/usr/bin/env python
# Plot for Segregation into Cluster without noise
import matplotlib.pyplot as plt
import numpy as np
import math

fig = plt.figure(num=None, figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
filename = "exp_150_5_radial.npy"

print "Loading Experiments", filename
experiments = np.matrix(np.load(filename))
print "Check:", experiments.shape
print "Done"


setup = [('b', '150 robots (5 types) 0% Noise'), ('r', '150 robots (5 types) 1% Noise'), ('g', '150 robots (5 types) 5% Noise'), ('y', '150 robots (5 types) 10% Noise'), ('c', '150 robots (5 types) 20% Noise')]
for i in range(experiments.shape[0]):
	experiment = experiments[i, :].T[50:]
	x = range(1, len(experiment)+1)
	plt.semilogx(x, experiment, setup[i][0]+'-', label=setup[i][1], linewidth=1)

plt.ylim( 0 ) 
plt.xlim( 0 ) 

plt.title('Radial Segregation Control Convergence')
plt.xlabel('Iterations (log scale)')
plt.ylabel(r'$||U(q)||$')
plt.legend()
plt.savefig('radial_noise.png')
plt.show()
