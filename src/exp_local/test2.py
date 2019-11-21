#!/usr/bin/env python
from segregation import Segregation
from termcolor import colored
import progressbar
import numpy as np
import time
import matplotlib.pyplot as plt


# Filename
filename = "data/exp_150_5_cluster_0_noise.npy"

# Setup
ROBOTS = 15
GROUPS = 3
WORLD	= 40.0
alpha = 1.0
noise = 0.00
dAA = np.array([5.0])
dAB		= 20.0
COMMRADIUS = 10
ITERATIONS = 5000
TRIALS = 5

# Structure data
data_metric = []
data_control = []

bar = progressbar.ProgressBar(maxval=ITERATIONS, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

print (colored("[Initializing the experiments] exp_150_5_cluster_radcomm_10", 'yellow'))


# Make 100 Experiments
for i in range(TRIALS):
	print (colored("[Experiment]", "yellow")),
	print (colored(i+1, 'green')),
	print (colored("of "+str(TRIALS), "red"))
	s = Segregation(ROBOTS=ROBOTS, GROUPS=GROUPS, WORLD=WORLD, alpha=alpha, noise=noise, dAA=dAA, dAB=dAB, seed=i, radius=COMMRADIUS, display_mode=True, which_metric='ncluster')
	bar.start()
	for j in range(ITERATIONS):
		bar.update(j)
		s.update()
		if j % 30 == 0:
			s.display()
		#data_control.append(s.a)
	plt.figure()
	data = np.array(s.metric_data)
	print(data)
	plt.plot(np.log(range(len(data[:]))), data[:], label="Clusters " + str(i) )
	plt.legend()
	plt.ylim(0, ROBOTS)
	plt.show()
	print ("\n")
	data_metric.append(s.metric_data)
bar.finish()

#plt.figure()
#for i in range(TRIALS):
#	data = np.array(data_metric[i])
#	plt.plot(np.log(range(len(data[:,0]))), data[:, 0], label="dAA " + str(i))
#	plt.plot(np.log(range(len(data[:,1]))), data[:, 1], '--', label="dAB " + str(i) )
#plt.legend()
#plt.show()
raw_input("Press the <ENTER> key to finish...")
print (colored("[Experiment has been completed!]", 'green'))
print (colored("[Saving Experiments]", 'grey'))
np.save(filename, data_metric)
print (colored("[Finish]", 'green'))

