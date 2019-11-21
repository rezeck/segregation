#!/usr/bin/env python
from segregation import Segregation
from termcolor import colored
import progressbar
import numpy as np
import time

bar = progressbar.ProgressBar(maxval=10000, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

# Filename
filename = "data/exp_150_5_radial_10_noise.npy"

# Setup
ROBOTS = 150
GROUPS = 5
WORLD	= 40.0
alpha = 1.0
noise = 0.10
dAA = np.linspace(5, GROUPS*5, GROUPS) # radial
dAB		= 7.5

# Structure data
data_metric = []
data_control = []

print colored("[Initializing the experiments] exp_150_5_radial_10_noise", 'yellow')


# Make 100 Experiments
for i in range(100):
	print colored("[Experiment]", "grey"), colored(i+1, 'blue'), colored("of 100", "grey")
	s = Segregation(ROBOTS=ROBOTS, GROUPS=GROUPS, WORLD=WORLD, alpha=alpha, noise=noise, dAA=dAA, dAB=dAB, seed=i)
	bar.start()
	for j in range(10000):
		bar.update(j)
		s.update()
		#data_control.append(s.a)
	print "\n"
	data_metric.append(s.metric_data)
bar.finish()
print colored("[Experiment has been completed!]", 'green')
print colored("[Saving Experiments]", 'grey')
np.save(filename, data_metric)
print colored("[Finish]", 'green')

