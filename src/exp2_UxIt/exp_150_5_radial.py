#!/usr/bin/env python
from segregation import Segregation
from termcolor import colored
import progressbar
import numpy as np
import time

bar = progressbar.ProgressBar(maxval=10000, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

# Filename
filename = "data/exp_150_5_radial.npy"

# Setup
ROBOTS = 150
GROUPS = 5
WORLD	= 40.0
alpha = 1.0
dAA = np.linspace(5, GROUPS*5, GROUPS) # radial
dAB		= 7.5

# Structure data
datas = []
print colored("[Initializing the experiments] exp_150_5_radial_0_noise", 'yellow')

noises = [0.0, 0.01, 0.05, 0.1, 0.2]

# Make 100 Experiments
for noise in noises:
	print colored("[Experiment] With Noise", "grey"), colored(noise, 'blue')
	s = Segregation(ROBOTS=ROBOTS, GROUPS=GROUPS, WORLD=WORLD, alpha=alpha, noise=noise, dAA=dAA, dAB=dAB, seed=0)
	bar.start()
	data_control = []
	for j in range(10000):
		bar.update(j)
		s.update()
		a = sum(sum(abs(s.a)))
		data_control.append(a)
	datas.append(data_control)
	print "\n"

bar.finish()
print colored("[Experiment has been completed!]", 'green')
print colored("[Saving Experiments]", 'grey')
np.save(filename, datas)
print colored("[Finish]", 'green')

