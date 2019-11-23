#!/usr/bin/env python
from segregation import Segregation
from termcolor import colored
import progressbar
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats
import math
import scipy as sp
import random
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

# Setup
ROBOTS = 150
GROUPS = 5
random.seed(0)
DEAD_ROBOTS = random.sample(list(range(0,30)), k=0)
DEAD_ROBOTS += random.sample(list(range(30,60)), k=5)
DEAD_ROBOTS += random.sample(list(range(60,90)), k=10)
DEAD_ROBOTS += random.sample(list(range(90,120)), k=15)
DEAD_ROBOTS += random.sample(list(range(120,150)), k=20)
WORLD	= 40.0
alpha = 1.0
noise_sensor = 0.00
noise_actuation = 0.00
dAA = np.array([2.0])
dAB		= 5.0
COMM_RADIUS = 10000000000
ITERATIONS = 10000
TRIALS = 30

# Structure data
data_metric = []
data_control = []

# Filename
filename = "data5/exp_150_5_cluster_unbalanced.npy"

bar = progressbar.ProgressBar(maxval=ITERATIONS, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

experiments = []
experiments_datactrl = []
DEAD_ROBOTS_EXP = [[], DEAD_ROBOTS]

for DROBOTS in DEAD_ROBOTS_EXP:
	# Make 100 Experiments
	if not DROBOTS:
		print (colored("[Initializing the experiments] exp_150_5_cluster_balanced", 'yellow'))
		ROBOTS = 100
	else:
		print (colored("[Initializing the experiments] exp_150_5_cluster_unbalanced", 'yellow'))
		ROBOTS = 150
	data_metric = []
	data_control = []
	for i in range(TRIALS):
		print (colored("[Experiment]", "yellow")),
		print (colored(i+1, 'green')),
		print (colored("of "+str(TRIALS), "red"))
		s = Segregation(ROBOTS=ROBOTS, GROUPS=GROUPS, WORLD=WORLD, alpha=alpha,  noise_sensor=noise_sensor, noise_actuation=noise_actuation, dAA=dAA, dAB=dAB, seed=i, radius=COMM_RADIUS, display_mode=True, which_metric='cluster', DEAD_ROBOTS=DROBOTS)
		bar.start()
		dc = []
		for j in range(ITERATIONS):
			bar.update(j)
			s.update()
			a = sum(sum(abs(s.a)))
			dc.append(a)
			if j % 50 == 0:
				s.display()
		
		if not DROBOTS:
			s.screenshot("data5/log/png/log_exp_150_5_cluster_balanced_trial_"+str(i)+".png")
			s.screenshot("data5/log/eps/log_exp_150_5_cluster_balanced_trial_"+str(i)+".eps")
		else:
			s.screenshot("data5/log/png/log_exp_150_5_cluster_unbalanced_trial_"+str(i)+".png")
			s.screenshot("data5/log/eps/log_exp_150_5_cluster_unbalanced_trial_"+str(i)+".eps")
		
		plt.close('all')
		print ("\n")
		data_metric.append(s.metric_data)
		data_control.append(dc)
	experiments.append(data_metric)
	experiments_datactrl.append(data_control)
bar.finish()

plt.figure()

#Metric NCluster
filenames = [("exp_150_5_cluster_0_noise.npy", "100 robots (5 types) balanced", "b"), ("exp_150_5_cluster_1_noise.npy", "100 robots (5 types) unbalanced", "r"), ("exp_150_5_cluster_5_noise.npy", "150 robots (5 types) 5% Noise", "g"), ("exp_150_5_cluster_10_noise.npy", "150 robots (5 types) 10% Noise", "y"), ("exp_150_5_cluster_20_noise.npy", "150 robots (5 types) 20% Noise", "c"),]
setup = [('b', '100 robots (5 types) balanced'), ('r', '100 robots (5 types) unbalanced'), ('g', '150 robots (5 types) 5% Noise'), ('y', '150 robots (5 types) 10% Noise'), ('c', '150 robots (5 types) 20% Noise')]

for i in range(len(experiments)):
	datas = np.array(experiments[i])
	mean = []
	lower_bound = []
	upper_bound = []
	for j in range(len(datas[0])):
		data = np.array(datas[:, j])
		m, l, u = mean_confidence_interval(data=data)
		mean.append(m)
		upper_bound.append(u)
		lower_bound.append(l)
	mean = np.array(mean)/33.0
	upper_bound = np.array(upper_bound)/33.0
	lower_bound = np.array(lower_bound)/33.0
	x = range(1, len(mean)+1)
	plt.semilogx(x, mean, filenames[i][2]+'-', label=filenames[i][1], linewidth=1)
	plt.semilogx(x, upper_bound, filenames[i][2]+'--', linewidth=0.8)
	plt.semilogx(x, lower_bound, filenames[i][2]+'--', linewidth=0.8)

plt.ylim( 0 ) 
plt.xlim( 0 ) 

plt.title('Cluster Segregation Metric')
plt.xlabel('Iterations (log scale)')
plt.ylabel(r'$M_{clu}(q,\tau))$')
plt.grid(True)
plt.legend()

plt.savefig("data5/plot_exp_150_5_cluster_unbalanced_mclu.png", dpi=500)
plt.savefig("data5/plot_exp_150_5_cluster_unbalanced_mclu.eps", dpi=500)
#plt.show()

plt.figure()

for i in range(len(experiments_datactrl)):
	datas = np.array(experiments_datactrl[i])
	mean = []
	for j in range(len(datas[0])):
		data = np.array(datas[:, j])
		mean.append(data.mean())	
	x = range(1, len(mean)+1)
	plt.semilogx(x[50:], mean[50:], setup[i][0]+'-', label=setup[i][1], linewidth=1)
	
plt.ylim(0) 
plt.xlim(0) 

plt.title('Control Input')
plt.xlabel('Iterations (log scale)')
plt.ylabel(r'$\sum{||u_i||}$')
plt.grid(True)
plt.legend()

plt.savefig("data5/plot_exp_150_5_cluster_unbalanced_control.png", dpi=500)
plt.savefig("data5/plot_exp_150_5_cluster_unbalanced_control.eps", dpi=500)
plt.show()


raw_input("Press the <ENTER> key to finish...")
print (colored("[Experiment has been completed!]", 'green'))
print (colored("[Saving Experiments]", 'grey'))
np.save("data5/exp_150_5_cluster_actuation_noise_mclu.npy", experiments)
np.save("data5/exp_150_5_cluster_actuation_noise_actuation.npy", experiments_datactrl)
print (colored("[Finish]", 'green'))

