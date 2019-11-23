#!/usr/bin/env python
from segregation import Segregation
from termcolor import colored
import progressbar
import numpy as np
import time
import matplotlib.pyplot as plt


# Setup
ROBOTS = 150
GROUPS = 5
WORLD	= 40.0
alpha = 1.0
noise_sensor = 0.00
noise_actuation = 0.00
dAA = np.array([2.0])
dAB		= 10.0
COMM_RADIUS = 5
ITERATIONS = 100000
TRIALS = 10

# Structure data
data_metric = []
data_control = []

# Filename
filename = "data1/exp_150_5_cluster_local.npy"

bar = progressbar.ProgressBar(maxval=ITERATIONS, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

experiments = []
rads = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
for rad in rads:
	# Make 100 Experiments
	print (colored("[Initializing the experiments] exp_150_5_cluster_local_"+str(rad), 'yellow'))
	data_metric = []
	for i in range(TRIALS):
		print (colored("[Experiment]", "yellow")),
		print (colored(i+1, 'green')),
		print (colored("of "+str(TRIALS), "red"))
		s = Segregation(ROBOTS=ROBOTS, GROUPS=GROUPS, WORLD=WORLD, alpha=alpha,  noise_sensor=noise_sensor, noise_actuation=noise_actuation, dAA=dAA, dAB=dAB, seed=i, radius=rad, display_mode=True, which_metric='ncluster')
		bar.start()
		for j in range(ITERATIONS):
			bar.update(j)
			if s.update():
				break
			if j % 50 == 0:
				s.display()
		s.get_score()
		s.screenshot("data1/log/png/log_exp_150_5_cluster_local_"+str(rad)+"_trial_"+str(i)+".png")
		s.screenshot("data1/log/eps/log_exp_150_5_cluster_local_"+str(rad)+"_trial_"+str(i)+".eps")
		plt.close('all')
		print ("\n")
		data_metric.append(s.metric_data)
	experiments.append(data_metric)
	
bar.finish()

plt.figure()
#Metric NCluster
mean_nclu = []
for i in range(len(experiments)):
	data = np.array(experiments[:][i])
	data = data[:][:,0]
	mean_nclu.append(data[:,0].mean())
	plt.errorbar(rads[i], data[:,0].mean(), yerr=data[:,0].std(), fmt='--o', color='black')
plt.plot(rads, mean_nclu, label="Number of clusters (5 types of robots)", color='black')
plt.axhline(y=5, linestyle='--', color='black')
plt.ylabel("Number of Clusters")
plt.xlim(0)
plt.ylim(0)
plt.grid(color='gray', linestyle='-', linewidth=0.1)
plt.xlabel("Sensor Radius (meters)")
plt.legend()
plt.savefig("data1/plot_exp_150_5_cluster_local_ncluster.png", dpi=500)
plt.savefig("data1/plot_exp_150_5_cluster_local_ncluster.eps", dpi=500)


#Metric Average Distance
plt.figure()
mean_daa = []
mean_dab = []
for i in range(len(experiments)):
	data = np.array(experiments[:][i])
	data = data[:][:,0]
	mean_daa.append(data[:,1].mean())
	mean_dab.append(data[:,2].mean())
	plt.errorbar(rads[i], data[:,1].mean(), yerr=data[:,1].std(),  fmt='--o', color='blue')
	plt.errorbar(rads[i], data[:,2].mean(), yerr=data[:,2].std(),  fmt='--o', color='red')
plt.plot(rads, mean_daa, label=r'Average $d_{AA}$', color='blue')
plt.plot(rads, mean_dab, label=r'Average $d_{AB}$', color='red')
plt.ylabel("Average Distance (meters)")
plt.xlim(0)
plt.ylim(0)
plt.grid(color='gray', linestyle='-', linewidth=0.1)
plt.xlabel("Sensor Radius (meters)")
plt.legend()
plt.savefig("data1/plot_exp_150_5_cluster_local_average.png", dpi=500)
plt.savefig("data1/plot_exp_150_5_cluster_local_average.eps", dpi=500)
plt.show()

raw_input("Press the <ENTER> key to finish...")
print (colored("[Experiment has been completed!]", 'green'))
print (colored("[Saving Experiments]", 'grey'))
np.save(filename, data_metric)
print (colored("[Finish]", 'green'))

