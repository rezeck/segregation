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
filename = "data3/exp_150_5_cluster_sensor_noise.npy"

bar = progressbar.ProgressBar(maxval=ITERATIONS, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

experiments = []
experiments_datactrl = []
SENSOR_NOISES = [0,1,5,10,20]
#SENSOR_NOISES = [0,20]

#for SENSOR_NOISE in SENSOR_NOISES:
#	# Make 100 Experiments
#	print (colored("[Initializing the experiments] exp_150_5_cluster_sensor_noise_"+str(SENSOR_NOISE), 'yellow'))
#	data_metric = []
#	data_control = []
#	for i in range(TRIALS):
#		print (colored("[Experiment]", "yellow")),
#		print (colored(i+1, 'green')),
#		print (colored("of "+str(TRIALS), "red"))
#		s = Segregation(ROBOTS=ROBOTS, GROUPS=GROUPS, WORLD=WORLD, alpha=alpha,  noise_sensor=SENSOR_NOISE/100.0, noise_actuation=noise_actuation, dAA=dAA, dAB=dAB, seed=i, radius=COMM_RADIUS, display_mode=True, which_metric='cluster')
#		bar.start()
#		dc = []
#		for j in range(ITERATIONS):
#			bar.update(j)
#			s.update()
#			a = sum(sum(abs(s.a)))
#			dc.append(a)
#			if j % 50 == 0:
#				s.display()
#		
#		s.screenshot("data3/log/png/log_exp_150_5_cluster_sensor_noise_"+str(SENSOR_NOISE)+"_trial_"+str(i)+".png")
#		s.screenshot("data3/log/eps/log_exp_150_5_cluster_sensor_noise_"+str(SENSOR_NOISE)+"_trial_"+str(i)+".eps")
#		plt.close('all')
#		print ("\n")
#		data_metric.append(s.metric_data)
#		data_control.append(dc)
#	experiments.append(data_metric)
#	experiments_datactrl.append(data_control)
#bar.finish()

experiments = np.load("data3/exp_150_5_cluster_sensor_noise.npy")
experiments_datactrl = np.load("data3/exp_150_5_cluster_sensor_noise_actuation.npy")

#plt.figure()
fig, ax = plt.subplots(num=None, facecolor='w', edgecolor='k')

axins = zoomed_inset_axes(ax, 10000 , loc=6)	
axins2 = zoomed_inset_axes(ax, 900 , loc=5)	


#Metric NCluster
filenames = [("exp_150_5_cluster_0_noise.npy", "150 robots (5 types) 0% noise", "b+"), ("exp_150_5_cluster_1_noise.npy", "150 robots (5 types) 1% noise", "r*"), ("exp_150_5_cluster_5_noise.npy", "150 robots (5 types) 5% noise", "g>"), ("exp_150_5_cluster_10_noise.npy", "150 robots (5 types) 10% noise", "yx"), ("exp_150_5_cluster_20_noise.npy", "150 robots (5 types) 20% noise", "c^"),]
setup = [('b+', '150 robots (5 types) 0% noise'), ('r*', '150 robots (5 types) 1% noise'), ('g>', '150 robots (5 types) 5% noise'), ('yx', '150 robots (5 types) 10% noise'), ('c^', '150 robots (5 types) 20% noise')]

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
	ax.semilogx(x, mean, filenames[i][2]+'-', label=filenames[i][1], linewidth=1, markersize=1, markevery=50)
	ax.semilogx(x, upper_bound, filenames[i][2][:-1]+'--', linewidth=0.8)
	ax.semilogx(x, lower_bound, filenames[i][2][:-1]+'--', linewidth=0.8)
	axins.semilogx(x, mean, filenames[i][2]+'-')
	axins2.semilogx(x, mean, filenames[i][2]+'-')

axins.set_xlim(4.99883, 5.00013)
axins.set_ylim(558.751, 558.774)
axins.set_yticks([])
axins.set_xticks([])

axins2.set_xlim(812.652, 814.993)
axins2.set_ylim(17.0987, 17.304)
axins2.set_yticks([])
axins2.set_xticks([])

ax.set_ylim( 0 ) 
ax.set_xlim( 0 ) 
ax.set_ylim( 0 ) 
ax.set_xlim( 0 ) 

ax.set_title('Cluster Segregation Metric', fontsize=16)
ax.set_xlabel('Iterations (log scale)',fontsize=16)
ax.set_ylabel(r'$M_{clu}(q,\tau))$', fontsize=16)
ax.grid(True)
ax.grid(color='gray', linestyle='-', linewidth=0.5)

ax.legend(prop={'size': 14})
legend = ax.legend(frameon=True)
for legend_handle in legend.legendHandles:
    legend_handle._legmarker.set_markersize(8)
ax.tick_params(axis='both', which='major', labelsize=12)

# ax.xticks(fontsize=12)
# ax.yticks(fontsize=12)

mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
mark_inset(ax, axins2, loc1=4, loc2=3, fc="none", ec="0.5")

plt.xticks(visible=True)
plt.yticks(visible=False)
plt.savefig("data3/plot_exp_150_5_cluster_sensor_noise_mclu.png", dpi=500)
plt.savefig("data3/plot_exp_150_5_cluster_sensor_noise_mclu.eps", dpi=500)
plt.show()

plt.figure()
for i in range(len(experiments_datactrl)):
	datas = np.array(experiments_datactrl[i])
	mean = []
	for j in range(len(datas[0])):
		data = np.array(datas[:, j])
		mean.append(data.mean())	
	x = range(1, len(mean)+1)
	plt.semilogx(x[50:], mean[50:], setup[i][0]+'-', label=setup[i][1], linewidth=1, markevery=100, markersize=6)

plt.ylim([0,2000]) 
plt.xlim(0) 
plt.title('Control Input', fontsize=16)
plt.xlabel('Iterations (log scale)', fontsize=16)
plt.ylabel(r'$\sum{||u_i||}$', fontsize=16)
plt.grid(True)
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.legend()
ax.legend(markerscale=60)
ax.legend(prop={'size': 12})
ax.tick_params(axis='both', which='major', labelsize=12)
plt.savefig("data3/plot_exp_150_5_cluster_sensor_noise_control.png", dpi=500)
plt.savefig("data3/plot_exp_150_5_cluster_sensor_noise_control.eps", dpi=500)
plt.show()


input("Press the <ENTER> key to finish...")
print (colored("[Experiment has been completed!]", 'green'))
print (colored("[Saving Experiments]", 'grey'))
np.save("data3/exp_150_5_cluster_sensor_noise.npy", experiments)
np.save("data3/exp_150_5_cluster_sensor_noise_actuation.npy", experiments_datactrl)
print (colored("[Finish]", 'green'))

