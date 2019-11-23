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
filename = "data4/exp_150_5_cluster_actuation_noise.npy"

bar = progressbar.ProgressBar(maxval=ITERATIONS, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

experiments = []
experiments_datactrl = []
ACTUATION_noiseS = [0,1,5,10,20]
#ACTUATION_noiseS = [0,20]

#for ACTUATION_noise in ACTUATION_noiseS:
#	# Make 100 Experiments
#	print (colored("[Initializing the experiments] exp_150_5_cluster_actuation_noise_"+str(ACTUATION_noise), 'yellow'))
#	data_metric = []
#	data_control = []
#	for i in range(TRIALS):
#		print (colored("[Experiment]", "yellow")),
#		print (colored(i+1, 'green')),
#		print (colored("of "+str(TRIALS), "red"))
#		s = Segregation(ROBOTS=ROBOTS, GROUPS=GROUPS, WORLD=WORLD, alpha=alpha,  noise_sensor=noise_sensor, noise_actuation=ACTUATION_noise/100.0, dAA=dAA, dAB=dAB, seed=i, radius=COMM_RADIUS, display_mode=True, which_metric='cluster')
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
#		s.screenshot("data4/log/png/log_exp_150_5_cluster_actuation_noise_"+str(ACTUATION_noise)+"_trial_"+str(i)+".png")
#		s.screenshot("data4/log/eps/log_exp_150_5_cluster_actuation_noise_"+str(ACTUATION_noise)+"_trial_"+str(i)+".eps")
#		plt.close('all')
#		print ("\n")
#		data_metric.append(s.metric_data)
#		data_control.append(dc)
#	experiments.append(data_metric)
#	experiments_datactrl.append(data_control)
#bar.finish()

experiments = np.load("data4/exp_150_5_cluster_actuation_noise.npy")
experiments_datactrl = np.load("data4/exp_150_5_cluster_actuation_noise_actuation.npy")

#plt.figure()
fig, ax = plt.subplots(num=None, facecolor='w', edgecolor='k')

axins = zoomed_inset_axes(ax, 10000 , loc=3)	
axins2 = zoomed_inset_axes(ax, 20000 , loc=5)	


#Metric NCluster
filenames = [("exp_150_5_cluster_0_noise.npy", "150 robots (5 types) 0% noise", "b"), ("exp_150_5_cluster_1_noise.npy", "150 robots (5 types) 1% noise", "r"), ("exp_150_5_cluster_5_noise.npy", "150 robots (5 types) 5% noise", "g"), ("exp_150_5_cluster_10_noise.npy", "150 robots (5 types) 10% noise", "y"), ("exp_150_5_cluster_20_noise.npy", "150 robots (5 types) 20% noise", "c"),]
setup = [('b', '150 robots (5 types) 0% noise'), ('r', '150 robots (5 types) 1% noise'), ('g', '150 robots (5 types) 5% noise'), ('y', '150 robots (5 types) 10% noise'), ('c', '150 robots (5 types) 20% noise')]

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
	ax.semilogx(x, mean, filenames[i][2]+'-', label=filenames[i][1], linewidth=1)
	ax.semilogx(x, upper_bound, filenames[i][2]+'--', linewidth=0.8)
	ax.semilogx(x, lower_bound, filenames[i][2]+'--', linewidth=0.8)
	axins.semilogx(x, mean, filenames[i][2]+'-')
	axins2.semilogx(x, mean, filenames[i][2]+'-')

axins.set_xlim(8.88672, 8.88863)
axins.set_ylim(523.822, 523.842)
axins.set_yticks([])
axins.set_xticks([])

axins2.set_xlim(114.169,114.178)
axins2.set_ylim(112.803,112.812)
axins2.set_yticks([])
axins2.set_xticks([])

ax.set_ylim( 0 ) 
ax.set_xlim( 0 ) 
ax.set_ylim( 0 ) 
ax.set_xlim( 0 ) 

ax.set_title('Cluster Segregation Metric')
ax.set_xlabel('Iterations (log scale)')
ax.set_ylabel(r'$M_{clu}(q,\tau))$')
ax.grid(True)
ax.legend()

mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
mark_inset(ax, axins2, loc1=2, loc2=3, fc="none", ec="0.5")

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.savefig("data4/plot_exp_150_5_cluster_actuation_noise_mclu.png", dpi=500)
plt.savefig("data4/plot_exp_150_5_cluster_actuation_noise_mclu.eps", dpi=500)
#plt.show()




#plt.figure()
fig1, ax1 = plt.subplots(num=None, facecolor='w', edgecolor='k')

axins3 = zoomed_inset_axes(ax1, 14 , loc=3)	
axins4 = zoomed_inset_axes(ax1, 1500 , loc=5)	

for i in range(len(experiments_datactrl)):
	datas = np.array(experiments_datactrl[i])
	mean = []
	for j in range(len(datas[0])):
		data = np.array(datas[:, j])
		mean.append(data.mean())	
	x = range(1, len(mean)+1)
	ax1.semilogx(x[50:], mean[50:], setup[i][0]+'-', label=setup[i][1], linewidth=1)
	axins3.semilogx(x[50:], mean[50:], filenames[i][2]+'-')
	axins4.semilogx(x[50:], mean[50:], filenames[i][2]+'-')

axins3.set_xlim(87.3356, 91.7836)
axins3.set_ylim(1013.36, 1041.2)
axins3.set_yticks([])
axins3.set_xticks([])

axins4.set_xlim(695.158, 695.652)
axins4.set_ylim(6.00697, 6.19215)
axins4.set_yticks([])
axins4.set_xticks([])

ax1.set_ylim( 0 ) 
ax1.set_xlim( 0 ) 
ax1.set_ylim( 0 ) 
ax1.set_xlim( 0 ) 

ax1.set_title('Control Input')
ax1.set_xlabel('Iterations (log scale)')
ax1.set_ylabel(r'$\sum{||u_i||}$')
ax1.grid(True)
ax1.legend()

mark_inset(ax1, axins3, loc1=1, loc2=2, fc="none", ec="0.5")
mark_inset(ax1, axins4, loc1=3, loc2=4, fc="none", ec="0.5")

plt.xticks(visible=False)
plt.yticks(visible=False)

plt.savefig("data4/plot_exp_150_5_cluster_actuation_noise_control.png", dpi=500)
plt.savefig("data4/plot_exp_150_5_cluster_actuation_noise_control.eps", dpi=500)
plt.show()


raw_input("Press the <ENTER> key to finish...")
print (colored("[Experiment has been completed!]", 'green'))
print (colored("[Saving Experiments]", 'grey'))
np.save("data4/exp_150_5_cluster_actuation_noise.npy", experiments)
np.save("data4/exp_150_5_cluster_actuation_noise_actuation.npy", experiments_datactrl)
print (colored("[Finish]", 'green'))

