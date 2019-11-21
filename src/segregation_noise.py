#!/usr/bin/env python
#    _________                                         __  .__                  #
#   /   _____/ ____   ___________   ____   _________ _/  |_|__| ____   ____     #
#   \_____  \_/ __ \ / ___\_  __ \_/ __ \ / ___\__  \\   __\  |/  _ \ /    \    #
#   /        \  ___// /_/  >  | \/\  ___// /_/  > __ \|  | |  (  <_> )   |  \   #
#  /_______  /\___  >___  /|__|    \___  >___  (____  /__| |__|\____/|___|  /   #
#          \/     \/_____/             \/_____/     \/                    \/    #
#  																				#
#################################################################################
# Based on Matlab code of Vinicius Graciano Santos 
# Author: Paulo Rezeck
#################################################################################
# Cluster Segregation: dAA < dAB
#  Example: dAA = 3.0 and dAB = 5.0
# 
#  Cluster Aggregation: dAA > dAB
#  Example: dAA = 5.0 and dAB = 3.0
# 
#  Radial Segregation.: dAA(1) < dAB < dAA(2) < ... < dAA(n)
#  Example: dAA = [3.0, 5.0] and dAB = 4.0
# 
#  For convexity, all values must be larger than sqrt(3)/9
#################################################################################
from __future__ import division
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from metric import ClusterMetric, RadialMetric
#####################################################################
# Constants (you may freely set these values).
#####################################################################
alpha	= 1.5 # scalar control gain

ROBOTS	= 2
GROUPS	= 2
WORLD	= 40.0
dt 		= 0.01

dAA = np.linspace(10, WORLD*2, GROUPS) # radial
#dAA = np.array([10]) # cluster
print ("dAA:" + str(dAA))
dAB		= 15.0

e_rate = 0.5

#####################################################################
# Validation step
#####################################################################
if ROBOTS % GROUPS != 0:
	print ("ROBOTS must be a multiple of GROUPS\n")
	quit()

if (len(dAA) > 1.0) and (len(dAA) != GROUPS):
	print ("length(dAA) must be equal to GROUPS\n")
	quit()

if any(dAA <= math.sqrt(3.0)/9.0) and dAB <= math.sqrt(3.0)/9.0:
	print ("Collective potential function is not strictly convex!\n")
	print ("dAA and dAB must be larger than sqrt(3)/9\n")
	quit()

if (len(dAA) > 1.0):
	print ("Robots will segregate to a radial configuration")
	which_metric = 'radial'
else:	
	print ("Robots will segregate to a cluster configuration")
	which_metric = 'cluster'
#####################################################################
# Initialization
#####################################################################
# AA[i,j] == 1 if i and j belong to the same team,  0 otherwise.
# AB[i,j] == 1 if i and j belong to distinct teams, 0 otherwise.
x = np.array(range(1, ROBOTS+1))
i, j = np.meshgrid(x, x)

gpr = float(GROUPS)/float(ROBOTS)

AA = (np.floor(gpr * (i-1.0)) == np.floor(gpr * (j-1.0))) * 1.0
AB = (np.floor(gpr * (i-1.0)) != np.floor(gpr * (j-1.0))) * 1.0

# Vectorization of dAA and dAB constraint
if len(dAA) == 1.0:
	const = np.multiply(dAA, AA) + np.multiply(dAB, AB)
else:
	const = np.kron(np.diag(dAA), np.ones((ROBOTS/GROUPS, ROBOTS/GROUPS))) + np.multiply(dAB, AB)

#print const
np.random.seed(2)
q = WORLD * (np.random.rand(ROBOTS,2) - 0.5) # position
np.random.seed(None)

v = np.zeros((ROBOTS, 2)) # velocity


#####################################################################
# Initialize all Plots
#####################################################################
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-WORLD, WORLD), ylim=(-WORLD, WORLD))
ax.grid(color='gray', linestyle='-', linewidth=0.1)
cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 500/GROUPS, 500)]

handler = []
for i in range(GROUPS):
	particles, = ax.plot([], [], 'o', color=colors[i], ms=5)
	start = int(math.floor((i) * ROBOTS/GROUPS))
	stop = int(math.floor((i+1) * ROBOTS/GROUPS))
	particles.set_data(q[start:stop, 0], q[start:stop, 1])
	handler.append(particles)

np.seterr(divide='ignore')


metric_data = []
# Choice which metric should the robots use
if which_metric == 'cluster':
	metric = ClusterMetric(GROUPS, ROBOTS)		
elif which_metric == 'radial':
	metric = RadialMetric(GROUPS, ROBOTS, const)

#####################################################################
# Simulation
#####################################################################
def animate(i):
	global handler, GROUPS, q, v, dt, metric_data
	last = time.clock()

	# Relative position among all pairs [q(j:2) - q(i:2)].
	xij = np.subtract(np.repeat(q[:,0], ROBOTS).reshape(ROBOTS,ROBOTS), q[:,0])
	yij = np.subtract(np.repeat(q[:,1], ROBOTS).reshape(ROBOTS,ROBOTS), q[:,1])

	# Relative velocity among all pairs [v(j:2) - v(i:2)].
	vxij = np.subtract(np.repeat(v[:,0], ROBOTS).reshape(ROBOTS,ROBOTS), v[:,0])
	vyij = np.subtract(np.repeat(v[:,1], ROBOTS).reshape(ROBOTS,ROBOTS), v[:,1])

	# Relative distance among all pairs.
	dsqr = xij**2 + yij**2
	dist = np.sqrt(dsqr)

	# Control equation.
	dU = np.multiply(alpha, (dist - const + 1.0/dist - const/dsqr))

	ax = np.multiply(-dU, xij)/dist - vxij # damping
	ay = np.multiply(-dU, yij)/dist - vyij # damping

	# a(i, :) -> acceleration input for robot i.
	a = np.array([np.nansum(ax, axis=1), np.nansum(ay, axis=1)]).T

	# Add noise to position
	a = np.random.normal(a, e_rate*abs(a))

 	# simple taylor expansion.
	q = q + v*dt + a*(0.5*dt**2)
	v = v + a*dt

	# Update data for drawing.
	for i in range(GROUPS):
		start = int(math.floor((i) * ROBOTS/GROUPS))
		stop = int(math.floor((i+1) * ROBOTS/GROUPS))
		handler[i].set_data(q[start:stop, 0], q[start:stop, 1])

	m = metric.compute(q)
	metric_data.append(m)
	#print "Time:", time.clock() - last
	print ("Velocity Sum:",  sum(v))
	return tuple(handler)

anim_running = False
def onClick(event):
	global anim_running, anim
	print (len(metric_data))
	if anim_running:
		anim.event_source.stop()
		anim_running = False
		print ("Paused")
	else:
		anim.event_source.start()
		anim_running = True
		print ("Running")

anim = animation.FuncAnimation(fig, animate, frames=10, interval=1, blit=False)
fig.canvas.mpl_connect('button_press_event', onClick)
plt.show()

fig = plt.figure()
plt.plot(np.log(range(1, len(metric_data)+1)), np.array(metric_data), 'r-')
if which_metric == 'cluster':
	plt.title('Cluster Segregation Metric')
	plt.xlabel('Iterations (log scale)')
	plt.ylabel('Mclu(q,t)')
elif which_metric == 'radial':
	plt.title('Radial Segregation Metric')
	plt.xlabel('Iterations (log scale)')
	plt.ylabel('Mrad(q,t)')
plt.show()





