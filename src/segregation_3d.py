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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
#####################################################################
# Constants (you may freely set these values).
#####################################################################
alpha	= 1.5 # scalar control gain
dAA		= np.array([40.0, 100.0, 300.0, 400.0])
#dAA 	= np.array([5.0])
dAB		= 200

ROBOTS	= 300
GROUPS	= 4
WORLD	= 400.0
dt 		= 0.005

#####################################################################
# Validation step
#####################################################################
if ROBOTS % GROUPS != 0:
	print "ROBOTS must be a multiple of GROUPS\n"
	quit()

if (len(dAA) > 1.0) and (len(dAA) != GROUPS):
	print "length(dAA) must be equal to GROUPS\n"
	quit()

if any(dAA <= math.sqrt(3.0)/9.0) and dAB <= math.sqrt(3.0)/9.0:
	print "Collective potential function is not strictly convex!\n"
	print "dAA and dAB must be larger than sqrt(3)/9\n"
	quit()

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

q = WORLD * (np.random.rand(ROBOTS,3) - 0.5) # position

v = np.zeros((ROBOTS, 3)) # velocity


#####################################################################
# Initialize all Plots
#####################################################################
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d', aspect='equal', autoscale_on=False, 
                     xlim=(-WORLD, WORLD), ylim=(-WORLD, WORLD), zlim=(-WORLD, WORLD))
ax.grid(color='gray', linestyle='-', linewidth=0.1)
cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 500/GROUPS, 500)]

handler = []
for i in range(GROUPS):
	particles, = ax.plot([], [], [], 'o', color=colors[i], ms=10)
	start = int(math.floor((i) * ROBOTS/GROUPS))
	stop = int(math.floor((i+1) * ROBOTS/GROUPS))
	particles.set_data(q[start:stop, 0], q[start:stop, 1])
	handler.append(particles)

np.seterr(divide='ignore')

#####################################################################
# Simulation
#####################################################################
def animate(i):
	global handler, GROUPS, q, v, dt
	last = time.clock()
	# Relative position among all pairs [q(j:2) - q(i:2)].
	xij = np.subtract(np.repeat(q[:,0], ROBOTS).reshape(ROBOTS,ROBOTS), q[:,0])
	yij = np.subtract(np.repeat(q[:,1], ROBOTS).reshape(ROBOTS,ROBOTS), q[:,1])
	zij = np.subtract(np.repeat(q[:,2], ROBOTS).reshape(ROBOTS,ROBOTS), q[:,2])

	# Relative velocity among all pairs [v(j:2) - v(i:2)].
	vxij = np.subtract(np.repeat(v[:,0], ROBOTS).reshape(ROBOTS,ROBOTS), v[:,0])
	vyij = np.subtract(np.repeat(v[:,1], ROBOTS).reshape(ROBOTS,ROBOTS), v[:,1])
	vzij = np.subtract(np.repeat(v[:,2], ROBOTS).reshape(ROBOTS,ROBOTS), v[:,2])

	# Relative distance among all pairs.
	dsqr = xij**2 + yij**2 + zij**2
	dist = np.sqrt(dsqr)

	# Control equation.
	dU = np.multiply(alpha, (dist - const + 1.0/dist - const/dsqr))
	ax = np.multiply(-dU, xij)/dist - vxij 
	ay = np.multiply(-dU, yij)/dist - vyij 
	az = np.multiply(-dU, zij)/dist - vzij 

	# a(i, :) -> acceleration input for robot i.
	a = np.array([np.nansum(ax, axis=1), np.nansum(ay, axis=1), np.nansum(az, axis=1)]).T

 	# simple taylor expansion.
	q = q + v * dt + a * (0.5 * dt**2)
	v = v + a * dt

	# Update data for drawing.
	for i in range(GROUPS):
		start = int(math.floor((i) * ROBOTS/GROUPS))
		stop = int(math.floor((i+1) * ROBOTS/GROUPS))
		handler[i].set_data(q[start:stop, 0], q[start:stop, 1])
		handler[i].set_3d_properties(q[start:stop, 2])
	return tuple(handler)

ani = animation.FuncAnimation(fig, animate, frames=25, interval=1, blit=False)
plt.hold(True)
plt.show()


