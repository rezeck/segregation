import numpy as np
import math

from RVO import RVO_update, reach, compute_V_des, reach

import matplotlib.pyplot as plt
import matplotlib.animation as animation
#####################################################################
# Constants (you may freely set these values).
#####################################################################

ROBOTS	= 20
GROUPS	= 20
WORLD	= 20.0
dt 		= 0.1

q = WORLD * (np.random.rand(ROBOTS,2) - 0.5) # position
v = np.zeros((ROBOTS, 2)) # velocity
v_max = np.ones((ROBOTS, 1)) # velocity
goal = WORLD * (np.random.rand(ROBOTS, 2) - 0.5) # position
goal = np.array([[(float(i) - ROBOTS/2.0)*2, 0.0] for i in range(ROBOTS)])


ws_model = dict()
#robot radius
ws_model['robot_radius'] = 0.8
#circular obstacles, format [x,y,rad]
# no obstacles
ws_model['circular_obstacles'] = []
#rectangular boundary, format [x,y,width/2,heigth/2]
ws_model['boundary'] = []
#####################################################################
# Initialize all Plots
#####################################################################
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-WORLD, WORLD), ylim=(-WORLD, WORLD))
ax.grid(color='gray', linestyle='-', linewidth=0.1)

cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 1, GROUPS)]

handler = []
for i in range(GROUPS):
	particles, = ax.plot([], [], 'o', color=colors[i], ms=10)
	start = int(math.floor((i) * ROBOTS/GROUPS))
	stop = int(math.floor((i+1) * ROBOTS/GROUPS))
	particles.set_data(q[start:stop, 0], q[start:stop, 1])
	handler.append(particles)

np.seterr(divide='ignore')

######################################################################
## Simulation
######################################################################
def animate(i):
	global handler, GROUPS, q, v, v_max, goal, dt

	v_des = compute_V_des(q, goal, v_max)
	v = RVO_update(q, v_des, v, ws_model)

	for i in xrange(len(q)):
		q[i][0] += v[i][0] * dt
		q[i][1] += v[i][1] * dt

	# Update data for drawing.
	for i in range(GROUPS):
		start = int(math.floor((i) * ROBOTS/GROUPS))
		stop = int(math.floor((i+1) * ROBOTS/GROUPS))
		handler[i].set_data(q[start:stop, 0], q[start:stop, 1])
	return tuple(handler)

ani = animation.FuncAnimation(fig, animate, frames=600, interval=1, blit=False)
plt.show()
#
#
#