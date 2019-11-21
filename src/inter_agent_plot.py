#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

from segregation import Segregation


ROBOTS = 150
GROUPS = 5
WORLD	= 10.0
dAA = np.linspace(3, WORLD*2, GROUPS) # radial
#dAA = np.array([2]) # cluster
dAB		= 5.0
seg = Segregation(ROBOTS=ROBOTS, GROUPS=GROUPS, dAA=dAA, dAB=dAB, WORLD=WORLD, noise=0.0)


fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-seg.WORLD, seg.WORLD), ylim=(-seg.WORLD, seg.WORLD))
ax.grid(color='gray', linestyle='-', linewidth=0.1)
cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 500/seg.GROUPS, 500)]

handler = []
for i in range(seg.GROUPS):
	particles, = ax.plot([], [], 'o', color=colors[i], ms=5)
	start = int(math.floor((i) * seg.ROBOTS/seg.GROUPS))
	stop = int(math.floor((i+1) * seg.ROBOTS/seg.GROUPS))
	particles.set_data(seg.q[start:stop, 0], seg.q[start:stop, 1])
	handler.append(particles)

np.seterr(divide='ignore')

anim_running = False
control = []
def onClick(event):
	global anim_running, anim
	print (len(seg.metric_data))
	if anim_running:
		anim.event_source.stop()
		anim_running = False
		print ("Paused")
	else:
		anim.event_source.start()
		anim_running = True
		print ("Running")

def animate(i):
	global handler
	for i in range(8):
		seg.update()
		control.append(sum(sum(abs(seg.a))))
	print (seg.feature())
	# Update data for drawing.
	for i in range(seg.GROUPS):
		start = int(math.floor((i) * seg.ROBOTS/seg.GROUPS))
		stop = int(math.floor((i+1) * seg.ROBOTS/seg.GROUPS))
		handler[i].set_data(seg.q[start:stop, 0], seg.q[start:stop, 1])

	#print "Velocity Sum:",  sum(seg.v)
	return tuple(handler)

anim = animation.FuncAnimation(fig, animate, frames=1, interval=1, blit=False)
fig.canvas.mpl_connect('button_press_event', onClick)
plt.show()

fig = plt.figure()
plt.plot(np.log(range(1, len(control)+1)), np.array(control), 'r-')
if seg.which_metric == 'cluster':
	plt.title('Cluster Segregation Metric')
	plt.xlabel('Iterations (log scale)')
	plt.ylabel(r'$M_{clu}(q,\tau)$')
elif seg.which_metric == 'radial':
	plt.title('Radial Segregation Metric')
	plt.xlabel('Iterations (log scale)')
	plt.ylabel(r'$M_{rad}(q,\tau)$')
plt.show()

#fig = plt.figure()
#plt.plot(np.log(range(1, len(seg.metric_data)+1)), np.array(seg.metric_data), 'r-')
#if seg.which_metric == 'cluster':
#	plt.title('Cluster Segregation Metric')
#	plt.xlabel('Iterations (log scale)')
#	plt.ylabel(r'$M_{clu}(q,\tau)$')
#elif seg.which_metric == 'radial':
#	plt.title('Radial Segregation Metric')
#	plt.xlabel('Iterations (log scale)')
#	plt.ylabel(r'$M_{rad}(q,\tau)$')
#plt.show()






