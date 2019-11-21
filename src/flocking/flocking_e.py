import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def phi_alpha(z):
	global r_alpha, d_alpha
	return rho_h(z/r_alpha) * phi(z-d_alpha)

def phi(z):
	global a,b,c
	return 1./2*((a+b)*(z+c)/sqrt(1+(z+c)**2) + (a-b) )

def rho_h(z):
	global h
	if (z>=0) and (z<h):
		return 1.;
	elif (z>=h) and (z<=1):
		return 1./2*(1+cos(pi*(z-h)/(1-h)))
	else:
		return 0.;

def sigma_norm(z):
	global epsilon
	return 1/epsilon*(sqrt(1+epsilon * (norm(z)**2) ) - 1)

def find_neighbor(x, iagent):
	global r, N
	neighbor=[];
	for inbor in range(N):
		pdiv = x[inbor,0:2] - x[iagent, 0:2]
		if (inbor != iagent) and norm(pdiv)<r:
			neighbor = hstack((neighbor, inbor))
	return neighbor

#####################################################################
# Constants (you may freely set these values).
#####################################################################
alpha	= 1.5 # scalar control gain
#dAA		= np.array([3.0, 5.0, 6.0, 12.0])
dAA 	= np.array([5.0])
dAB		= 20.0

ROBOTS	= 12
GROUPS	= 4
WORLD	= 20.0
dt 		= 0.02


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

# vectorization of dAA and dAB.
if len(dAA) == 1.0:
	const = np.multiply(dAA, AA) + np.multiply(dAB, AB)
else:
	const = np.kron(np.diag(dAA), np.ones((ROBOTS/GROUPS, ROBOTS/GROUPS))) + np.multiply(dAB, AB)

q = WORLD * (np.random.rand(ROBOTS,2) - 0.5) # position

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
	particles, = ax.plot([], [], 'o', color=colors[i], ms=10)
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

	# Relative position among all pairs [q(j:2) - q(i:2)].
	xij = np.subtract(np.repeat(q[:,0], ROBOTS).reshape(ROBOTS,ROBOTS), q[:,0])
	yij = np.subtract(np.repeat(q[:,1], ROBOTS).reshape(ROBOTS,ROBOTS), q[:,1])

	#xij += np.random.normal(0, 1, ROBOTS)
	#yij += np.random.normal(0, 1, ROBOTS)

	# Relative velocity among all pairs [v(j:2) - v(i:2)].
	vxij = np.subtract(np.repeat(v[:,0], ROBOTS).reshape(ROBOTS,ROBOTS), v[:,0])
	vyij = np.subtract(np.repeat(v[:,1], ROBOTS).reshape(ROBOTS,ROBOTS), v[:,1])

	

	# Relative distance among all pairs.
	dsqr = xij**2 + yij**2
	dist = np.sqrt(dsqr)

	# Control equation.
	dV = np.multiply(alpha, (dist - const + 1.0/dist - const/dsqr))
	ax = np.multiply(-dV, xij)/dist - vxij
	ay = np.multiply(-dV, yij)/dist - vyij

	# a(i, :) -> acceleration input for robot i.
	a = np.array([np.nansum(ax, axis=1), np.nansum(ay, axis=1)]).T

 	# simple taylor expansion.
	q = q + v*dt + a*(0.5*dt**2) + np.random.normal(0, 1, (ROBOTS, 2))*0.05
	v = v + a*dt + np.random.normal(0, 1, (ROBOTS, 2))*0.0

	# Update data for drawing.
	for i in range(GROUPS):
		start = int(math.floor((i) * ROBOTS/GROUPS))
		stop = int(math.floor((i+1) * ROBOTS/GROUPS))
		handler[i].set_data(q[start:stop, 0], q[start:stop, 1])
	return tuple(handler)

ani = animation.FuncAnimation(fig, animate, frames=600, interval=1, blit=False)
plt.show()


