#!/usr/bin/env python
#    _________                                         __  .__                  #
#   /   _____/ ____   ___________   ____   _________ _/  |_|__| ____   ____     #
#   \_____  \_/ __ \ / ___\_  __ \_/ __ \ / ___\__  \\   __\  |/  _ \ /    \    #
#   /        \  ___// /_/  >  | \/\  ___// /_/  > __ \|  | |  (  <_> )   |  \   #
#  /_______  /\___  >___  /|__|    \___  >___  (____  /__| |__|\____/|___|  /   #
#          \/     \/_____/             \/_____/     \/                    \/    #
#  																				#
#################################################################################
# Adapted from Matlab code of Vinicius Graciano Santos 
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
import numpy as np
import math
from metric import *
np.seterr(divide='ignore', invalid='ignore')
class Segregation(object):
	"""docstring for Segregation"""
	def __init__(self, alpha=1.5, ROBOTS=15, GROUPS=3, WORLD=40, dt=0.01, dAA=[7], dAB=20, noise=0.05):
		self.alpha = alpha
		self.ROBOTS = ROBOTS
		self.GROUPS = GROUPS
		self.WORLD = WORLD
		self.dt = dt
		self.dAA = np.array(dAA)
		self.dAB = dAB
		self.noise = noise
		# validation step
		self.validation()
		# Initialization
		self.setup()
		
	def validation(self):
		if self.ROBOTS % self.GROUPS != 0:
			print "ROBOTS must be a multiple of GROUPS\n"
			quit()

		if (len(self.dAA) > 1.0) and (len(self.dAA) != self.GROUPS):
			print "length(dAA) must be equal to GROUPS\n"
			quit()

		if any(self.dAA <= math.sqrt(3.0)/9.0) and self.dAB <= math.sqrt(3.0)/9.0:
			print "Collective potential function is not strictly convex!\n"
			print "dAA and dAB must be larger than sqrt(3)/9\n"
			quit()

		if (len(self.dAA) > 1.0):
			#print "Robots will segregate to a radial configuration"
			self.which_metric = 'radial'
		else:	
			#print "Robots will segregate to a cluster configuration"
			self.which_metric = 'cluster'

	def setup(self):
		x = np.array(range(1, self.ROBOTS+1))
		i, j = np.meshgrid(x, x)

		gpr = float(self.GROUPS)/float(self.ROBOTS)
		
		AA = (np.floor(gpr*(i-1.0)) == np.floor(gpr*(j-1.0)))*1.0
		AB = (np.floor(gpr*(i-1.0)) != np.floor(gpr*(j-1.0)))*1.0
		
		# Vectorization of dAA and dAB.
		if len(self.dAA) == 1.0:
			self.const = np.multiply(self.dAA, AA) + np.multiply(self.dAB, AB)
		else:
			self.const = np.kron(np.diag(self.dAA), np.ones((self.ROBOTS/self.GROUPS, self.ROBOTS/self.GROUPS))) + np.multiply(self.dAB, AB)
		
		#np.random.seed(2)
		self.q = self.WORLD * (np.random.rand(self.ROBOTS, 2) - 0.5) # position
		#np.random.seed(None)

		self.v = np.zeros((self.ROBOTS, 2)) # velocity

		self.metric_data = []
		# Choice which metric should the robots use
		if self.which_metric == 'cluster':
			self.metric = ClusterMetric(self.GROUPS, self.ROBOTS)		
		elif self.which_metric == 'radial':
			self.metric = RadialMetric(self.GROUPS, self.ROBOTS, self.const)

	def update(self):
		last = time.clock()

		# Relative position among all pairs [q(j:2) - q(i:2)].
		xij = np.subtract(np.repeat(self.q[:,0], self.ROBOTS).reshape(self.ROBOTS, self.ROBOTS), self.q[:,0])
		yij = np.subtract(np.repeat(self.q[:,1], self.ROBOTS).reshape(self.ROBOTS, self.ROBOTS), self.q[:,1])
	
		# Relative velocity among all pairs [v(j:2) - v(i:2)].
		vxij = np.subtract(np.repeat(self.v[:,0], self.ROBOTS).reshape(self.ROBOTS, self.ROBOTS), self.v[:,0])
		vyij = np.subtract(np.repeat(self.v[:,1], self.ROBOTS).reshape(self.ROBOTS, self.ROBOTS), self.v[:,1])
	
		# Relative distance among all pairs.
		dsqr = xij**2 + yij**2

		# Add noise to sensor
		if self.noise != 0.00:
			s = (dsqr) * (self.noise)/3.0 + np.finfo(float).eps
			# Setup Normal-RNG to operate with max "noise" percent error with an error of 99.7% -> 3s = e
			dsqr = np.random.normal(dsqr, s)

		dist = np.sqrt(dsqr)
	
		# Control equation.
		dU = np.multiply(self.alpha, (dist - self.const + 1.0/dist - self.const/dsqr))
	
		ax = np.multiply(-dU, xij)/dist - vxij # damping
		ay = np.multiply(-dU, yij)/dist - vyij # damping
	
		# a(i, :) -> acceleration input for robot i.
		self.a = np.array([np.nansum(ax, axis=1), np.nansum(ay, axis=1)]).T
		# Add noise to movement
		# s = (abs(a)*self.noise/3.0) + np.finfo(float).eps
		#a = np.random.normal(a, s)
	
	 	# simple taylor expansion.
		self.q = self.q + self.v*self.dt + self.a*(0.5*self.dt**2)
		self.v = self.v + self.a*self.dt
	
	def feature(self):
		return self.metric.compute(self.q)
