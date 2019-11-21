# Implement Cluster and Radial Metrics

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from numba import jit


class ClusterMetric(object):
	"""	This class implement a metric to evaluate the pairwise intersection area 
	between the convex hulls of the robots acoording to their type partition. 
	That is, the convex hull of all robots of type k is computed and 
	intersected with each other. """
	def __init__(self, GROUPS, ROBOTS):
		self.GROUPS = GROUPS
		self.ROBOTS = ROBOTS

	def compute(self, q):
	# Compute for each combination of groups of robots the intersection area and accumulate it
		mclu = 0.0
		for i in range(self.GROUPS):
			idx_i = (int(math.floor((i) * self.ROBOTS/self.GROUPS)), int(math.floor((i+1) * self.ROBOTS/self.GROUPS)))
			qi = q[idx_i[0]:idx_i[1]]
			for j in range(self.GROUPS):
				idx_j = (int(math.floor((j) * self.ROBOTS/self.GROUPS)), int(math.floor((j+1) * self.ROBOTS/self.GROUPS)))
				qj = q[idx_j[0]:idx_j[1]]
				if idx_i != idx_j:
					mclu += self.compute_area(qi, qj)
		return mclu

	#def feature(self):
	# Return the sum of area of all combinations of intersection between convex hulls formed by group of robots.
	#	return self.mclu

	def compute_area(self, A, B):
	# Compute the area between two polygons formed by two convex hull.
		# Evaluate a convex hull for two groups of robots
		A_hull = ConvexHull(A)
		B_hull = ConvexHull(B)

		# Get convex hull points
		A_hull = A[A_hull.vertices]
		B_hull = B[B_hull.vertices]

		# Transform points to polygon
		A_poly = Polygon(A_hull)
		B_poly = Polygon(B_hull)

		# Compute the intersections points between the two polygons
		AB_inter = A_poly.intersection(B_poly)

		# Then return the area of the intersection
		return AB_inter.area

class RadialMetric(object):
	"""This class implement Gross et al. (2009) metric. 
	Their metric requires robots to segregate around a particulary stationary point c."""
	def __init__(self, GROUPS, ROBOTS, const):
		self.GROUPS = GROUPS
		self.ROBOTS = ROBOTS
		self.const = const

	def compute(self, q):
		# This method implement the radial segregation metric. It counts the number of robots outrside the 
		# correct spherical shells according to the values of each dAA_k and normalizes the result into the unit interval.
		self.mrad = 0.0
		self.q = q
		# Compute the stationary point c
		self.n = float(len(q))
		self.c = np.sum(self.q, axis=0)/self.n


		#for i in range(self.ROBOTS):
		#	for j in range(self.ROBOTS):
		#		self.mrad += self.radial_error(i, j)

		self.mrad = fast(self.ROBOTS, self.const, self.q, self.c)
		self.mrad /= (self.n**2)
		return self.mrad

	def feature(self):
		# Return the feature of the current segregation state
		return self.mrad

	def radial_error(self, i, j):
		# Apply a cost when robots which sould be "inside" the other and closer to the stationary point
		dAAk = self.const[i, i]
		dAAl = self.const[j, j]
		if dAAk != dAAl: # if dissimilar groups
			if dAAk < dAAl and self.distToC(i) >= self.distToC(j):
				return 1.0
			if dAAk > dAAl and self.distToC(i) < self.distToC(j):
				return 1.0
		return 0.0

	# Compute the Euclidean distance from robot r to the stationary point c
	def distToC(self, r):
		d = self.q[r] - self.c
		dist = np.sqrt(d[0]**2 + d[1]**2)
		return dist

@jit
def fast(ROBOTS, const, q, c):
	mrad = 0.0
	for i in range(ROBOTS):
		for j in range(ROBOTS):
			mrad += radial_error(i, j, const, q, c)
	return mrad

@jit
def radial_error( i, j, const, q, c):
		# Apply a cost when robots which sould be "inside" the other and closer to the stationary point
		dAAk = const[i, i]
		dAAl = const[j, j]
		if dAAk != dAAl: # if dissimilar groups
			if dAAk < dAAl and distToC(i, q, c) >= distToC(j, q, c):
				return 1.0
			if dAAk > dAAl and distToC(i, q, c) < distToC(j, q, c):
				return 1.0
		return 0.0

@jit
def distToC(r, q, c):
		d = q[r] - c
		dist = np.sqrt(d[0]**2 + d[1]**2)
		return dist