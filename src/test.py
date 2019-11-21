#!/usr/bin/env python
#import numpy as np
from scipy.stats import norm
from scipy.stats import invgauss
import time
from scipy import signal

#last = time.clock()
#mu, sigma = 0, 2 # mean and standard deviation
#
#e_max = 0.05
#
#
#def check(q_, q, e):
#	r = abs(q_-q)/q
#	if r > e:
#		print "Check:", q, "with", q_, "Failed", r, e
#	else:
#		return
#		print "Check:", q, "with", q_, "Pass", r, e
#
#q = 15
#e = 0.05
#
#
#s = (q + np.finfo(float).eps) * (e)/3.0
#
#for i in range(1000):
#	q_ = np.random.normal(q, s)
#	check(q_, q, e)
#
from termcolor import colored

#a = range(10)
#filename = "test.npz"
#np.savez(filename, a)
#b = range(1,6)
#np.savez(filename, b)
#c = np.load(filename)
#a = -3


#dAA = np.linspace(5, 25, 5) # radial
#print dAA

exp = 100
it = 10000
dim = 3
print exp*it*dim
print np.log10(10000)