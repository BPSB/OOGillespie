#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

if __name__ == "__main__":
	# example-start
	from oogillespie import Gillespie
	
	A = np.array([
			[  0  , 0.5 ,  0  ],
			[  0  ,  0  , 0.3 ],
			[ 0.1 ,  0  ,  0  ],
		])
	
	α = [ 0.2, 0.01, 0.01 ]
	
	class RPS(Gillespie):
		def initialise(self,N=None):
			self.N = N or np.array([0,0,0])
		
		@Gillespie.event(α)
		def birth(self,i):
			self.N[i] += 1

		@Gillespie.event
		def eat(self,i,j):
			self.N[i] += 1
			self.N[j] -= 1
		
		@eat.rate
		def eat_prob(self):
			return self.N*A*self.N[:,None]
		
		def state(self):
			return self.time,self.N
	
	for time,N in RPS(max_steps=10000):
		print(time,*N)

