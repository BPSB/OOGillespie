#!/usr/bin/python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
	# example-start
	from oogillespie import Gillespie
	
	A = [
			[  0  , 0.5 ,  0  ],
			[  0  ,  0  , 0.3 ],
			[ 0.1 ,  0  ,  0  ],
		]
	
	α = [ 0.2, 0.01, 0.0 ]
	
	class RPS(Gillespie):
		def initialise(self,N=None):
			self.N = N or [0,0,0]
		
		@Gillespie.multievent(α)
		def birth(self,i)
			self.N[i] += 1

		@Gillespie.multievent(3,3)
		def eat(self,i,j):
			self.N[i] += 1
			self.N[j] -= 1
		
		@eat.rate
		def eat_prob(self,i,j):
			return self.N[i] * self.N[j] * A[i,j]
		
		def state(self):
			return self.time,self.N
	
	for time,N in RPS(max_steps=10000):
		print(time,N)

