"""
This is an example on how to use the Gillespie class.
It implements a solution to task 2D from week 5’s exercise sheet.
"""

from oogillespie import Gillespie

α = 0.1
μ = 0.00005
ω_0 = 0.001

class birth_death(Gillespie):
	def initialise(self,N=0,ω=ω_0):
		# This defines the initial condition; arguments can be passed when initialising an instance of the class.
		self.N = N
		self.ω = ω
	
	# The decorator “@Gillespie.event” marks this as an event. The rate is given as an argument to the decorator.
	@Gillespie.event(α)
	def birth(self):
		self.N += 1
	
	# If the rate is not constant, use the decorator without an argument, …
	@Gillespie.event
	def death(self):
		self.N -= 1
	
	# … and mark the function that determines the rate with another decorator:
	@death.rate
	def death_prob(self):
		return self.ω*self.N
	
	@Gillespie.event(μ)
	def mutation(self):
		self.ω /= 2
	
	# What is returned when iterating over the object:
	def state(self):
		return self.time,self.N

# We now create an instance of our class and immedeately iterate over it:
for time,N in birth_death(max_steps=10000,N=0):
	print(time,N)

