#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
We want to implement a simple birth–death process where births happen with a constant rate of :math:`α=0.1` and each individual dies with a rate of :math:`ω=0.001`, starting at a population of :math:`N=0`.

We start with importing the Gillespie class as well as setting the control parameters:

.. literalinclude:: ../examples/birth_death.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 1-4

We now start defining the class for our specific process.
It inherits from `Gillespie` and most of the remainder of this example will happen in the class definition.

.. literalinclude:: ../examples/birth_death.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 6

The first of the method we need is `initialise`, which is automatically called upon initialisation and sets the initial state of the system. It gets passed parameters from the constructor calls. Here we only need to set :math:`N`, which we want to default to `0`:

.. literalinclude:: ../examples/birth_death.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 7-8

Next we implement the birth event. For this, we need a method that is called upon such an event and increases :math:`N` by 1. To register this method as an event and specify its rate, we use the decorator `Gillespie.event` and provide the constant rate :math:`α` as an argument:

.. literalinclude:: ../examples/birth_death.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 10-12

Now, we implement the death event. The main difference to birth events is that the rate is not constant. Therefore we use the `event` decorator without an argument. We then use the event-specific decorator `death.rate` to mark the function which calculates the rate of the event, here `death_prob`:

.. literalinclude:: ../examples/birth_death.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 14-20

Finally, we implement a function `state` that returns whatever we consider relevant information about the state of the system.
Here it is the time and :math:`N`:

.. literalinclude:: ../examples/birth_death.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 22-23

This concludes the definition of the class. We can now simulate a realisation of such a process by iterating over an instance of this class. To make the loop terminate, we have to specify either the maximum number of steps or the maximum time as an argument. We here want the maximum number of steps to be 10000. Finally, to keep this example simple, we print the result, but we could as well further process or plot it:

.. literalinclude:: ../examples/birth_death.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 25-26

Taken together, our script looks like this:

.. literalinclude:: ../examples/birth_death.py
	:start-after: example-st\u0061rt
	:dedent: 1
"""

if __name__ == "__main__":
	# example-start
	from oogillespie import Gillespie
	
	α = 0.1
	ω = 0.001
	
	class birth_death(Gillespie):
		def initialise(self,N=0):
			self.N = N
		
		@Gillespie.event(α)
		def birth(self):
			self.N += 1
		
		@Gillespie.event
		def death(self):
			self.N -= 1
		
		@death.rate
		def death_prob(self):
			return ω*self.N
		
		def state(self):
			return self.time,self.N
	
	for time,N in birth_death(max_steps=10000):
		print(time,N)

