#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
We want to implement a process with three populations :math:`N_0, N_1, N_2` whose individuals transform into each other.
Individuals of population :math:`N_i` are born with a rate :math:`α_i`.
Transformations from :math:`i` to :math:`j` happen with a rate :math:`N_i·A_{ij}·N_j`.

More specifically, we want :math:`α = (0.2, 0.01, 0.01)` and the transformations to be cyclical (rock–paper–scissors):

.. math::

	A = \\pmatrix{
			 0  & 0.5 &  0  \\\\
			 0  &  0  & 0.3 \\\\
			0.1 &  0  &  0  \\\\
		}.

We start with importing the Gillespie class as well as setting the control parameters as NumPy arrays:

.. literalinclude:: ../examples/rock_paper_scissors.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 1-10

We then continue with defining the class for our process.
Nothing new happens in the `initialise` and `state` functions, except that `N` is an array:

.. literalinclude:: ../examples/rock_paper_scissors.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 12-17

In contrast to the previous example, there are now three different birth events, one for each population. However, being healthily lazy, we do not want to write three nearly identical functions for this but want to have a parametrised event, where the parameter specifies which population increases.

To achieve this, we pass a sequence (`α`) instead of a number as an argument to the `Event` decorator. This way, we specify that the birth event comes in different versions with different rates, one for each element of `α`. Whenever such an event happens, the respective index of `α` is passed to `birth` as an additional argument.

.. literalinclude:: ../examples/rock_paper_scissors.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 19-21

For the transformation event, two things change:

* The rate depends on the current state. This is handled like before, i.e., by passing a method instead of a number to the `Event` decorator.
* The event depends on two parameters. This is reflected by the parameters `i` and `j` of `transform` and `transform_rate` returning a two-dimensional array of rates instead of a number.

.. literalinclude:: ../examples/rock_paper_scissors.py
	:start-after: example-st\u0061rt
	:dedent: 1
	:lines: 23-29

Altogether our code would look like this:

.. literalinclude:: ../examples/rock_paper_scissors.py
	:start-after: example-st\u0061rt
	:dedent: 1
"""

if __name__ == "__main__":
	# example-start
	from oogillespie import Gillespie, Event
	import numpy
	
	α = numpy.array([ 0.2, 0.01, 0.01 ])
	
	A = numpy.array([
			[  0  , 0.5 ,  0  ],
			[  0  ,  0  , 0.3 ],
			[ 0.1 ,  0  ,  0  ],
		])
	
	class RPS(Gillespie):
		def initialise(self,N=None):
			self.N = numpy.asarray(N or [0,0,0])
		
		def state(self):
			return self.time,self.N
		
		@Event(α)
		def birth(self,i):
			self.N[i] += 1
		
		def transform_rate(self):
			return self.N*A*self.N[:,None]
		
		@Event(transform_rate)
		def transform(self,i,j):
			self.N[i] -= 1
			self.N[j] += 1
	
	for time,N in RPS(max_steps=100000):
		print(time,*N)
