import numpy as np
from itertools import chain
import random

# Classes for events
class Event(object):
	def __init__(self,function):
		self.action = function
	
	def rate(self,function):
		self.rate_getter = function

class FixedRateEvent(object):
	def __init__(self,function,rate):
		self.action = function
		self._rate = rate

class Gillespie(object):
	"""
	This class only works if inherited from and if the methods `initialise` and `state` are replaced. Also, at least method has to be marked as an event with the respective decorator.
	
	The constructor takes the following arguments.
	All further arguments are forwarded to `initialise`.
	
	Parameters
	----------
	time = 0
		The starting time.
	
	max_steps = 1000
	max_t = ∞
		The maximum number of steps or time, respectively.
		If either of the two is exceeded, the simulation is aborted.
	
	"""
	
	def __init__(self,**kwargs):
		self.time = kwargs.pop("time",0)
		self.steps = 0
		self.max_steps = kwargs.pop("max_steps",1000)
		self.max_t = kwargs.pop("max_t",np.inf)
		self.initialise(**kwargs)
		self._get_events()
	
	def _get_events(self):
		self.actions = []
		self.constant_rates = []
		self.rate_getters = []
		
		for name,member in self._members():
			if isinstance(member,FixedRateEvent):
				self.actions.append(member.action)
				self.constant_rates.append(member._rate)
		
		for name,member in self._members():
			if isinstance(member,Event):
				self.actions.append(member.action)
				try:
					rate_getter = member.rate_getter
				except AttributeError:
					raise SyntaxError(f"No rate function was assigned to variable-rate event {name}.")
				self.rate_getters.append(rate_getter)
				
	
	def _members(self):
		visited = set()
		for cls in [self.__class__] + self.__class__.mro():
			for name,member in cls.__dict__.items():
				if name not in visited:
					yield name,member
					visited.add(name)
	
	def event(arg):
		"""
		Decorator that marks a method as an event.
		You can use this either with no or one argument:
		
		* If called with an argument, this must be a number specifying the rate of the event.

		* If called witaout argument, the function obtains a member `rate`, which is in turn a decorator that must be used to mark the function that returns the rate of that event.
		"""
		
		if callable(arg):
			return Event(arg)
		else:
			def wrapper(function):
				return FixedRateEvent(function,arg)
			return wrapper
	
	def _get_cum_rates(self):
		return np.cumsum(self.constant_rates + [
				rate_getter(self)
				for rate_getter in self.rate_getters
			])
	
	def __next__(self):
		cum_rates = self._get_cum_rates()
		total_rate = cum_rates[-1]
		dt = random.expovariate(total_rate)
		
		if self.time+dt>self.max_t or self.steps>=self.max_steps:
			raise StopIteration
		
		self.steps += 1
		self.time += dt
		
		random.choices(self.actions,cum_weights=cum_rates)[0](self)
		return self.state()
	
	def __iter__(self):
		return self
	
	def initialise(self):
		"""
		You have to overwrite this method. It is called when intialising the integrator. Use it to set internal parameters to initial values. It gets passed surplus arguments from the constructor.
		"""
		raise SyntaxError("You have to overwrite the initiate method when inheriting from Gillespie")
	
	def state(self):
		"""
		You have to overwrite this method. It is called to determine the return value when iterating/simulating. Use it to return whatever properties you are interested in.
		"""
		raise SyntaxError("You have to overwrite the state method when inheriting from Gillespie")

