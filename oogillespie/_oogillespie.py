import numpy as np
import random
from inspect import signature
from itertools import product, chain
from functools import partialmethod, partial

# Classes for events
class Event(object):
	def __init__(self,function):
		self.action = function
		self.dim = len(signature(self.action).parameters)-1
	
	def check_dim(self):
		if self.dim != len(self.shape):
			raise SyntaxError(f"Length of rate does not match number of arguments of event {self.action.__name__}.")
	
	def par_combos(self):
		pars = list(signature(self.action).parameters.keys())[1:]
		for combo in product(*map(range,self.shape)):
			kwargs = dict(zip(pars,combo))
			yield combo, kwargs

class FixedRateEvent(Event):
	def __init__(self,function,rates):
		super().__init__(function)
		self._rates = np.array(rates)
		self.shape = self._rates.shape
		self.check_dim()
	
	def get_rates(self):
		for combo,_ in self.par_combos():
			yield self._rates[combo]

class VariableRateEvent(Event):
	def rate(self,function):
		self._rate_getter = function
	
	@property
	def rate_getter(self):
		try:
			return self._rate_getter
		except AttributeError:
			raise SyntaxError(f"No rate function was assigned to variable-rate event {self.action.__name__}.")
	
	def get_rates(self):
		rates = np.array(self.rate_getter(self.parent))
		for combo,_ in self.par_combos():
			yield rates[combo]
	
	@property
	def parent(self):
		return self._parent
	
	@parent.setter
	def parent(self,new_parent):
		self._parent = new_parent
		self.shape = np.shape(self.rate_getter(self._parent))

	@property
	def shape(self):
		return self._shape
	
	@shape.setter
	def shape(self,new_shape):
		self._shape = new_shape
		self.check_dim()


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
	
	seed = None
		Seed for random number generation.
		If `None`, system time or similar is used (like random.seed).
	"""
	
	def __init__(self,**kwargs):
		self.time = kwargs.pop("time",0)
		self.steps = 0
		self.max_steps = kwargs.pop("max_steps",1000)
		self.max_t = kwargs.pop("max_t",np.inf)
		self.R = random.Random(kwargs.pop("seed",None))
		
		self.initialise(**kwargs)
		self._get_events()
	
	def _get_events(self):
		self.actions = []
		self.constant_rates = []
		self.rate_getters = []
		
		for name,member in self._members(FixedRateEvent):
			for combo,kwargs in member.par_combos():
				if member._rates[combo]:
					self.actions.append(partial(member.action,**kwargs))
					self.constant_rates.append(member._rates[combo])
		
		for name,member in self._members(VariableRateEvent):
			member.parent = self
			for combo,kwargs in member.par_combos():
				self.actions.append(partial(member.action,**kwargs))
			self.rate_getters.append(member.get_rates)
		
		if not self.actions:
			raise SyntaxError("No event (with non-zero rate) defined. You need to mark at least one method as an event by using the Gillespie.event decorator.")
	
	def _members(self,Class):
		"""
		Provides an iterator over all members of self that are an instance of Class.
		"""
		visited = set()
		for cls in [self.__class__] + self.__class__.mro():
			for name,member in cls.__dict__.items():
				if name not in visited:
					if isinstance(member,Class):
						yield name,member
					visited.add(name)
	
	def event(arg):
		"""
		Decorator that marks a method as an event.
		You can use this either with no or one argument:
		
		* If called with an argument, this must be a number specifying the rate of the event.
		
		* If called without argument, the function obtains a member `rate`, which is in turn a decorator that must be used to mark the function that returns the rate of that event.
		"""
		
		if callable(arg):
			return VariableRateEvent(arg)
		else:
			def wrapper(function):
				return FixedRateEvent(function,arg)
			return wrapper
	
	def _get_cum_rates(self):
		return np.cumsum(self.constant_rates + [
				rate
				for rate_getter in self.rate_getters
				for rate in rate_getter()
			])
	
	def __next__(self):
		cum_rates = self._get_cum_rates()
		total_rate = cum_rates[-1]
		dt = self.R.expovariate(total_rate)
		
		if self.time+dt>self.max_t or self.steps>=self.max_steps:
			raise StopIteration
		
		self.steps += 1
		self.time += dt
		
		self.R.choices(self.actions,cum_weights=cum_rates)[0](self)
		return self.state()
	
	def __iter__(self):
		return self
	
	def initialise(self):
		"""
		You have to overwrite this method. It is called when initialising the integrator. Use it to set internal parameters to initial values. It gets passed surplus arguments from the constructor.
		"""
		raise SyntaxError("You have to overwrite the initiate method when inheriting from Gillespie")
	
	def state(self):
		"""
		You have to overwrite this method. It is called to determine the return value when iterating/simulating. Use it to return whatever properties you are interested in.
		"""
		raise SyntaxError("You have to overwrite the state method when inheriting from Gillespie")

