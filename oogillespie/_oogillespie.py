import numpy as np
import random
from inspect import signature
from itertools import product
from functools import partial

class GillespieUsageError(SyntaxError):
	pass

class Event(object):
	"""
	Decorator that marks a method as an event.

	There are several valid use cases of the argument `rates` of the decorator and the number of arguments of the decorated method:

	* The event method has no arguments other than `self`, and `rates` is a non-negative number specifying the rate of the event.

	* The event has one argument other than `self`, and `rates` is a sequence of non-negative numbers. In this case, the numbers specify the rates of different variants of the event. If an event happens, the location of the respective rate in the sequence is passed as an argument to the event method.

	* A generalisation of the above: The event has k arguments other than `self`, and `rates` is a nested sequence of non-negative numbers as an argument, with k levels of nesting. If an event happens, the location of the respective rate in the sequence is passed as arguments to the event method. All sequences on the same level must have the same length.

	* `rates` is a method returning a number or (nested) sequence of numbers as above. Note that the method must be defined before the event method.
	"""
	
	def __init__(self,rates):
		self._variable = callable(rates)
		if self._variable:
			# Retaining name for error message, if called without an argument
			self.__name__ = rates.__name__
			self._rate_function = rates
		else:
			self._rates = np.asarray(rates)
	
	def __call__(self,function):
		self._action = function
		self.__name__ = function.__name__
		self.dim = len(signature(self._action).parameters)-1
		return self
	
	def actions(self):
		for combo in self._par_combos:
			yield partial(self._action,self.parent,*combo)
	
	@property
	def rates(self):
		if self._variable:
			return np.asarray(self._rate_function(self.parent))
		else:
			return self._rates
	
	def get_rates(self):
		return self.rates[self._transposed_par_combos]
	
	@property
	def parent(self):
		return self._parent
	
	@parent.setter
	def parent(self,new_parent):
		if not hasattr(self,"_action"):
			raise GillespieUsageError(f"Decorator for event {self.__name__} has no rate argument.")
		self._parent = new_parent
		shape = self.rates.shape
		if self.dim != len(shape):
			raise GillespieUsageError(f"Length of rate does not match number of arguments of event {self._action.__name__}.")
		self._par_combos = list(product(*map(range,shape)))
		self._transposed_par_combos = tuple(map(np.array,zip(*self._par_combos)))

class Gillespie(object):
	"""
	This class only works if inherited from and if the methods `initialise` and `state` are replaced. Also, at least one method has to be marked as an event with the respective decorator.
	
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
		self._R = random.Random(kwargs.pop("seed",None))
		
		self.initialise(**kwargs)
		self._get_events()
		self._n = len(self._actions)
		# dummy array for efficiency:
		self._rates = np.empty(self._n)
	
	def _get_events(self):
		self._actions = []
		self._rate_getters = []
		
		for member in self._members(Event):
			member.parent = self
			self._actions.extend(member.actions())
			self._rate_getters.append(member.get_rates)
		
		if not self._actions:
			raise GillespieUsageError("No event defined. You need to mark at least one method as an event by using the Event decorator.")
	
	def _members(self,Class):
		"""
		Provides an iterator over all members of `self` that are an instance of `Class`.
		"""
		visited = set()
		for cls in [self.__class__] + self.__class__.mro():
			for name,member in cls.__dict__.items():
				if name not in visited:
					if isinstance(member,Class):
						yield member
					visited.add(name)
	
	def _get_cum_rates(self):
		i = 0
		for rate_getter in self._rate_getters:
			rates = rate_getter()
			j = i+rates.size
			self._rates[i:j] = rates
			i = j
		
		return np.cumsum(self._rates)
	
	def __next__(self):
		cum_rates = self._get_cum_rates()
		total_rate = cum_rates[-1]
		dt = self._R.expovariate(total_rate)
		
		if self.time+dt>self.max_t or self.steps>=self.max_steps:
			raise StopIteration
		
		self.steps += 1
		self.time += dt
		
		self._R.choices(self._actions,cum_weights=cum_rates)[0]()
		return self.state()
	
	def __iter__(self):
		return self
	
	def initialise(self):
		"""
		You have to overwrite this method. It is called when initialising the integrator. Use it to set internal parameters to initial values. It gets passed surplus arguments from the constructor.
		"""
		raise GillespieUsageError("You have to overwrite the initiate method when inheriting from Gillespie")
	
	def state(self):
		"""
		You have to overwrite this method. It is called to determine the return value when iterating/simulating. Use it to return whatever properties you are interested in.
		"""
		raise GillespieUsageError("You have to overwrite the state method when inheriting from Gillespie")

