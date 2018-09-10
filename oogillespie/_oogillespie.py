import numpy as np
import random
from warnings import warn
from inspect import signature
from itertools import product
from functools import partial, update_wrapper

class GillespieUsageError(SyntaxError):
	pass

class Event(object):
	"""
	Decorator that marks a method as an event.
	
	There are several valid use cases of the argument `rates` of the decorator and the number of arguments of the decorated method:
	
	* The event method has no arguments other than `self`, and `rates` is a non-negative number specifying the rate of the event.
	
	* The event has one argument other than `self`, and `rates` is a sequence of non-negative numbers. In this case, the numbers specify the rates of different variants of the event. If an event happens, the location of the respective rate in the sequence is passed as an argument to the event method.
	
	* A generalisation of the above: The event has k arguments other than `self`, and `rates` is a nested sequence of non-negative numbers as an argument, with k levels of nesting. If an event happens, the location of the respective rate in the sequence is passed as arguments to the event method. All sequences on the same level must have the same length.
	
	* `rates` is a method returning a number or (nested) sequence of numbers as above.
	
	In any case the number, sequence, or method providing the rates must be defined before the event method.
	"""
	
	def __init__(self,rates):
		self._is_variable = callable(rates)
		if self._is_variable:
			self._rate_function = rates
			# Retaining name for error message, if called without an argument:
			self.__name__ = rates.__name__
		else:
			self._rates = np.asarray(rates)
			if np.any(self._rates<0):
				raise ValueError("Rates must be non-negative.")
	
	def __call__(self,function):
		# The decorated function is passed through this and replaced by its output.
		self._action = function
		update_wrapper(self,function)
		self.dim = len(signature(self._action).parameters)-1
		return self
	
	@property
	def _was_called(self):
		return hasattr(self,"_action")
	
	@property
	def rates(self):
		if self._is_variable:
			return np.asarray(self._rate_function(self._parent))
		else:
			return self._rates
	
	def get_flattened_rates(self):
		return self.rates[self._transposed_par_combos]
	
	def set_parent(self,parent):
		"""
		Sets the instance of the class on which the event is to be executed. Without this, the event is useless.
		"""
		self._parent = parent
		
		if not self._was_called:
			raise GillespieUsageError(f"Decorator for event {self.__name__} has no rate argument.")
		
		shape = self.rates.shape
		if self.dim != len(shape):
			raise GillespieUsageError(f"Length of rate does not match number of arguments of event {self.__name__}.")
		
		self._par_combos = list(product(*map(range,shape)))
		self._transposed_par_combos = tuple(map(np.array,zip(*self._par_combos)))
	
	def actions(self):
		for combo in self._par_combos:
			yield partial(self._action,self._parent,*combo)

class Gillespie(object):
	"""
	This class only works if inherited from and if the methods `initialise` and `state` are replaced. Also, at least one method has to be marked as an event with the respective decorator.
	
	Parameters
	----------
	time
		The starting time.
	
	max_steps
	max_t
		The maximum number of steps or time, respectively.
		Before either of the two is exceeded (or all event rates become zero), the simulation is aborted.
	
	seed
		Seed for random number generation.
		If `None`, system time or similar is used (like random.seed).
	
	kwargs
		Keyword arguments to be forwarded to `initialise`.
	"""
	
	def __init__(self,max_steps=1000,max_t=np.inf,time=0,seed=None,**kwargs):
		self.time = time
		self.max_steps = max_steps
		self.max_t = max_t
		self._R = random.Random(seed)
		self.steps = 0
		
		self.initialise(**kwargs)
		self._get_events()
		self._n = len(self._actions)
		# dummy array for efficiency:
		self._rates = np.empty(self._n)
	
	def _get_events(self):
		self._actions = []
		self._rate_getters = []
		
		for event in self._members(Event):
			event.set_parent(self)
			self._actions.extend(event.actions())
			self._rate_getters.append(event.get_flattened_rates)
		
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
		
		return self._rates.cumsum()
	
	def __next__(self):
		cum_rates = self._get_cum_rates()
		total_rate = cum_rates[-1]
		try:
			dt = self._R.expovariate(total_rate)
		except ZeroDivisionError:
			dt = np.inf
		
		if self.time+dt > self.max_t:
			self.time = self.max_t
			raise StopIteration
		
		if self.steps >= self.max_steps:
			raise StopIteration
		
		self.steps += 1
		self.time += dt
		
		self._R.choices(self._actions,cum_weights=cum_rates)[0]()
		return self.state()
	
	def __iter__(self):
		return self
	
	def initialise(self):
		"""
		You must overwrite this method. It is called when initialising the integrator. Use it to set internal parameters to initial values. It gets passed surplus arguments from the constructor.
		"""
		raise GillespieUsageError("You must overwrite the initiate method when inheriting from Gillespie")
	
	def state(self):
		"""
		You must overwrite this method. It is called to determine the return value when iterating/simulating. Use it to return whatever properties you are interested in.
		"""
		raise GillespieUsageError("You must overwrite the state method when inheriting from Gillespie")

