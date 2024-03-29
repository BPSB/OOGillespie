from abc import ABC, abstractmethod
import numpy as np
import random
from inspect import signature
from itertools import product
from functools import partial, update_wrapper

class GillespieUsageError(SyntaxError):
	pass

class Event(object):
	"""
	Decorator that marks a method (of a subclass of Gillespie) as an event.
	
	There are several valid use cases of the argument `rates` of the decorator and the number of arguments of the decorated method:
	
	* The event method has no arguments other than `self`, and `rates` is a non-negative number specifying the rate of the event.
	
	* The event has one argument other than `self`, and `rates` is a sequence of non-negative numbers. In this case, the numbers specify the rates of different variants of the event. If an event happens, the location of the respective rate in the sequence is passed as an argument to the event method.
	
	* A generalisation of the above: The event has k arguments other than `self`, and `rates` is a nested sequence of non-negative numbers as an argument, with k levels of nesting. All sequences on the same level must have the same length. For example, for k=2, `rates` is a matrix. If an event happens, the location of the respective rate in the sequence is passed as arguments to the event method.
	
	* `rates` is a method returning a number or (nested) sequence of numbers as above. This method must take `self` and only `self` as an argument.
	
	In any case the number, sequence, or method providing the rates must be defined before the event method.
	"""
	
	def __init__(self,rates):
		self._has_variable_rates = callable(rates)
		if self._has_variable_rates:
			self._rate_method = rates
			# Retaining name for error message of _assert_decorator_argument:
			self.__name__ = rates.__name__
		else:
			self._rates = np.asarray(rates)
			if np.any(self._rates<0):
				raise ValueError("Rates must be non-negative.")
	
	def __call__(self,method):
		# The decorated method is passed through this and replaced by its output, i.e., this instance of Event.
		self._method = method
		# Retain method’s metadata:
		update_wrapper(self,method)
		self.k = len(signature(self._method).parameters)-1
		return self
	
	def _assert_decorator_argument(self):
		# If the decorator has an argument, it was called (self.__call__) and thus it must have the attribute _method:
		if not hasattr(self,"_method"):
			raise GillespieUsageError(f"Decorator for event {self.__name__} has no rate argument.")
	
	@property
	def rates(self):
		if self._has_variable_rates:
			return np.asarray(self._rate_method(self._parent))
		else:
			return self._rates
	
	def set_parent(self,parent):
		"""
		Sets the instance of the class (usually a subclass of Gillespie) on which the event is to be executed. Without this, the event is useless. Gillespie calls this when registering Events.
		"""
		self._parent = parent
		self._assert_decorator_argument()
		
		shape = self.rates.shape
		if self.k != len(shape):
			raise GillespieUsageError(f"Length of rate does not match number of arguments of event {self.__name__}.")
		
		# flattened iterables of argument combinations for uniquely mapping actions and rates in 1D.
		self._par_combos = list(product(*map(range,shape)))
		self._transposed_par_combos = tuple(map(np.array,zip(*self._par_combos)))
	
	def get_flattened_rates(self):
		return self.rates[self._transposed_par_combos]
	
	def actions(self):
		"""
		Yields zero-argument functions (actions) for each variant of the event.
		"""
		for combo in self._par_combos:
			yield partial(self._method,self._parent,*combo)

class Gillespie(ABC):
	"""
	Base class for Gillespie models. This class only works if inherited from and if the methods `initialise` and `state` are replaced. Also, at least one method has to be marked as an event with the respective decorator.
	
	When an instance of the class is iterated over, the actual simulation is performed and the iterator returns the output of `state` after each step.
	
	Parameters
	----------
	time
		The starting time.
	
	max_steps
	max_t
		The maximum number of steps or time, respectively. Before either of the two is exceeded (or all event rates become zero), the simulation is aborted.
	
	seed
		Seed for random number generation. If `None`, system time or similar is used (like `random.seed`).
	
	kwargs
		Keyword arguments to be forwarded to `initialise`.
	"""
	
	def __init__(self,max_steps=1000,max_t=np.inf,time=0,seed=None,**kwargs):
		self.time = time
		self.max_steps = max_steps
		self.max_t = max_t
		self._RNG = random.Random(seed)
		self.steps_taken = 0
		
		self.initialise(**kwargs)
		self._register_events()
		
		# dummy arrays for efficiency of _get_cum_rates:
		self._rates = np.empty(len(self._actions))
		self._cum_rates = np.empty(len(self._actions))
	
	def _register_events(self):
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
		for member in map(self.__getattribute__,dir(self)):
			if isinstance(member,Class):
				yield member
	
	def _compute_cum_rates(self):
		# Initialised in constructor for efficiency:
		# self._rates = np.empty(len(self._actions))
		# self._cum_rates = np.empty(len(self._actions))
		i = 0
		for rate_getter in self._rate_getters:
			new_rates = rate_getter()
			j = i+new_rates.size
			self._rates[i:j] = new_rates
			i = j
		
		self._rates.cumsum(out=self._cum_rates)
	
	def __next__(self):
		# Perform one step of the Gillespie algorithm
		self._compute_cum_rates()
		total_rate = self._cum_rates[-1]
		try:
			dt = self._RNG.expovariate(total_rate)
		except ZeroDivisionError:
			dt = np.inf
		
		if self.time+dt > self.max_t:
			self.time = self.max_t
			raise StopIteration
		
		if self.steps_taken >= self.max_steps:
			raise StopIteration
		
		self.steps_taken += 1
		self.time += dt
		
		self._RNG.choices(self._actions,cum_weights=self._cum_rates)[0]()
		return self.state()
	
	def __iter__(self):
		return self
	
	@abstractmethod
	def initialise(self):
		"""
		You must overwrite this method. It is called when initialising the integrator. Use it to set internal parameters to initial values. It gets passed surplus arguments from the constructor.
		"""
		pass
	
	@abstractmethod
	def state(self):
		"""
		You must overwrite this method. It is called to determine the return value when iterating/simulating. Use it to return whatever properties you are interested in.
		"""
		pass
