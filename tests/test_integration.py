from oogillespie import Gillespie, Event
from math import sqrt
from pytest import approx, fixture, mark
from numpy import diag

def binomial_approx(n,p):
	"""
	Returns an approx object for the binomial distribution with n samples and probability p. Compare a number with this object to see whether its compatible with this distribution.
	"""
	return approx(n*p,abs=3*sqrt(n*p*(1-p)))

def poisson_approx(t,λ):
	"""
	Returns an approx object for the Poisson distribution with a total time t and rate λ. Compare a number with this object to see whether its compatible with this distribution.
	"""
	return approx(t*λ,3*sqrt(λ))

rates = (1,2,3,0,0)
max_time = 2e3

class ProcessBase(Gillespie):
	def initialise(self):
		self.N = [0,0,0]
	
	def state(self):
		return self.N

class SimpleProcess(ProcessBase):
	@Event(rates[0])
	def increase_N0(self):
		self.N[0] += 1
	
	def N1_rate(self):
		return rates[1]
	
	@Event(N1_rate)
	def increase_N1(self):
		self.N[1] += 1
	
	@Event(rates[2])
	def increase_N2(self):
		self.N[2] += 1
	
	@Event(0)
	def impossible_event(self):
		raise AssertionError("This event should never happen")

class MultiEventProcess(ProcessBase):
	@Event(rates)
	def increase(self,i):
		self.N[i] += 1

class TwoMultiEventProcess(ProcessBase):
	@Event(rates[:2])
	def increase_01(self,i):
		self.N[i] += 1
	
	@Event(rates[2:])
	def increase_2(self,i):
		self.N[i+2] += 1

class MultiEventProcess2D(ProcessBase):
	@Event(diag(rates))
	def increase(self,i,j):
		assert i==j
		self.N[i] += 1

class VariableRateMultiEventProcess(ProcessBase):
	def increase_rate(self):
		return rates
	
	@Event(increase_rate)
	def increase(self,i):
		self.N[i] += 1

class VariableRateMultiEventProcess2D(ProcessBase):
	def increase_rate(self):
		return diag(rates)
	
	@Event(increase_rate)
	def increase(self,i,j):
		assert i==j
		self.N[i] += 1

@mark.parametrize("Process",[
		SimpleProcess,
		MultiEventProcess,
		MultiEventProcess2D,
		TwoMultiEventProcess,
		VariableRateMultiEventProcess,
		VariableRateMultiEventProcess2D,
	])
class TestIntegration(object):
	@fixture
	def process(self,Process):
		proc = Process(max_t=max_time,max_steps=1e10)
		for _ in proc:
			pass
		return proc
	
	def test_step_number(self,process):
		assert sum(process.state()) == process.steps == poisson_approx(max_time,sum(rates))
	
	def test_rates(self,process):
		for i,number in enumerate(process.state()):
			assert number == binomial_approx(process.steps,rates[i]/sum(rates))
	
	def test_time(self,process):
		assert process.time == max_time
	
	def test_seeds(self,Process):
		seed = 42
		run_1 = list(Process(seed=42))
		run_2 = list(Process(seed=42))
		run_3 = list(Process(seed=23))
		
		assert run_1 == run_2
		assert run_3 != run_1

