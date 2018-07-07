from oogillespie import Gillespie
from math import sqrt
from pytest import approx, fixture, mark
from numpy import diag

def binomial_approx(n,p):
	return approx(n*p,abs=3*sqrt(n*p*(1-p)))

def poisson_approx(t,λ):
	return approx(t*λ,3*sqrt(λ))

rates = (1,2,3,0)
max_time = 2e3

class ProcessBase(Gillespie):
	def initialise(self):
		self.N = [0,0,0]
	
	def state(self):
		return self.N

class SimpleProcess(ProcessBase):
	@Gillespie.event(rates[0])
	def increase_N0(self):
		self.N[0] += 1
	
	@Gillespie.event
	def increase_N1(self):
		self.N[1] += 1
	
	@increase_N1.rate
	def N1_rate(self):
		return rates[1]
	
	@Gillespie.event(rates[2])
	def increase_N2(self):
		self.N[2] += 1
	
	@Gillespie.event(0)
	def impossible_event(self):
		raise AssertionError("This event should never happen")

class MultiEventProcess(ProcessBase):
	@Gillespie.event(rates)
	def increase(self,i):
		if i<3:
			self.N[i] += 1
		else:
			raise AssertionError("This event should never happen")

class MultiEventProcess2D(ProcessBase):
	@Gillespie.event(diag(rates))
	def increase(self,i,j):
		assert i==j
		if i<3:
			self.N[i] += 1
		else:
			raise AssertionError("This event should never happen")

class VariableRateMultiEventProcess(ProcessBase):
	@Gillespie.event
	def increase(self,i):
		if i<3:
			self.N[i] += 1
		else:
			raise AssertionError("This event should never happen")
	
	@increase.rate
	def increase_rate(self):
		return rates

class VariableRateMultiEventProcess2D(ProcessBase):
	@Gillespie.event
	def increase(self,i,j):
		assert i==j
		if i<3:
			self.N[i] += 1
		else:
			raise AssertionError("This event should never happen")
	
	@increase.rate
	def increase_rate(self):
		return diag(rates)

@mark.parametrize("Process",[
		SimpleProcess,
		MultiEventProcess,
		MultiEventProcess2D,
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
		assert process.time < max_time
		assert process.time > max_time - 10/sum(rates)
	
	def test_seeds(self,Process):
		seed = 42
		run_1 = list(Process(seed=42))
		run_2 = list(Process(seed=42))
		run_3 = list(Process(seed=23))
		
		assert run_1 == run_2
		assert run_3 != run_1

