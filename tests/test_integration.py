from oogillespie import Gillespie
from math import sqrt
from pytest import approx, fixture

def binomial_approx(n,p):
	return approx(n*p,abs=3*sqrt(n*p*(1-p)))

def poisson_approx(t,λ):
	return approx(t*λ,3*sqrt(λ))

rates = (1,2,3)
max_time = 2e3

class SimpleProcess(Gillespie):
	def initialise(self):
		self.a = 0
		self.b = 0
		self.c = 0
	
	@Gillespie.event(rates[0])
	def increase_a(self):
		self.a += 1
	
	@Gillespie.event
	def increase_b(self):
		self.b += 1
	
	@increase_b.rate
	def b_rate(self):
		return rates[1]
	
	@Gillespie.event(rates[2])
	def increase_c(self):
		self.c += 1
	
	@Gillespie.event(0)
	def impossible_event(self):
		raise AssertionError("This event should never happen")
	
	def state(self):
		return self.a, self.b, self.c

class TestIntegration(object):
	@fixture
	def process(self):
		proc = SimpleProcess(max_t=max_time,max_steps=1e10)
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
	

