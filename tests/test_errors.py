from oogillespie import Gillespie
from pytest import raises

class ProcessWithoutInitialise(Gillespie):
	pass

def test_no_initialise():
	with raises(SyntaxError):
		ProcessWithoutInitialise()

class ProcessWithoutState(Gillespie):
	def initialise(self):
		self.a = 0
	
	@Gillespie.event(1)
	def increase_a(self):
		self.a += 1

def test_no_state():
	with raises(SyntaxError):
		for state in ProcessWithoutState():
			pass

class ProcessWithoutRate(Gillespie):
	def initialise(self):
		self.a = 0
	
	@Gillespie.event
	def increase_a(self):
		self.a += 1

def test_no_rate():
	with raises(SyntaxError):
		ProcessWithoutRate()

class ProcessWithoutEvent(Gillespie):
	def initialise(self):
		self.a = 0
	
	def state(self):
		return self.a

def test_no_event():
	with raises(SyntaxError):
		ProcessWithoutEvent()


