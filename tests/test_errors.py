from oogillespie import Gillespie
from pytest import raises

class ProcessWithoutInitialise(Gillespie):
	pass

def test_no_initialise():
	with raises(SyntaxError):
		ProcessWithoutInitialise()

class ProcessWithoutState(Gillespie):
	def initialise(self): pass
	
	@Gillespie.event(1)
	def do_nothing(self): pass

def test_no_state():
	with raises(SyntaxError):
		for state in ProcessWithoutState():
			pass

class ProcessWithoutRate(Gillespie):
	def initialise(self): pass
	
	@Gillespie.event
	def do_nothing(self): pass

def test_no_rate():
	with raises(SyntaxError):
		ProcessWithoutRate()

class ProcessWithoutEvent(Gillespie):
	def initialise(self): pass
	
	def state(self):
		return self.time

def test_no_event():
	with raises(SyntaxError):
		ProcessWithoutEvent()

class TestDimensionMismatch(object):
	def test_no_arg(self):
		with raises(SyntaxError):
			class ProcessWithRateDimensionMismatch(ProcessWithoutEvent):
				@Gillespie.event([1,2])
				def do_nothing(self): pass
	
	def test_too_many_args(self):
		with raises(SyntaxError):
			class ProcessWithRateDimensionMismatch(ProcessWithoutEvent):
				@Gillespie.event([[1,2],[3,4]])
				def do_nothing(self,i,j,k): pass

