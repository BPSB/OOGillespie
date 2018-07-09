from oogillespie import Gillespie, GillespieUsageError, Event
from pytest import raises

class ProcessWithoutInitialise(Gillespie):
	pass

def test_no_initialise():
	with raises(GillespieUsageError):
		ProcessWithoutInitialise()

class ProcessWithoutState(Gillespie):
	def initialise(self): pass
	
	@Event(1)
	def do_nothing(self): pass

def test_no_state():
	with raises(GillespieUsageError):
		for state in ProcessWithoutState():
			pass

class ProcessWithoutRate(Gillespie):
	def initialise(self): pass
	
	@Event
	def do_nothing(self): pass

def test_no_rate():
	with raises(GillespieUsageError) as exc:
		ProcessWithoutRate()
	assert "no rate argument" in exc.value.args[0]

class ProcessWithoutEvent(Gillespie):
	def initialise(self): pass
	
	def state(self):
		return self.time

def test_no_Event():
	with raises(GillespieUsageError):
		ProcessWithoutEvent()

class ProcessWithRateDimensionMismatch(ProcessWithoutEvent):
	@Event([1,2])
	def do_nothing(self): pass

def test_no_arg():
	with raises(GillespieUsageError):
		ProcessWithRateDimensionMismatch()

class ProcessWithRateDimensionMismatch2(ProcessWithoutEvent):
	@Event([[1,2],[3,4]])
	def do_nothing(self,i,j,k): pass

def test_too_many_args():
	with raises(GillespieUsageError):
		ProcessWithRateDimensionMismatch2()

