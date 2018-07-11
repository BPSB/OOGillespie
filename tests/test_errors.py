from oogillespie import Gillespie, GillespieUsageError, Event
from pytest import raises

class ProcessWithoutInitialise(Gillespie):
	pass

def test_no_initialise():
	with raises(GillespieUsageError) as exc:
		ProcessWithoutInitialise()
	assert "must overwrite" in exc.value.args[0]

class ProcessWithoutState(Gillespie):
	def initialise(self): pass
	
	@Event(1)
	def do_nothing(self): pass

def test_no_state():
	with raises(GillespieUsageError) as exc:
		for state in ProcessWithoutState():
			pass
	assert "must overwrite" in exc.value.args[0]

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
	with raises(GillespieUsageError) as exc:
		ProcessWithoutEvent()
	assert "No event defined." in exc.value.args[0]

class ProcessWithRateDimensionMismatch(ProcessWithoutEvent):
	@Event([1,2])
	def do_nothing(self): pass

def test_no_arg():
	with raises(GillespieUsageError) as exc:
		ProcessWithRateDimensionMismatch()
	assert "does not match" in exc.value.args[0]

class ProcessWithRateDimensionMismatch2(ProcessWithoutEvent):
	@Event([[1,2],[3,4]])
	def do_nothing(self,i,j,k): pass

def test_too_many_args():
	with raises(GillespieUsageError) as exc:
		ProcessWithRateDimensionMismatch2()
	assert "does not match" in exc.value.args[0]

def test_negative_rate():
	with raises(ValueError) as exc:
		class ProcessWithNegativeEventRate(ProcessWithoutEvent):
			@Event([0,2,-1])
			def do_nothing(self,i): pass
	assert "non-negative" in exc.value.args[0]

