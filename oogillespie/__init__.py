from ._oogillespie import Gillespie, GillespieUsageError, Event

try:
	from .version import version as __version__
except ImportError:
	from warnings import warn
	warn('Failed to find (autogenerated) version.py. Do not worry about this unless you really need to know the version.')
