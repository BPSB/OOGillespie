#!/usr/bin/python3

from setuptools import setup
from io import open

setup(
		name = 'oogillespie',
		description = 'Gillespie algorithm.',
		long_description = open('README.rst', encoding='utf8').read(),
		python_requires=">=3.6",
		packages = ['oogillespie'],
		install_requires = ['numpy'],
		setup_requires = ['setuptools_scm','pytest-runner'],
		tests_require = ['pytest'],
		use_scm_version = {'write_to': 'oogillespie/version.py'},
		classifiers = [
				'License :: OSI Approved :: BSD License',
				'Operating System :: POSIX',
				'Operating System :: MacOS :: MacOS X',
				'Operating System :: Microsoft :: Windows',
				'Programming Language :: Python',
				'Topic :: Scientific/Engineering',
			],
	)

