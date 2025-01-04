#!/usr/bin/python

from setuptools import setup, find_packages

setup(
    name='pygments-style-algfordmdark',
    description='Pygments style for Algorithms for Decision Making (dark mode).',
    long_description=open('README.md').read(),
    keywords='pygments algfordmdark style',
    packages=find_packages(),
    install_requires=['pygments>=2.2'],
    entry_points='''[pygments.styles]
                    algfordmdark=pygments_style_algfordmdark:AlgForDMStyle
					''',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Plugins',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
