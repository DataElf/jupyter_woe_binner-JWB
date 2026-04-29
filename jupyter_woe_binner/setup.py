#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name='jupyter_woe_binner',
    version='0.1.0',
    description='Interactive WOE binning tool for Jupyter',
    packages=find_packages(),
    install_requires=[
        'ipywidgets>=7.6',
        'plotly>=4.0',
        'pandas>=1.0',
        'numpy',
        'ipyevents>=0.9',
    ],
    python_requires='>=3.7',
)

