#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ftr-benchmark",
    version="0.1.0",
    description="FTR Benchmark - Foothill Terrain Robot Benchmark",
    author="FTR Team",
    packages=find_packages(include=["ftr_algo", "ftr_envs", "scripts"]),
    install_requires=requirements,
    python_requires=">=3.8",
)
