from setuptools import Extension, setup

tensorbase_module = Extension(name="match.tensorbase", sources=["src/match/tensorbase/tensorbasemodule.c"])

setup(ext_modules=[tensorbase_module])
