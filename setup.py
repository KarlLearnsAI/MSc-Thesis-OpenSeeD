from setuptools import setup, find_packages

setup(
    name="OpenSeeD",
    version="0.5",
    packages=find_packages(include=["openseed", "openseed.*", "tools", "tools.*"])
)