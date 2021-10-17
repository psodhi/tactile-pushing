from setuptools import setup, find_packages

install_requires = [line.rstrip() for line in open("requirements.txt", "r")]

setup(
    name="pushestpy",
    version="0.1.0",
    description="Object pose tracking using tactile images during planar pushing",
    url="",
    author="Paloma Sodhi",
    author_email="psodhi@cs.cmu.edu",
    license="LICENSE",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    python_requires=">=3.6",
)