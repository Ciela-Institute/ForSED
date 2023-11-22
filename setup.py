from setuptools import setup, find_packages
import lighthouse.__init__ as lh
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
def read_lines(fname):
    ret = list(open(os.path.join(os.path.dirname(__file__), fname)).readlines())
    return ret

setup(
    name="lighthouse",
    version=lh.__version__,    
    description="A differentiable SED modelling tool for the future",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url="https://github.com/Ciela-Institute/Light-House",
    author=lh.__author__,
    author_email=lh.__email__,
    license="MIT license",
    packages=find_packages(),
    install_requires=read_lines("requirements.txt"),
    keywords = [
        "SED modelling",
        "astrophysics",
        "differentiable programming",
        "pytorch",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
