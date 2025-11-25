from setuptools import setup, find_packages

# Read from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()
    install_requires = required

setup(
    name='ULTR-AI',
    version='0.1',
    author='Trevor Brokowski',
    author_email='',
    description='ULTR-AI',
    long_description='ULTR-AI',
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    install_requires=install_requires,
)