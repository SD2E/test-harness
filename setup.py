from setuptools import setup, find_packages

# with open("README", 'r') as f:
#     long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='test_harness',
    version='1.1',
    description='Test Harness and Leaderboard functionality for Protein Stability predictions',
    author='Hamed Eramian',
    author_email='eramian@netrias.com',
    packages=['test_harness'] + ['test_harness/' + s for s in find_packages('test_harness')],
    include_package_data=True,
    install_requires=requirements
)
