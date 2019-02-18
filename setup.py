from setuptools import setup, find_packages

VERSION = '3.2.1'
DISTNAME = 'test-harness'
DESCRIPTION = 'A tool for collaboration on models and comparison of model results.'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Hamed'
MAINTAINER_EMAIL = 'eramian@netrias.com'
# DOWNLOAD_URL = ''
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    packages=['harness'] + ['harness/' + s for s in find_packages('harness')],
    include_package_data=True,
    install_requires=requirements
)
