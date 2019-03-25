from setuptools import setup, find_packages
from os import path as p
import os
from harness.utils.get_project_root import get_project_root

version_file = open(os.path.join(get_project_root(), 'VERSION'))
VERSION = version_file.read().strip()
DISTNAME = 'test-harness'
DESCRIPTION = 'A tool for collaboration on models and comparison of model results.'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Hamed'
MAINTAINER_EMAIL = 'eramian@netrias.com'


def read(filename, parent=None):
    parent = (parent or __file__)

    try:
        with open(p.join(p.dirname(parent), filename)) as f:
            return f.read()
    except IOError:
        return ''


def parse_requirements(filename, parent=None):
    parent = (parent or __file__)
    filepath = p.join(p.dirname(parent), filename)
    content = read(filename, parent)

    for line_number, line in enumerate(content.splitlines(), 1):
        candidate = line.strip()

        if candidate.startswith('-r'):
            for item in parse_requirements(candidate[2:].strip(), filepath):
                yield item
        else:
            yield candidate


setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    packages=['harness'] + ['harness/' + s for s in find_packages('harness')],
    include_package_data=True,
    install_requires=list(parse_requirements('requirements.txt'))
)
