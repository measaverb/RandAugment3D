from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

_VERSION = '0.1'

REQUIRED_PACKAGES = [
    'torch',
    'tqdm',
    'plyfile'
]

DEPENDENCY_LINKS = [
]

setuptools.setup(
    name='RandAugment3D',
    version=_VERSION,
    description="Unofficial PyTorch modification of RandAugment by ildoonet's RandAugment",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url='https://github.com/kim-younghan/RandAugment3D',
    license='MIT License',
    include_package_data=True,
    package_dir={},
    packages=setuptools.find_packages(exclude=['tests']),
)