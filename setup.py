"""setup.py"""

import pkg_resources
import setuptools # type: ignore

pkg_resources.require(['pip >= 21.3.1'])

with open("README.md", "r") as f:
    long_description = f.read()

DESCRIPTION = "Image augmentation library"
NAME = 'imgaug'
AUTHOR = 'Urasaki Keisuke'
AUTHOR_EMAIL = 'urasakikeisuke.ml@gmail.com'
URL = 'https://github.com/urasakikeisuke/imgaug.git'
LICENSE = 'MIT License'
DOWNLOAD_URL = 'https://github.com/urasakikeisuke/imgaug.git'
VERSION = "1.0.0"
PYTHON_REQUIRES = ">=3.6"
INSTALL_REQUIRES = [
    "typing_extensions",
    "numpy",
    "opencv-python",
]
PACKAGES = setuptools.find_packages()
CLASSIFIERS = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

setuptools.setup(
    name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
)