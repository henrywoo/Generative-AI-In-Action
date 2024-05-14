from setuptools import setup
import os

here = os.path.dirname(os.path.realpath(__file__))
HAS_CUDA = os.system("nvidia-smi > /dev/null 2>&1") == 0

VERSION = (
    "1.1.12"
    if "PKG_VERSION" not in os.environ or not os.environ["PKG_VERSION"]
    else os.environ["PKG_VERSION"]
)
DESCRIPTION = "nanoT5 - T5-style models in nano shape"

packages = ["nanoT5"]


def read_file(filename: str):
    try:
        lines = []
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines if not line.startswith("#")]
        return lines
    except:
        return []


def package_files(ds):
    paths = []
    for d in ds:
        for path, directories, filenames in os.walk(d):
            for filename in filenames:
                if "__pycache__" not in str(filename):
                    paths.append(str(os.path.join(path, filename))[len("nanoT5/") :])
    return paths


extra_files = package_files(["nanoT5/"])

setup(
    name="nanoT5",
    version=VERSION,
    author="Henry Fuheng Wu",
    description=DESCRIPTION,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=read_file(f"{here}/requirements.txt"),
    keywords=[
        "nanoT5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=packages,
    include_package_data=True,
    package_data={"nanoT5": extra_files},
)