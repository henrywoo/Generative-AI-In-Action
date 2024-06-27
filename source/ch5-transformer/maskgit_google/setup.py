from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.realpath(__file__))

VERSION = "0.0.1.dev0"
DESCRIPTION = "maskgit"

def read_file(filename: str):
    try:
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines if not line.startswith("#")]
        return lines
    except:
        return []

def package_files(directory, skip_patterns):
    paths = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, here)
            if not any(skip_pattern in relative_path for skip_pattern in skip_patterns):
                paths.append(relative_path)
    return paths

# Define directories and patterns to skip
directories = ["maskgit/"]
skip_patterns = ["__pycache__", ".pyc", ".pyo", ".git", ".bin", "pretrained_maskgit", "data", ".huggingface"]

# Get the list of extra files
extra_files = package_files("maskgit", skip_patterns)

setup(
    name="maskgit",
    version=VERSION,
    author="Shadow Walker",
    description=DESCRIPTION,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=read_file(f"{here}/requirements.txt"),
    keywords=[
        "maskgit",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
    packages=find_packages(include=['maskgit', 'maskgit.*']),
    package_data={
        'maskgit': extra_files,
    },
)
