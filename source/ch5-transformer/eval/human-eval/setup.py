import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="human-eval",
    version="1.0.0",  # Choose an appropriate version
    author="OpenAI",
    author_email=" ", # You can provide a contact email (optional)
    description="Code for the paper 'Evaluating Large Language Models Trained on Code'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openai/human-eval",
    project_urls={
        "Bug Tracker": "https://github.com/openai/human-eval/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},   # Since your modules are directly in the root
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.7",  # Specify your supported Python versions
    # Include package data
    include_package_data=True,
    package_data={
        "human_eval": ["data/*.jsonl", "data/*.gz"]  # Include all .jsonl and .gz files
    },
    install_requires=[
        "fire",
        "numpy",
        "tqdm",
    ],  # Add any additional dependencies from requirements.txt
    entry_points={
        "console_scripts": [
            "evaluate_functional_correctness = human_eval.evaluate_functional_correctness:main",
        ]
    },
)
