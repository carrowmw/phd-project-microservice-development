from setuptools import setup, find_packages


# Function to parse requirements.txt
def parse_requirements(filename):
    """
    Parse a requirements.txt file returning non-empty, non-comment lines
    as a list of strings.
    """
    with open(filename, "r", encoding="utf8") as file:
        return [line.strip() for line in file if line and not line.startswith("#")]


setup(
    name="phd_package",
    version="0.1.0",
    author="Carrow Morris-Wiltshire",
    author_email="c.morris-wiltshire@ncl.ac.uk",
    description="Python package for my PhD project",
    long_description=open("README.md", "r", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/carrowmw/phd-project",
    packages=find_packages(where="phd_package"),
    package_dir={"": "phd_package"},
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "run-dashboard=dashboard.__main__",
            "run-pipeline=pipeline.__main__",
            "run-api=api.__main__",
            "run-database=database.__main__",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10.13",
)
