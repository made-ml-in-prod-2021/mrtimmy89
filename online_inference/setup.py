from setuptools import find_packages, setup


def parse_requirements(path):
    with open(path) as fin:
        requirements = []
        for line in fin:
            requirements.append(line.strip())
    return requirements


setup(
    name="homework2",
    packages=find_packages(),
    version="0.1.0",
    description="Heart disease classifier (a.k.a. 'Homework2')",
    author="Artem Akopian",
    install_requires=parse_requirements("./requirements.txt"),
    license="MIT",
)
