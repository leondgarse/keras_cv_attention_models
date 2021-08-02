from setuptools import find_packages
from setuptools import setup

setup(
    name="keras-attention-models",
    version="1.0.0",
    author="Leondgarse",
    author_email="leondgarse@google.com",
    url="https://github.com/leondgarse/keras_attention_models",
    description="keras attention models",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "tensorflow",
        "tensorflow-addons",
    ],
    packages=find_packages(),
    license="Apache 2.0",
)
