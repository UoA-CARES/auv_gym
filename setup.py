from setuptools import setup, find_packages

setup(
    name='auv_gym',
    version='1.0.0',
    packages=find_packages(),
    url='',
    license='MIT',
    author='RL_TEAM',
    author_email='cares@aucklanduni.ac.nz',
    description='',
    install_requires=[
        'numpy==1.26.4',
        'opencv-contrib-python==4.6.0.66',
        'scipy==1.15.3',
        'pydantic==1.10.13',
        'matplotlib==3.10.1',
        'pandas==1.5.3',
    ],
)
