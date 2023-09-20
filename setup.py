from setuptools import setup

setup(
    name='wootils',
    version='0.0.1',    
    description='A Python package for doing things I do too often to re-type.',
    url='https://github.com/williamedge/wootils',
    author='William Edge',
    author_email='william.edge@uwa.edu.au',
    license='BSD 3-clause',
    packages=['wootils'],
    install_requires=['numpy',
                      'matplotlib'],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)